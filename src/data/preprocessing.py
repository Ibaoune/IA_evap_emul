"""
preprocessing.py
================
All preprocessing steps as explicit functions.
Called from train.py after data loading.
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.ndimage import distance_transform_edt

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.utils import vprint, build_morocco_mask


# ==========================================================
# Year-based train / val / test split
# ==========================================================
def split_by_years(X_all, Y_all, times, cfg):
    """
    Split (T, ...) arrays into train / val / test by year.

    Returns:
        X_train, X_val, X_test
        Y_train, Y_val, Y_test
        T_train, T_val, T_test  (pd.DatetimeIndex)
    """
    years = times.year

    train_mask = np.isin(years, cfg.train_years)
    val_mask   = np.isin(years, cfg.val_years)
    test_mask  = np.isin(years, cfg.test_years)

    vprint(f"Train : {train_mask.sum()} steps  ({cfg.train_years[0]}–{cfg.train_years[-1]})")
    vprint(f"Val   : {val_mask.sum()} steps  ({cfg.val_years[0]}–{cfg.val_years[-1]})")
    vprint(f"Test  : {test_mask.sum()} steps  ({cfg.test_years[0]}–{cfg.test_years[-1]})")

    return (
        X_all[train_mask], X_all[val_mask], X_all[test_mask],
        Y_all[train_mask], Y_all[val_mask], Y_all[test_mask],
        times[train_mask], times[val_mask], times[test_mask],
    )


def _fill_nan_nearest(da):
    """Fill NaNs spatially using nearest valid neighbour, per time step."""
    values = da.values.copy()
    for t in range(values.shape[0]):
        sl       = values[t]
        nan_mask = np.isnan(sl)
        if not nan_mask.any():
            continue
        _, indices = distance_transform_edt(nan_mask, return_indices=True)
        sl[nan_mask] = sl[indices[0][nan_mask], indices[1][nan_mask]]
        values[t] = sl
    return da.copy(data=values)


# ==========================================================
# Morocco mask
# ==========================================================
def build_mask(lats, lons, shp_path):
    """
    Thin wrapper around utils.build_morocco_mask.
    Kept here so all preprocessing calls go through this module.
    """
    return build_morocco_mask(lats, lons, shp_path)


# ==========================================================
# Main point-wise preprocessing entry point
# ==========================================================
def preprocess_dataPointwise(cfg, X_all, Y_all, times, lats, lons, channel_names):
    """
    Full preprocessing pipeline for point-wise models (LSTM, MLP, XGB, MLR-RF):
      1. Build Morocco land mask
      2. Split by years

    Returns everything needed by PointWiseTrain.
    """
    T, nlat, nlon, nfeat = X_all.shape

    vprint("Building Morocco land mask...")
    morocco_mask = build_mask(lats, lons, cfg.shp_regions_path)

    vprint("Splitting data by year...")
    (
        X_train, X_val, X_test,
        Y_train, Y_val, Y_test,
        T_train, T_val, T_test,
    ) = split_by_years(X_all, Y_all, times, cfg)

    vprint(f"Spatial grid : {nlat} lat × {nlon} lon  |  valid Morocco points: {morocco_mask.sum()}")

    return (
        X_train, Y_train, T_train,
        X_val,   Y_val,   T_val,
        X_test,  Y_test,  T_test,
        channel_names, nlat, nlon, lats, lons, times, morocco_mask,
    )


# =============================================prerocessing function used in data_loading.py=========================
def regrid_to_fine(coarse_vars, evap_lmdz35):
    """Reindex all coarse predictors to the fine LMDZ35 grid (nearest neighbour)."""
    datasets = {}
    for name, da in coarse_vars.items():
        datasets[name] = da.reindex_like(evap_lmdz35, method="nearest").fillna(0)
        vprint(f"  {name} reindexed to {datasets[name].shape}")
    return datasets


def add_static_fields(datasets, evap_lmdz35, altitude_2d, soiltile_tiles):
    """
    Interpolate altitude and soiltile to fine grid and broadcast to (T, lat, lon).
    Adds 'altitude', 'soiltile_bare', 'soiltile_short', 'soiltile_tall' to datasets.
    """
    import xarray as xr
    import numpy as np

    T_all = len(evap_lmdz35.time_counter)

    # altitude
    if altitude_2d is not None:
        altitude_fine = altitude_2d.interp(
            lat=evap_lmdz35.lat.values, lon=evap_lmdz35.lon.values, method="linear"
        ).fillna(0)
        datasets["altitude"] = xr.DataArray(
            np.broadcast_to(altitude_fine.values[np.newaxis],
                            (T_all, *altitude_fine.shape)).copy().astype(np.float32),
            dims=["time_counter", "lat", "lon"],
            coords={"time_counter": evap_lmdz35.time_counter,
                    "lat": evap_lmdz35.lat, "lon": evap_lmdz35.lon},
            name="altitude"
        )

    # soiltiles
    if soiltile_tiles is not None:
        for name, tile_2d in soiltile_tiles.items():
            tile_fine = tile_2d.interp(
                lat=evap_lmdz35.lat.values,
                lon=evap_lmdz35.lon.values,
                method="linear"
            ).fillna(0)
            da = xr.DataArray(
                np.broadcast_to(
                    tile_fine.values[np.newaxis],
                    (T_all, *tile_fine.shape)
                ).copy().astype(np.float32),
                dims=["time_counter", "lat", "lon"],
                coords={"time_counter": evap_lmdz35.time_counter,
                        "lat": evap_lmdz35.lat, "lon": evap_lmdz35.lon},
                name=name
            )
            datasets[name] = _fill_nan_nearest(da)

    return datasets

def compute_net_radiation(ds_rad):
    """Compute net SW and LW radiation from up/down components."""
    swnet = (ds_rad["SWdnSFC"] - ds_rad["SWupSFC"]).rename("SWnet")
    lwnet = (ds_rad["LWdnSFC"] - ds_rad["LWupSFC"]).rename("LWnet")
    return swnet, lwnet


def convert_evap_units(evap_da):
    """Convert evaporation from kg/m2/s to mm/day and clip negatives."""
    return (evap_da * 86400).clip(min=0)