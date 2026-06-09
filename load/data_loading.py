import os
import sys
import numpy as np
import xarray as xr
import rioxarray
import pandas as pd
from preprocessing.preprocessing import (
    regrid_to_fine, add_static_fields, _fill_nan_nearest,
    compute_net_radiation, convert_evap_units,
)

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import vprint, mask_dataset, build_morocco_mask
from preprocessing.preprocessing import regrid_to_fine, add_static_fields, _fill_nan_nearest

# -------------------------------------------------------
#  Low-level helpers
# -------------------------------------------------------

def _load_lmdz(filename, varname, path):
    ds  = xr.open_dataset(os.path.join(path, filename))
    var = mask_dataset(ds, varname)[varname]
    return var.sortby("lat")


def _load_soil(cfg, fname, varname):
    da = xr.open_dataset(os.path.join(cfg.soil_moisture_path, fname))[varname]
    da = da.assign_coords(lon=(((da.lon + 180) % 360) - 180)).sortby("lon")
    da = mask_dataset(da, varname)
    da = _fill_nan_nearest(da)
    return da.rename({"time": "time_counter"})


def _load_soiltile(cfg):
    """
    Load soiltile.nc and return 3 raw 2-D DataArrays
    (bare, short, tall vegetation) on the original soiltile grid.
    """
    ds_soil = xr.open_dataset(os.path.join(cfg.soiltile_path, "soiltile.nc"))
    ds_soil = ds_soil.assign_coords(
        lon=(((ds_soil.lon + 180) % 360) - 180)
    ).sortby("lon")

    soiltile = ds_soil["soiltile"]   # (soiltyp, lat, lon)

    tiles = {}
    for soiltyp_val, name in [
        (1.0, "soiltile_bare"),
        (2.0, "soiltile_short"),
        (3.0, "soiltile_tall"),
    ]:
        tiles[name] = soiltile.sel(soiltyp=soiltyp_val).drop_vars("soiltyp")  # (lat, lon)
    return tiles


# -------------------------------------------------------
#  Shared loader
# -------------------------------------------------------
def _load_variables(cfg):
    """
    Load all raw variables (coarse grid) + fine-grid target.

    Returns:
        coarse_vars  : dict of xr.DataArray  (all on LMDZ250 coarse grid)
        evap_lmdz35  : xr.DataArray          (fine grid — the target)
        altitude_2d  : xr.DataArray          (raw DEM, not yet interpolated)
    """
    # load Radiation and humidity variable
    vprint("Loading LMDZ250 radiation and humidity...")
    ds_rad = xr.open_dataset(os.path.join(cfg.data_LMDZ250_path, "LMDZ_rad_q2m_surface.nc"))
    ds_rad = mask_dataset(ds_rad, "q2m_radiation").sortby("lat")
    swnet, lwnet = compute_net_radiation(ds_rad)

    # load Temperature, pressure, wind
    vprint("Loading LMDZ250 temperature, pressure, wind...")
    t2m_ds  = _load_lmdz("t2m-hist1.nc",   "t2m",  cfg.data_LMDZ250_path)
    pres_ds = _load_lmdz("pres-hist2.nc",   "pres", cfg.data_LMDZ250_path)
    u10m_ds = _load_lmdz("u10m-hist1.nc",   "u10m", cfg.data_LMDZ250_path)
    v10m_ds = _load_lmdz("v10m-hist1.nc",   "v10m", cfg.data_LMDZ250_path)

    # load LMDZ250 evaporation (coarse grid predictor)
    vprint("Loading LMDZ250 evaporation (coarse grid predictor)...")
    evap_lmdz250 = mask_dataset(
        xr.open_dataset(os.path.join(cfg.data_LMDZ250_path, "evap-hist1.nc")), "evap"
    )["evap"]
    evap_lmdz250 = convert_evap_units(evap_lmdz250.sortby("lat"))

    # load soil moisture variables
    vprint("Loading soil moisture...")
    mrso  = _load_soil(cfg, "mrso.nc",  "mrso")
    mrsos = _load_soil(cfg, "mrsos.nc", "mrsos")

    # load LMDZ35 evaporation (fine grid, the target)
    vprint("Loading LMDZ35 target + ancillary...")
    evap_lmdz35 = mask_dataset(
        xr.open_dataset(os.path.join(cfg.data_LMDZ35_path, "evap_1979_2014_MA.nc")), "evap"
    )["evap"]
    evap_lmdz35 = convert_evap_units(evap_lmdz35.sortby("lat"))

    # load albedo and LAI
    vprint("Loading albedo and LAI...")
    albedo_ds   = _load_lmdz("albedo_glob-hist1.nc", "albedo_glob", cfg.data_LMDZ35_path).fillna(0)
    mean_lai_ds = _load_lmdz("LAImean-hist1.nc",     "LAImean",     cfg.data_LMDZ35_path).fillna(0)

    vprint("Loading altitude (raw DEM)...")
    altitude_raw = rioxarray.open_rasterio(os.path.join(cfg.altitude_path, "elevation.tif"))
    altitude_2d  = altitude_raw.squeeze("band", drop=True).rename({"x": "lon", "y": "lat"})

    vprint("Loading soil temperature (tsol)...")
    tsol = mask_dataset(xr.open_dataset(os.path.join(cfg.tsol_path, "tsol.nc")), "tsol")["tsol"]

    coarse_vars = {
        "t2m":          t2m_ds,
        "q2m":          ds_rad["q2m"],
        "swnet":        swnet,
        "lwnet":        lwnet,
        "pres":         pres_ds,
        "u10m":         u10m_ds,
        "v10m":         v10m_ds,
        "evap_lmdz250": evap_lmdz250,
        "albedo":       albedo_ds,
        "lai":          mean_lai_ds,
        "mrso":         mrso,
        "mrsos":        mrsos,
        "tsol":         tsol,
    }

    # Sort lat ascending for all variables (ensures consistent grid alignment)
    for name, da in coarse_vars.items():
        if "lat" in da.dims:
            coarse_vars[name] = da.sortby("lat")
        elif "latitude" in da.dims:
            coarse_vars[name] = da.sortby("latitude")

    return coarse_vars, evap_lmdz35, altitude_2d


# -------------------------------------------------------
#  Fine-grid loader  (LSTM / MLP / XGB / MLR-RF)
# -------------------------------------------------------
def load_datasets_fine(cfg):
    vprint("=== Loading datasets (fine grid) ===")

    coarse_vars, evap_lmdz35, altitude_2d = _load_variables(cfg)
    soiltile_tiles = _load_soiltile(cfg)

    vprint("Reindexing predictors to fine LMDZ35 grid (nearest)...")
    datasets = regrid_to_fine(coarse_vars, evap_lmdz35)

    vprint("Interpolating altitude and soiltile to fine grid...")
    datasets = add_static_fields(datasets, evap_lmdz35, altitude_2d, soiltile_tiles)

    channel_names = list(datasets.keys())
    vprint("Stacking predictors into arrays...")
    X_all = np.stack([datasets[k].values for k in channel_names], axis=-1)
    Y_all = evap_lmdz35.values
    times = pd.to_datetime(evap_lmdz35.time_counter.values)
    lats  = evap_lmdz35["lat"].values
    lons  = evap_lmdz35["lon"].values

    vprint(f"Fine grid: {len(lats)} lat x {len(lons)} lon  |  Channels: {channel_names}")
    vprint("=== Finished loading datasets ===")
    return (X_all, Y_all, times, lats, lons, channel_names)