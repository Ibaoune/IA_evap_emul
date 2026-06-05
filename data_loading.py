import os
import numpy as np
import xarray as xr
import rioxarray
from scipy.ndimage import distance_transform_edt
import pandas as pd
from utils import vprint, mask_dataset, build_morocco_mask


# -------------------------------------------------------
#  Low-level helpers
# -------------------------------------------------------

def _load_lmdz(filename, varname, path):
    ds  = xr.open_dataset(os.path.join(path, filename))
    var = mask_dataset(ds, varname)[varname]
    return var.sortby("lat")


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
#  Shared loader, called by both coarse and fine
# -------------------------------------------------------
def _load_variables(cfg):
    """
    Load all raw variables (coarse grid) + fine-grid target.

    Returns:
        coarse_vars  : dict of xr.DataArray  (all on LMDZ250 coarse grid)
        evap_lmdz35  : xr.DataArray          (fine grid — the target)
        t2m_ds       : xr.DataArray          (reference coarse grid)
        altitude_2d  : xr.DataArray          (raw DEM, not yet interpolated)
    """


    # load Radiation and humidity variable, extract net radiation components
    vprint("Loading LMDZ250 radiation and humidity...")
    ds_rad = xr.open_dataset(os.path.join(cfg.data_LMDZ250_path, "LMDZ_rad_q2m_surface.nc"))
    ds_rad = mask_dataset(ds_rad, "q2m_radiation").sortby("lat")
    swnet = (ds_rad["SWdnSFC"] - ds_rad["SWupSFC"]).rename("SWnet")
    lwnet = (ds_rad["LWdnSFC"] - ds_rad["LWupSFC"]).rename("LWnet")

    # load Temperature, pressure, wind
    vprint("Loading LMDZ250 temperature, pressure, wind...")
    t2m_ds       = _load_lmdz("t2m-hist1.nc",   "t2m",  cfg.data_LMDZ250_path)
    pres_ds      = _load_lmdz("pres-hist2.nc",   "pres", cfg.data_LMDZ250_path)
    u10m_ds      = _load_lmdz("u10m-hist1.nc",   "u10m", cfg.data_LMDZ250_path)
    v10m_ds      = _load_lmdz("v10m-hist1.nc",   "v10m", cfg.data_LMDZ250_path)

    # load LMDZ250 evaporation and transform it to mm/day (coarse grid), replace ET<0 with 0
    vprint("Loading LMDZ250 evaporation (coarse grid predictor)...")
    evap_lmdz250 = mask_dataset(
        xr.open_dataset(os.path.join(cfg.data_LMDZ250_path, "evap-hist1.nc")), "evap"
    )["evap"] * 86400
    evap_lmdz250 = evap_lmdz250.sortby("lat").clip(min=0)

    # load soil moisture variables
    vprint("Loading soil moisture...")
    mrso  = _load_soil(cfg, "mrso.nc",  "mrso")
    mrsos = _load_soil(cfg, "mrsos.nc", "mrsos")

    # load LMDZ35 evaporation (fine grid, the target), replace ET<0 with 0
    vprint("Loading LMDZ35 target + ancillary...")
    evap_lmdz35 = mask_dataset(
        xr.open_dataset(os.path.join(cfg.data_LMDZ35_path, "evap_1979_2014_MA.nc")), "evap"
    )["evap"] * 86400
    evap_lmdz35 = evap_lmdz35.sortby("lat").clip(min=0)

    # load and reindex albedo and LAI to coarse grid, replace filling NaNs with 0
    vprint("load and reindex albedo and LAI to coarse grid, replace filling NaNs with 0..")
    albedo_ds   = _load_lmdz("albedo_glob-hist1.nc", "albedo_glob", cfg.data_LMDZ35_path).fillna(0)
    mean_lai_ds = _load_lmdz("LAImean-hist1.nc",     "LAImean",     cfg.data_LMDZ35_path).fillna(0)

    vprint("Loading altitude (raw DEM)...")
    altitude_raw = rioxarray.open_rasterio(os.path.join(cfg.altitude_path, "elevation.tif"))
    altitude_2d  = altitude_raw.squeeze("band", drop=True).rename({"x": "lon", "y": "lat"})

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
    }

    return coarse_vars, evap_lmdz35, altitude_2d

# -------------------------------------------------------
#  LSTM / MLP loader, fine grid
# -------------------------------------------------------
def load_datasets_fine(cfg):
    """
    For LSTM / MLP: all predictors reindexed to LMDZ35 fine grid.
    
    return flat Morocco-filtered DataFrames for point-wise models (LSTM, MLP).
    """
    vprint("=== Loading datasets (fine grid / LSTM-MLP) ===")

    coarse_vars, evap_lmdz35, altitude_2d = _load_variables(cfg)

    # Reindex all coarse predictors to fine grid
    vprint("Reindexing predictors to fine LMDZ35 grid (with nearest)...")
    datasets = {}
    for name, da in coarse_vars.items():
        datasets[name] = da.reindex_like(evap_lmdz35, method="nearest").fillna(0)
        vprint(f"  {name} reindexed to {datasets[name].shape}")

    # Interpolate altitude to fine grid
    vprint("Interpolating altitude to fine grid...")
    altitude_fine = altitude_2d.interp(
        lat=evap_lmdz35.lat.values, lon=evap_lmdz35.lon.values, method="linear"
    ).fillna(0)
    T_all = len(evap_lmdz35.time_counter)
    datasets["altitude"] = xr.DataArray(
        np.broadcast_to(altitude_fine.values[np.newaxis],
                        (T_all, *altitude_fine.shape)).copy().astype(np.float32),
        dims=["time_counter", "lat", "lon"],
        coords={"time_counter": evap_lmdz35.time_counter,
                "lat": evap_lmdz35.lat, "lon": evap_lmdz35.lon},
        name="altitude"
    )

    vprint("Loading and interpolating soiltile to fine grid...")
    soiltile_tiles = _load_soiltile(cfg)
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
            coords={
                "time_counter": evap_lmdz35.time_counter,
                "lat":          evap_lmdz35.lat,
                "lon":          evap_lmdz35.lon,
            },
            name=name
        )
        datasets[name] = _fill_nan_nearest(da)


    channel_names = list(datasets.keys())
    vprint("Stacking predictors into arrays...")
    X_all       = np.stack([datasets[k].values for k in channel_names], axis=-1)  # (T, nlat, nlon, C)
    Y_all       = evap_lmdz35.values                                                # (T, nlat, nlon)
    times = pd.to_datetime(evap_lmdz35.time_counter.values)

    lats = evap_lmdz35["lat"].values
    lons = evap_lmdz35["lon"].values

    vprint(f"Fine grid: {len(evap_lmdz35.lat)} lat x {len(evap_lmdz35.lon)} lon")
    vprint(f"Channels : {channel_names}")
    vprint("=== Finished loading datasets (fine) ===")
    return (X_all, Y_all, times, lats, lons, channel_names)
