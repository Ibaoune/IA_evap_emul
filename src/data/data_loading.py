import os
import sys
import numpy as np
import xarray as xr
import rioxarray
import pandas as pd
from src.data.preprocessing import (
    regrid_to_fine, add_static_fields, _fill_nan_nearest,
    compute_net_radiation, convert_evap_units,
)

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.core.utils import vprint, mask_dataset, build_morocco_mask
from src.data.preprocessing import regrid_to_fine, add_static_fields, _fill_nan_nearest

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
<<<<<<< HEAD
    vprint("Skipping LMDZ250 radiation and humidity... (missing predictor)")
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

=======
    vprint("Loading LMDZ250 radiation and humidity...")
    ds_rad = xr.open_dataset(os.path.join(cfg.data_LMDZ250_path, "LMDZ_rad_q2m_surface.nc"))
    swnet, lwnet = compute_net_radiation(ds_rad)
    q2m = mask_dataset(ds_rad, "q2m")["q2m"].sortby("lat")
    swnet = mask_dataset(swnet.to_dataset(name="swnet"), "swnet")["swnet"].sortby("lat")
    lwnet = mask_dataset(lwnet.to_dataset(name="lwnet"), "lwnet")["lwnet"].sortby("lat")

    # load Temperature, pressure, wind
    vprint("Loading LMDZ250 temperature, pressure, wind...")
    t2m_ds  = _load_lmdz("t2m-hist1.nc",  "t2m",  cfg.data_LMDZ250_path)
    pres_ds = _load_lmdz("pres-hist2.nc", "pres", cfg.data_LMDZ250_path)
    u10m_ds = _load_lmdz("u10m-hist1.nc", "u10m", cfg.data_LMDZ250_path)
    v10m_ds = _load_lmdz("v10m-hist1.nc", "v10m", cfg.data_LMDZ250_path)

    # load LMDZ250 evaporation (coarse grid predictor)
    vprint("Loading LMDZ250 evaporation (predictor)...")
    evap250_ds = _load_lmdz("evap-hist1.nc", "evap", cfg.data_LMDZ250_path)
>>>>>>> origin/melaabar
    # load soil moisture variables
    mrso, mrsos = None, None
    if cfg.soil_moisture_path:
        vprint("Loading soil moisture...")
        mrso  = _load_soil(cfg, "mrso_day_IPSL-CM6A-LR_amip_r1i1p1f1_gr_19790101-20141231.nc",  "mrso")
        mrsos = _load_soil(cfg, "mrsos_day_IPSL-CM6A-LR_amip_r1i1p1f1_gr_19790101-20141231.nc", "mrsos")
    else:
        vprint("WARNING: soil_moisture_path is missing/commented. Skipping mrso and mrsos.")

    # load target evaporation
    vprint("Loading target evaporation...")
    evap_lmdz35 = xr.open_dataset(os.path.join(cfg.data_LMDZ35_path, "evap_1979_2014_MA.nc"))["evap"]
    evap_lmdz35 = mask_dataset(evap_lmdz35.to_dataset(), "evap")["evap"]
    evap_lmdz35 = evap_lmdz35.sortby("lat")

    # load albedo and LAI
    vprint("Loading albedo and LAI...")
    albedo_ds   = _load_lmdz("albedo_glob-hist1.nc", "albedo_glob", cfg.data_LMDZ35_path).fillna(0)
    mean_lai_ds = _load_lmdz("LAImean-hist1.nc",     "LAImean",     cfg.data_LMDZ35_path).fillna(0)

    altitude_2d = None
    if cfg.altitude_path and os.path.exists(os.path.join(cfg.altitude_path, "elevation.tif")):
        vprint("Loading altitude (raw DEM)...")
        altitude_raw = rioxarray.open_rasterio(os.path.join(cfg.altitude_path, "elevation.tif"))
        altitude_2d  = altitude_raw.squeeze("band", drop=True).rename({"x": "lon", "y": "lat"})
    else:
        vprint("WARNING: altitude_path missing or elevation.tif not found. Skipping altitude.")

    tsol = None
    if cfg.tsol_path:
        vprint("Loading soil temperature (tsol)...")
        tsol = mask_dataset(xr.open_dataset(os.path.join(cfg.tsol_path, "tsol_1979_2014.nc")), "tsol")["tsol"]
    else:
        vprint("WARNING: tsol_path is missing/commented. Skipping tsol.")

    coarse_vars = {
        "t2m":          t2m_ds,
<<<<<<< HEAD
        "q2m":          ds_rad["q2m"],
=======
        "q2m":          q2m,
>>>>>>> origin/melaabar
        "swnet":        swnet,
        "lwnet":        lwnet,
        "pres":         pres_ds,
        "u10m":         u10m_ds,
        "v10m":         v10m_ds,
        "evap250":      evap250_ds,
        "albedo":       albedo_ds,
        "lai":          mean_lai_ds,
    }
    
    if mrso is not None:
        coarse_vars["mrso"] = mrso
    if mrsos is not None:
        coarse_vars["mrsos"] = mrsos
    if tsol is not None:
        coarse_vars["tsol"] = tsol

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
    
    soiltile_tiles = None
    if cfg.soiltile_path:
        soiltile_tiles = _load_soiltile(cfg)
    else:
        vprint("WARNING: soiltile_path is missing/commented. Skipping soiltile.")

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