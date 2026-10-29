"""
postproc/cache.py
=================
On-disk cache of the computed results of one config, so figures can be
redrawn without reloading the predictions pickle or recomputing metrics.

Layout (postprocessing/results/single/<model>/<conf>/cache/):
    metrics_annual.csv    per-pixel metrics over the whole test period
    metrics_seasonal.csv  per-pixel metrics for each season
    series.nc             obs + pred test series on the valid Morocco points
    cache_info.yaml       where the cache comes from, used to detect staleness

The cache is stale when the predictions pickle or the shapefiles that
define the points/regions change. Plot settings never invalidate it.
"""

import os
import datetime
import subprocess

import pandas as pd
import xarray as xr
import yaml


CACHE_VERSION = 1

# cache_info.yaml keys compared to decide whether the cache is still valid
_SOURCE_KEYS = ["cache_version", "preds_size", "preds_mtime_ns",
                "shp_path", "shp_regions_path"]


def _source_info(cfg, preds_path):
    st = os.stat(preds_path)
    return {
        "cache_version":    CACHE_VERSION,
        "preds_file":       os.path.normpath(preds_path),
        "preds_size":       st.st_size,
        "preds_mtime_ns":   st.st_mtime_ns,
        "shp_path":         os.path.normpath(str(cfg.shp_path)),
        "shp_regions_path": os.path.normpath(str(cfg.shp_regions_path)),
    }


def _git_commit():
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


# ==========================================================
# Status / save / load
# ==========================================================

def cache_status(cache_dir, cfg, preds_path):
    """
    Returns:
        "ok"         : cache matches the current predictions pickle
        "stale"      : predictions or shapefiles changed since it was written
        "missing"    : no cache
        "unverified" : cache exists but the pickle is absent, so it cannot be checked
    """
    info_path = os.path.join(cache_dir, "cache_info.yaml")
    if not os.path.exists(info_path):
        return "missing"
    if not os.path.exists(preds_path):
        return "unverified"

    with open(info_path) as f:
        saved = yaml.safe_load(f) or {}
    current = _source_info(cfg, preds_path)
    same = all(saved.get(k) == current[k] for k in _SOURCE_KEYS)
    return "ok" if same else "stale"


def save_cache(cache_dir, cfg, preds_path, r):
    """Write the metrics tables, the stacked series and cache_info.yaml."""
    os.makedirs(cache_dir, exist_ok=True)

    r["df_annual"].to_csv(os.path.join(cache_dir, "metrics_annual.csv"),     index=False)
    r["df_seasonal"].to_csv(os.path.join(cache_dir, "metrics_seasonal.csv"), index=False)

    # The 'points' MultiIndex (lat, lon) cannot be written to NetCDF:
    # store lat/lon as plain coordinates and rebuild the index on load.
    ds = xr.Dataset({"obs": r["obs"], "pred": r["pred"]}).reset_index("points")
    ds.to_netcdf(os.path.join(cache_dir, "series.nc"))

    # cache_info.yaml is written last: it marks the cache as complete
    info = _source_info(cfg, preds_path)
    info["git_commit"] = _git_commit()
    info["created"]    = datetime.datetime.now().isoformat(timespec="seconds")
    with open(os.path.join(cache_dir, "cache_info.yaml"), "w") as f:
        yaml.safe_dump(info, f, sort_keys=False)


def load_cache(cache_dir):
    """Load what save_cache wrote, in the format of _stack_and_compute_metrics."""
    df_annual   = pd.read_csv(os.path.join(cache_dir, "metrics_annual.csv"))
    df_seasonal = pd.read_csv(os.path.join(cache_dir, "metrics_seasonal.csv"))

    with xr.open_dataset(os.path.join(cache_dir, "series.nc")) as ds:
        ds = ds.load()
    lats = ds["lat"].values
    lons = ds["lon"].values
    ds   = ds.set_index(points=["lat", "lon"])

    return {
        "df_annual":   df_annual,
        "df_seasonal": df_seasonal,
        "obs":         ds["obs"],
        "pred":        ds["pred"],
        "lats":        lats,
        "lons":        lons,
    }
