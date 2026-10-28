"""
postprocess_single.py
=====================
Run full evaluation (main + extreme) for ONE model config.

Usage:
    python postprocess_single.py --config configs/xgb/conf0.yaml

Outputs saved in:
    postprocessing/results/<model>/<conf>/main/
    postprocessing/results/<model>/<conf>/extreme/
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

from pathlib import Path

# Project root (for src/) and postprocessing/ (for the postproc package)
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "postprocessing"))

from src.core.config import load_config
from src.core.utils import vprint, set_verbose, build_morocco_mask


from postproc.utils   import get_valid_morocco_points, stack_on_points, get_region_point_indices
from postproc.metrics import (
    compute_metrics_df, compute_seasonal_metrics_df, add_region_labels
)
from postproc.main.plots_main import (
    plot_spatial_metrics, plot_boxplots_global, plot_boxplots_by_season,
    plot_boxplots_by_region, plot_et_distribution,
    plot_mean_et_map, plot_et_seasonal_maps,
)
from postproc.extreme.plots_extreme import (
    plot_spatial_extreme, plot_boxplots_extreme,
    plot_boxplots_extreme_by_season, plot_boxplots_extreme_by_region,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="Path to model config yaml, e.g. configs/xgb/conf0.yaml")
    args = parser.parse_args()

    cfg = load_config(config_path=args.config, train_mode=False)
    set_verbose(cfg.verbose)

    model_type = cfg.model_type
    model_name = model_type.upper()   # display label for plot titles
    conf_name  = cfg.conf_name

    # Output dirs
    #main
    postprocessing_dir = os.path.join(ROOT, "postprocessing")
    path_main         = os.path.join(postprocessing_dir, "results", model_type, conf_name, "main")
    path_metrics_dist = os.path.join(path_main, "metric_distribution")
    path_et_dist      = os.path.join(path_main, "et_distribution")
    os.makedirs(path_metrics_dist, exist_ok=True)
    os.makedirs(path_et_dist,      exist_ok=True)
    #extrem
    path_extreme         = os.path.join(postprocessing_dir, "results", model_type, conf_name, "extreme")
    path_extreme_metrics = os.path.join(path_extreme, "metric_distribution")
    os.makedirs(path_extreme_metrics, exist_ok=True)

    os.makedirs(path_main,    exist_ok=True)
    os.makedirs(path_extreme, exist_ok=True)

    vprint(f"=== Single-model postprocessing  |  {model_name}  {conf_name} ===")

    # ------------------------------------------------------------------
    # Load predictions pickle saved by training
    # ------------------------------------------------------------------
    preds_path = os.path.join(cfg.path_out, f"{cfg.model_type}_preds.pkl")
    with open(preds_path, "rb") as f:
        saved = pickle.load(f)

    model_preds      = saved["model_preds"]
    nlat, nlon       = saved["nlat"], saved["nlon"]
    lat_fine         = saved["lats"]
    lon_fine         = saved["lons"]

    time_test_list = sorted(set(
        t for entry in model_preds for t in entry["te_df"]["time"]
    ))
    time_index = {t: i for i, t in enumerate(time_test_list)}
    n_time     = len(time_test_list)
    time_test  = np.array(time_test_list)

    Y_pred    = np.full((n_time, nlat, nlon), np.nan)
    Y_test_np = np.full((n_time, nlat, nlon), np.nan)

    for entry in model_preds:
        te   = entry["te_df"]
        pred = entry["pred"]
        ti   = te["time"].map(time_index).values
        li   = int(te["lat_idx"].iloc[0])
        lo   = int(te["lon_idx"].iloc[0])

        n = min(len(ti), len(pred))
        Y_pred[ti[:n], li, lo]    = pred[:n]
        Y_test_np[ti[:n], li, lo] = te["target"].values[:n]
    
    morocco_mask_fine = build_morocco_mask(lat_fine, lon_fine, cfg.shp_path)
    Y_pred[:,    ~morocco_mask_fine] = np.nan
    Y_test_np[:, ~morocco_mask_fine] = np.nan

    # Create xr DataArrays
    pred_da = xr.DataArray(Y_pred, dims=["time","lat","lon"], coords={"time": time_test, "lat": lat_fine, "lon": lon_fine})
    obs_da  = xr.DataArray(Y_test_np, dims=["time","lat","lon"], coords={"time": time_test, "lat": lat_fine, "lon": lon_fine})
    
    # Save NC for postprocess_compare.py
    ds_out = xr.Dataset({"evap_pred": pred_da, "evap_obs": obs_da})
    ds_out["evap_pred"].attrs["units"] = "mm/day"
    ds_out["evap_obs"].attrs["units"]  = "mm/day"
    ds_out.attrs["model"] = model_name
    nc_path = os.path.join(cfg.path_out, f"{model_type}_ET_predictions.nc")
    ds_out.to_netcdf(nc_path)
    vprint(f"Saved NC file: {nc_path}")

    # ------------------------------------------------------------------
    # Valid Morocco points + stack
    # ------------------------------------------------------------------
    vprint("Building valid Morocco points...")
    valid_lats, valid_lons = get_valid_morocco_points(obs_da, cfg.shp_path)

    obs_stacked  = stack_on_points(obs_da,  valid_lats, valid_lons)
    pred_stacked = stack_on_points(pred_da, valid_lats, valid_lons)

    # ------------------------------------------------------------------
    # Region GeoDataFrames
    # ------------------------------------------------------------------
    shp_files   = [f for f in os.listdir(cfg.shp_path) if f.endswith(".shp")]
    gdf_morocco = gpd.read_file(os.path.join(cfg.shp_path, shp_files[0]))

    region_names = ["north", "north_east", "east", "south"]
    gdfs_regions = {}
    for rname in region_names:
        shp = os.path.join(cfg.shp_regions_path, f"{rname}.shp")
        if os.path.exists(shp):
            gdfs_regions[rname] = gpd.read_file(shp)

    # ------------------------------------------------------------------
    # Compute per-pixel metrics
    # ------------------------------------------------------------------
    vprint("Computing per-pixel metrics (annual)...")
    df_annual = compute_metrics_df(
        obs_stacked, pred_stacked, valid_lats, valid_lons, model_name
    )
    df_annual = add_region_labels(df_annual, gdfs_regions)

    vprint("Computing per-pixel metrics (seasonal)...")
    df_seasonal = compute_seasonal_metrics_df(
        obs_stacked, pred_stacked, valid_lats, valid_lons, model_name
    )
    df_seasonal = add_region_labels(df_seasonal, gdfs_regions)

    # Save metrics CSVs
    df_annual.to_csv(os.path.join(path_main, "metrics_annual.csv"),   index=False)
    df_seasonal.to_csv(os.path.join(path_main, "metrics_seasonal.csv"), index=False)
    vprint("  Metrics CSVs saved.")

    # ------------------------------------------------------------------
    # MAIN plots
    # ------------------------------------------------------------------
    vprint("Generating main evaluation plots...")
    plot_spatial_metrics(df_annual,   model_name, gdf_morocco, path_metrics_dist)
    plot_boxplots_global(df_annual,   model_name, path_metrics_dist)
    plot_boxplots_by_season(df_seasonal, model_name, path_metrics_dist)
    plot_boxplots_by_region(df_annual,   model_name, path_metrics_dist)

    plot_et_distribution(obs_stacked, pred_stacked, valid_lats, valid_lons,
                        gdfs_regions, model_name, path_et_dist)
    plot_mean_et_map(obs_stacked, pred_stacked, valid_lats, valid_lons,
                    model_name, gdf_morocco, path_et_dist)
    plot_et_seasonal_maps(obs_stacked, pred_stacked, valid_lats, valid_lons,
                        model_name, gdf_morocco, path_et_dist)
    # ------------------------------------------------------------------
    # EXTREME plots
    # ------------------------------------------------------------------
    vprint("Generating extreme evaluation plots...")
    plot_spatial_extreme(df_annual,            model_name, gdf_morocco, path_extreme_metrics)
    plot_boxplots_extreme(df_annual,           model_name, path_extreme_metrics)
    plot_boxplots_extreme_by_season(df_seasonal, model_name, path_extreme_metrics)
    plot_boxplots_extreme_by_region(df_annual,   model_name, path_extreme_metrics)
        

    vprint(f"=== Done  |  main → {path_main}  |  extreme → {path_extreme} ===")


if __name__ == "__main__":
    main()
