"""
postprocess_single.py
=====================
Run full evaluation (main + extreme) for ONE model config.

Usage:
    python postprocess_single.py --config configs/XGB/conf0.yaml

Outputs saved in:
    results/main/<model>/<conf>/
    results/extreme/<model>/<conf>/
"""

import os
import sys
import argparse
import pickle
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import sys
from pathlib import Path

# Go up one directory (to the project root)
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

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
                        help="Path to model config yaml, e.g. configs/XGB/conf0.yaml")
    args = parser.parse_args()

    cfg = load_config(config_path=args.config, train_mode=False)
    set_verbose(cfg.verbose)

    model_name = cfg.model_type.upper()
    conf_name  = cfg.conf_name

    # Output dirs
    #main
    postprocessing_dir = os.path.dirname(os.path.abspath(__file__))
    path_main         = os.path.join(postprocessing_dir, "results", model_name, conf_name, "main")
    path_metrics_dist = os.path.join(path_main, "metric_distribution")
    path_et_dist      = os.path.join(path_main, "et_distribution")
    os.makedirs(path_metrics_dist, exist_ok=True)
    os.makedirs(path_et_dist,      exist_ok=True)
    #extrem
    path_extreme         = os.path.join(postprocessing_dir, "results", model_name, conf_name, "extreme")
    path_extreme_metrics = os.path.join(path_extreme, "metric_distribution")
    os.makedirs(path_extreme_metrics, exist_ok=True)

    os.makedirs(path_main,    exist_ok=True)
    os.makedirs(path_extreme, exist_ok=True)

    vprint(f"=== Single-model postprocessing  |  {model_name}  {conf_name} ===")

    # ------------------------------------------------------------------
    # Load NC predictions saved by training
    # ------------------------------------------------------------------
    nc_path = os.path.join(cfg.path_out, f"{model_name}_ET_predictions.nc")
    ds      = xr.open_dataset(nc_path)
    pred_da = ds["evap_pred"]   # (time, lat, lon)
    obs_da  = ds["evap_obs"]    # (time, lat, lon)

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
