"""
postprocess_compare.py
======================
Intercomparison across multiple trained model configs.

Usage:
    # Compare specific configs
    python postprocess_compare.py \
        --configs configs/xgb/conf0.yaml configs/xgb/conf1.yaml configs/lstm/conf0.yaml

    # Compare ALL configs of one model
    python postprocess_compare.py --model xgb

    # Compare ALL trained configs across all models
    python postprocess_compare.py --all

Outputs saved in:
    postprocessing/results/intercomparison/<run_label>/
"""

import os
import sys
import argparse
import glob
import numpy as np
import xarray as xr
import geopandas as gpd

from pathlib import Path

# Project root (for src/) and postprocessing/ (for the postproc package)
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "postprocessing"))

from src.core.config import load_config
from src.core.utils import vprint, set_verbose, build_morocco_mask

from postproc.utils   import get_valid_morocco_points, stack_on_points
from postproc.metrics import compute_seasonal_metrics_df, compute_metrics_df, add_region_labels
from postproc.intercomparison.plots_intercomp import (
    plot_intercomparison_by_region,
    plot_spatial_metrics_intercomp,
    plot_boxplots_global_intercomp,
    plot_boxplots_by_season_intercomp,
    plot_boxplots_by_region_intercomp,
    plot_mean_et_map_intercomp,
    plot_et_seasonal_maps_intercomp,
    plot_et_boxplots_intercomp,
    DEFAULT_COLORS,
)



def _load_all_for_config(config_path):
    """Load metrics df + raw stacked DataArrays for one config."""
    cfg        = load_config(config_path=config_path, train_mode=False)
    model_name = cfg.model_type.upper()
    label      = f"{model_name} {cfg.conf_name}"

    nc_path = os.path.join(cfg.path_out, f"{cfg.model_type}_ET_predictions.nc")
    if not os.path.exists(nc_path):
        print(f"  [SKIP] NC not found: {nc_path}")
        return None

    ds      = xr.open_dataset(nc_path)
    pred_da = ds["evap_pred"]
    obs_da  = ds["evap_obs"]

    valid_lats, valid_lons = get_valid_morocco_points(obs_da, cfg.shp_path)
    obs_stacked  = stack_on_points(obs_da,  valid_lats, valid_lons)
    pred_stacked = stack_on_points(pred_da, valid_lats, valid_lons)

    region_names = ["north", "north_east", "east", "south"]
    gdfs_regions = {}
    for rname in region_names:
        shp = os.path.join(cfg.shp_regions_path, f"{rname}.shp")
        if os.path.exists(shp):
            gdfs_regions[rname] = gpd.read_file(shp)

    df_annual   = compute_metrics_df(obs_stacked, pred_stacked,
                                     valid_lats, valid_lons, label)
    df_annual   = add_region_labels(df_annual, gdfs_regions)

    df_seasonal = compute_seasonal_metrics_df(obs_stacked, pred_stacked,
                                              valid_lats, valid_lons, label)
    df_seasonal = add_region_labels(df_seasonal, gdfs_regions)

    print(f"  Loaded: {label}  ({len(df_annual)} pixels)")
    return {
        "label":       label,
        "cfg":         cfg,
        "df_annual":   df_annual,
        "df_seasonal": df_seasonal,
        "obs":         obs_stacked,
        "pred":        pred_stacked,
        "lats":        valid_lats,
        "lons":        valid_lons,
        "gdfs_regions": gdfs_regions,
    }


def main():
    parser = argparse.ArgumentParser()
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--configs", nargs="+")
    group.add_argument("--model")
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.configs:
        config_paths = args.configs
    elif args.model:
        config_paths = sorted(glob.glob(
            os.path.join("configs", args.model.lower(), "*.yaml")
        ))
    elif args.all:
        config_paths = sorted(glob.glob(os.path.join("configs", "*", "*.yaml")))
        config_paths = [p for p in config_paths if "common" not in p]

    if not config_paths:
        print("No config files found.")
        sys.exit(1)

    # Load everything
    results = [_load_all_for_config(cp) for cp in config_paths]
    results = [r for r in results if r is not None]

    if not results:
        print("No valid predictions found.")
        sys.exit(1)

    labels       = [r["label"]       for r in results]
    dfs          = [r["df_annual"]   for r in results]
    dfs_seasonal = [r["df_seasonal"] for r in results]
    obs_list     = [r["obs"]         for r in results]
    pred_list    = [r["pred"]        for r in results]
    lats_list    = [r["lats"]        for r in results]
    lons_list    = [r["lons"]        for r in results]
    gdfs_regions = results[0]["gdfs_regions"]
    first_cfg    = results[0]["cfg"]

    colors = DEFAULT_COLORS[:len(labels)]

    # Output dir
    run_label = "_vs_".join(l.replace(" ", "_") for l in labels).lower()
    postprocessing_dir = os.path.join(ROOT, "postprocessing")
    path_out = os.path.join(postprocessing_dir, "results", "intercomparison", run_label)
    os.makedirs(path_out, exist_ok=True)
    print(f"\nOutputs → {path_out}")

    # Morocco GDF
    shp_files   = [f for f in os.listdir(first_cfg.shp_path) if f.endswith(".shp")]
    gdf_morocco = gpd.read_file(os.path.join(first_cfg.shp_path, shp_files[0]))

    # ── Bar plots (region intercomparison)
    plot_intercomparison_by_region(dfs, labels, path_out, colors)

    # ── Metric spatial maps
    plot_spatial_metrics_intercomp(dfs, labels, gdf_morocco, path_out)

    # ── Metric boxplots
    plot_boxplots_global_intercomp(dfs, labels, colors, path_out)
    plot_boxplots_by_season_intercomp(dfs_seasonal, labels, colors, path_out)
    plot_boxplots_by_region_intercomp(dfs, labels, colors, path_out)

    # ── ET maps
    plot_mean_et_map_intercomp(obs_list, pred_list, lats_list, lons_list,
                                labels, gdf_morocco, path_out)
    plot_et_seasonal_maps_intercomp(obs_list, pred_list, lats_list, lons_list,
                                     labels, gdf_morocco, path_out)

    # ── ET boxplots
    plot_et_boxplots_intercomp(obs_list, pred_list, lats_list, lons_list,
                                labels, colors, gdfs_regions, path_out)

    print(f"\n=== Intercomparison done  |  {path_out} ===")


if __name__ == "__main__":
    main()
