"""
postprocess.py
==============
Postprocessing of trained models, in two modes:

Predictions are always read from the training pickle
results/<model>/<conf>/<model>_preds.pkl (no intermediate file needed).

1. Single-model evaluation (main + extreme) for ONE config.

       python postprocess.py --config configs/xgb/conf0.yaml

   Add --save-nc to also export the gridded predictions as
   results/<model>/<conf>/<model>_ET_predictions.nc

   Outputs saved in:
       postprocessing/results/<model>/<conf>/main/
       postprocessing/results/<model>/<conf>/extreme/

2. Intercomparison across multiple trained configs.

       # Compare specific configs
       python postprocess.py \
           --configs configs/xgb/conf0.yaml configs/xgb/conf1.yaml configs/lstm/conf0.yaml

       # Compare ALL configs of one model
       python postprocess.py --model xgb

       # Compare ALL trained configs across all models
       python postprocess.py --all

   Outputs saved in:
       postprocessing/results/intercomparison/<run_label>/
"""

import os
import sys
import argparse
import glob
import pickle
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

POSTPROCESSING_DIR = os.path.join(ROOT, "postprocessing")
REGION_NAMES       = ["north", "north_east", "east", "south"]


# ======================================================================
# Shared helpers
# ======================================================================

def _load_morocco_gdf(cfg):
    shp_files = [f for f in os.listdir(cfg.shp_path) if f.endswith(".shp")]
    return gpd.read_file(os.path.join(cfg.shp_path, shp_files[0]))


def _load_region_gdfs(cfg):
    gdfs_regions = {}
    for rname in REGION_NAMES:
        shp = os.path.join(cfg.shp_regions_path, f"{rname}.shp")
        if os.path.exists(shp):
            gdfs_regions[rname] = gpd.read_file(shp)
    return gdfs_regions


def _stack_and_compute_metrics(cfg, obs_da, pred_da, label):
    """Stack obs/pred on valid Morocco points and compute annual + seasonal metrics."""
    valid_lats, valid_lons = get_valid_morocco_points(obs_da, cfg.shp_path)
    obs_stacked  = stack_on_points(obs_da,  valid_lats, valid_lons)
    pred_stacked = stack_on_points(pred_da, valid_lats, valid_lons)

    gdfs_regions = _load_region_gdfs(cfg)

    df_annual = compute_metrics_df(
        obs_stacked, pred_stacked, valid_lats, valid_lons, label
    )
    df_annual = add_region_labels(df_annual, gdfs_regions)

    df_seasonal = compute_seasonal_metrics_df(
        obs_stacked, pred_stacked, valid_lats, valid_lons, label
    )
    df_seasonal = add_region_labels(df_seasonal, gdfs_regions)

    return {
        "label":        label,
        "cfg":          cfg,
        "df_annual":    df_annual,
        "df_seasonal":  df_seasonal,
        "obs":          obs_stacked,
        "pred":         pred_stacked,
        "lats":         valid_lats,
        "lons":         valid_lons,
        "gdfs_regions": gdfs_regions,
    }


# ======================================================================
# Single-model evaluation
# ======================================================================

def _build_prediction_grids(cfg):
    """Rebuild (time, lat, lon) obs/pred DataArrays from the training pickle."""
    preds_path = os.path.join(cfg.path_out, f"{cfg.model_type}_preds.pkl")
    with open(preds_path, "rb") as f:
        saved = pickle.load(f)

    model_preds = saved["model_preds"]
    nlat, nlon  = saved["nlat"], saved["nlon"]
    lat_fine    = saved["lats"]
    lon_fine    = saved["lons"]

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

    coords  = {"time": time_test, "lat": lat_fine, "lon": lon_fine}
    pred_da = xr.DataArray(Y_pred,    dims=["time", "lat", "lon"], coords=coords)
    obs_da  = xr.DataArray(Y_test_np, dims=["time", "lat", "lon"], coords=coords)
    return obs_da, pred_da


def _save_netcdf(cfg, obs_da, pred_da):
    """Export gridded obs/pred as <model_type>_ET_predictions.nc in cfg.path_out."""
    ds_out = xr.Dataset({"evap_pred": pred_da, "evap_obs": obs_da})
    ds_out["evap_pred"].attrs["units"] = "mm/day"
    ds_out["evap_obs"].attrs["units"]  = "mm/day"
    ds_out.attrs["model"] = cfg.model_type.upper()
    nc_path = os.path.join(cfg.path_out, f"{cfg.model_type}_ET_predictions.nc")
    ds_out.to_netcdf(nc_path)
    vprint(f"Saved NC file: {nc_path}")


def run_single(config_path, save_nc=False):
    cfg = load_config(config_path=config_path, train_mode=False)
    set_verbose(cfg.verbose)

    model_type = cfg.model_type
    model_name = model_type.upper()   # display label for plot titles
    conf_name  = cfg.conf_name

    # Output dirs
    path_main            = os.path.join(POSTPROCESSING_DIR, "results", model_type, conf_name, "main")
    path_metrics_dist    = os.path.join(path_main, "metric_distribution")
    path_et_dist         = os.path.join(path_main, "et_distribution")
    path_extreme         = os.path.join(POSTPROCESSING_DIR, "results", model_type, conf_name, "extreme")
    path_extreme_metrics = os.path.join(path_extreme, "metric_distribution")
    for d in (path_metrics_dist, path_et_dist, path_extreme_metrics):
        os.makedirs(d, exist_ok=True)

    vprint(f"=== Single-model postprocessing  |  {model_name}  {conf_name} ===")

    # ------------------------------------------------------------------
    # Load predictions pickle saved by training
    # ------------------------------------------------------------------
    obs_da, pred_da = _build_prediction_grids(cfg)
    if save_nc:
        _save_netcdf(cfg, obs_da, pred_da)

    # ------------------------------------------------------------------
    # Per-pixel metrics
    # ------------------------------------------------------------------
    vprint("Computing per-pixel metrics (annual + seasonal)...")
    r = _stack_and_compute_metrics(cfg, obs_da, pred_da, model_name)
    df_annual, df_seasonal = r["df_annual"], r["df_seasonal"]
    obs_stacked, pred_stacked = r["obs"], r["pred"]
    valid_lats, valid_lons = r["lats"], r["lons"]
    gdfs_regions = r["gdfs_regions"]
    gdf_morocco  = _load_morocco_gdf(cfg)

    df_annual.to_csv(os.path.join(path_main, "metrics_annual.csv"),     index=False)
    df_seasonal.to_csv(os.path.join(path_main, "metrics_seasonal.csv"), index=False)
    vprint("  Metrics CSVs saved.")

    # ------------------------------------------------------------------
    # MAIN plots
    # ------------------------------------------------------------------
    vprint("Generating main evaluation plots...")
    plot_spatial_metrics(df_annual,      model_name, gdf_morocco, path_metrics_dist)
    plot_boxplots_global(df_annual,      model_name, path_metrics_dist)
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
    plot_spatial_extreme(df_annual,              model_name, gdf_morocco, path_extreme_metrics)
    plot_boxplots_extreme(df_annual,             model_name, path_extreme_metrics)
    plot_boxplots_extreme_by_season(df_seasonal, model_name, path_extreme_metrics)
    plot_boxplots_extreme_by_region(df_annual,   model_name, path_extreme_metrics)

    vprint(f"=== Done  |  main → {path_main}  |  extreme → {path_extreme} ===")


# ======================================================================
# Intercomparison
# ======================================================================

def _load_all_for_config(config_path):
    """Load metrics df + raw stacked DataArrays for one config from its preds pickle."""
    cfg        = load_config(config_path=config_path, train_mode=False)
    model_name = cfg.model_type.upper()
    label      = f"{model_name} {cfg.conf_name}"

    preds_path = os.path.join(cfg.path_out, f"{cfg.model_type}_preds.pkl")
    if not os.path.exists(preds_path):
        print(f"  [SKIP] predictions not found: {preds_path}")
        return None

    obs_da, pred_da = _build_prediction_grids(cfg)
    r = _stack_and_compute_metrics(cfg, obs_da, pred_da, label)

    print(f"  Loaded: {label}  ({len(r['df_annual'])} pixels)")
    return r


def run_compare(config_paths):
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
    path_out  = os.path.join(POSTPROCESSING_DIR, "results", "intercomparison", run_label)
    os.makedirs(path_out, exist_ok=True)
    print(f"\nOutputs → {path_out}")

    gdf_morocco = _load_morocco_gdf(first_cfg)

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


# ======================================================================
# Entry point
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Single-model evaluation (--config) or intercomparison "
                    "(--configs / --model / --all)."
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--config",
                       help="Evaluate one config, e.g. configs/xgb/conf0.yaml")
    group.add_argument("--configs", nargs="+",
                       help="Intercompare these configs")
    group.add_argument("--model",
                       help="Intercompare all configs of one model type, e.g. xgb")
    group.add_argument("--all", action="store_true",
                       help="Intercompare all configs of all model types")
    parser.add_argument("--save-nc", action="store_true",
                        help="With --config: also export <model_type>_ET_predictions.nc")
    args = parser.parse_args()

    if args.config:
        run_single(args.config, save_nc=args.save_nc)
        return

    if args.configs:
        config_paths = args.configs
    elif args.model:
        config_paths = sorted(glob.glob(
            os.path.join("configs", args.model.lower(), "*.yaml")
        ))
    else:
        config_paths = sorted(glob.glob(os.path.join("configs", "*", "*.yaml")))
        config_paths = [p for p in config_paths if "common" not in p]

    if not config_paths:
        print("No config files found.")
        sys.exit(1)

    run_compare(config_paths)


if __name__ == "__main__":
    main()
