"""
postprocess.py
==============
Postprocessing of trained models, in two modes:

Predictions are read from the training pickle
results/<model>/<conf>/<model>_preds.pkl. The computed metrics and the
stacked obs/pred series are cached per config, so figures can be redrawn
(all of them or only some) without recomputing anything.

    python postprocess.py --help           # all arguments and examples
    python postprocess.py --list-figures   # all figure names and groups

Figure style (sizes, colors, colorbars, output format) comes from
configs/plots/default.yaml, optionally overridden with --plot-config.

1. Single-model evaluation (main + extreme) for ONE config.

       python postprocess.py --config configs/train/xgb/conf0.yaml
       python postprocess.py --config configs/train/xgb/conf0.yaml --plot-only --figures main

   Outputs saved in:
       postprocessing/results/single/<model>/<conf>/cache/
       postprocessing/results/single/<model>/<conf>/main/
       postprocessing/results/single/<model>/<conf>/extreme/

2. Intercomparison across multiple trained configs (reuses each config's cache).

       # Compare specific configs
       python postprocess.py \
           --configs configs/train/xgb/conf0.yaml configs/train/xgb/conf1.yaml configs/train/lstm/conf0.yaml

       # Compare ALL configs of one model
       python postprocess.py --model xgb

       # Compare ALL trained configs across all models
       python postprocess.py --all

   Outputs saved in:
       postprocessing/results/intercomparison/<run_label>/
"""

import os
import sys
import glob
import pickle
import types
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
from src.core.cli import TableHelpParser, render_table

from postproc.utils   import get_valid_morocco_points, stack_on_points
from postproc.metrics import (
    compute_metrics_df, compute_seasonal_metrics_df, add_region_labels
)
from postproc.cache   import cache_status, save_cache, load_cache
from postproc.figures import SINGLE_FIGURES, INTERCOMP_FIGURES, resolve_figures
from postproc.plot_config import load_plot_config, PlotConfigError

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


def _single_dir(cfg):
    return os.path.join(POSTPROCESSING_DIR, "results", "single", cfg.model_type, cfg.conf_name)


def get_results(cfg, label, plot_only=False, recompute=False):
    """
    Return the computed results of one config, from its cache when valid.

    The cache is computed (and saved) when missing or stale, unless
    plot_only is set. Returns None, with a printed reason, when there is
    nothing usable.
    """
    preds_path = os.path.join(cfg.path_out, f"{cfg.model_type}_preds.pkl")
    cache_dir  = os.path.join(_single_dir(cfg), "cache")
    status     = cache_status(cache_dir, cfg, preds_path)
    tag        = f"{cfg.model_type}/{cfg.conf_name}"

    if not recompute and status in ("ok", "unverified"):
        if status == "unverified":
            print(f"  [WARN] {tag}: predictions pickle not found, using its cache unchecked")
        vprint(f"  {tag}: using cache {cache_dir}")
        r = load_cache(cache_dir)
    elif plot_only:
        print(f"  [SKIP] {tag}: cache is {status}, run without --plot-only to compute it")
        return None
    elif not os.path.exists(preds_path):
        print(f"  [SKIP] {tag}: predictions not found: {preds_path}")
        return None
    else:
        vprint(f"  {tag}: computing per-pixel metrics (annual + seasonal)...")
        obs_da, pred_da = _build_prediction_grids(cfg)
        r = _stack_and_compute_metrics(cfg, obs_da, pred_da, cfg.model_type.upper())
        save_cache(cache_dir, cfg, preds_path, r)
        vprint(f"  {tag}: cache saved in {cache_dir}")

    r["label"] = label
    r["cfg"]   = cfg
    r["gdfs_regions"] = _load_region_gdfs(cfg)
    for df in (r["df_annual"], r["df_seasonal"]):
        df["Model"] = label
    return r


# ======================================================================
# Single-model evaluation
# ======================================================================

def run_single(config_path, figures, style, plot_only=False, recompute=False):
    cfg = load_config(config_path=config_path, train_mode=False)
    set_verbose(cfg.verbose)

    model_name = cfg.model_type.upper()   # display label for plot titles
    vprint(f"=== Single-model postprocessing  |  {model_name}  {cfg.conf_name} ===")

    r = get_results(cfg, model_name, plot_only, recompute)
    if r is None:
        sys.exit(1)

    # Output dirs
    path_main    = os.path.join(_single_dir(cfg), "main")
    path_extreme = os.path.join(_single_dir(cfg), "extreme")
    ctx = types.SimpleNamespace(
        **r,
        style                = style,
        model_name           = model_name,
        gdf_morocco          = _load_morocco_gdf(cfg),
        path_metrics_dist    = os.path.join(path_main, "metric_distribution"),
        path_et_dist         = os.path.join(path_main, "et_distribution"),
        path_extreme_metrics = os.path.join(path_extreme, "metric_distribution"),
    )
    for d in (ctx.path_metrics_dist, ctx.path_et_dist, ctx.path_extreme_metrics):
        os.makedirs(d, exist_ok=True)

    for name in figures:
        vprint(f"Plotting {name}...")
        SINGLE_FIGURES[name].draw(ctx)

    record = style.write_record(_single_dir(cfg))
    vprint(f"Style used → {record}")
    vprint(f"=== Done  |  main → {path_main}  |  extreme → {path_extreme} ===")


# ======================================================================
# Intercomparison
# ======================================================================

def run_compare(config_paths, figures, style, plot_only=False, recompute=False):
    results = []
    for cp in config_paths:
        cfg   = load_config(config_path=cp, train_mode=False)
        label = f"{cfg.model_type.upper()} {cfg.conf_name}"
        r     = get_results(cfg, label, plot_only, recompute)
        if r is not None:
            print(f"  Loaded: {label}  ({len(r['df_annual'])} pixels)")
            results.append(r)

    if not results:
        print("No valid predictions found.")
        sys.exit(1)

    labels = [r["label"] for r in results]

    # Output dir
    run_label = "_vs_".join(l.replace(" ", "_") for l in labels).lower()
    path_out  = os.path.join(POSTPROCESSING_DIR, "results", "intercomparison", run_label)
    os.makedirs(path_out, exist_ok=True)
    print(f"\nOutputs → {path_out}")

    ctx = types.SimpleNamespace(
        style        = style,
        labels       = labels,
        colors       = style.model_colors(len(labels)),
        dfs          = [r["df_annual"]   for r in results],
        dfs_seasonal = [r["df_seasonal"] for r in results],
        obs_list     = [r["obs"]         for r in results],
        pred_list    = [r["pred"]        for r in results],
        lats_list    = [r["lats"]        for r in results],
        lons_list    = [r["lons"]        for r in results],
        gdfs_regions = results[0]["gdfs_regions"],
        gdf_morocco  = _load_morocco_gdf(results[0]["cfg"]),
        path_out     = path_out,
    )

    for name in figures:
        print(f"Plotting {name}...")
        INTERCOMP_FIGURES[name].draw(ctx)

    record = style.write_record(path_out)
    print(f"Style used → {record}")
    print(f"\n=== Intercomparison done  |  {path_out} ===")


# ======================================================================
# Entry point
# ======================================================================

def _print_figure_list():
    rows = [["single", f.group, name, f.description] for name, f in SINGLE_FIGURES.items()]
    rows += [["intercomparison", f.group, name, f.description]
             for name, f in INTERCOMP_FIGURES.items()]
    print(render_table(["Mode", "Group", "Figure", "Description"], rows))
    print("\nPass figure names and/or groups to --figures; 'all' selects every "
          "figure of the current mode.")


def _build_parser():
    examples = [
        ("python postprocess.py --config configs/train/xgb/conf0.yaml",
         "Evaluate one config: compute or reuse its cache, draw all figures (style: configs/plots/default.yaml)"),
        ("python postprocess.py --config configs/train/xgb/conf0.yaml --plot-only --figures main",
         "Redraw only the main figures from the cache"),
        ("python postprocess.py --config configs/train/xgb/conf0.yaml --plot-only --figures extreme",
         "Redraw only the extreme figures from the cache"),
        ("python postprocess.py --config configs/train/xgb/conf0.yaml --plot-only --figures spatial_metrics mean_et_map",
         "Redraw two specific figures"),
        ("python postprocess.py --config configs/train/xgb/conf0.yaml --recompute",
         "Ignore the cache: recompute metrics from the predictions pickle"),
        ("python postprocess.py --configs configs/train/xgb/conf0.yaml configs/train/lstm/conf0.yaml",
         "Intercompare these configs (all intercomparison figures)"),
        ("python postprocess.py --model xgb",
         "Intercompare all configs of one model type"),
        ("python postprocess.py --all --plot-only --figures intercomp",
         "Intercompare all trained configs from their caches"),
        ("python postprocess.py --all --plot-only --figures mean_et_map et_boxplots",
         "Redraw two intercomparison figures"),
        ("python postprocess.py --config configs/train/xgb/conf0.yaml --plot-only --figures spatial_metrics --plot-config configs/plots/overrides/paper.yaml",
         "Redraw one figure with the paper style (overrides/paper.yaml merged on configs/plots/default.yaml)"),
        ("python postprocess.py --list-figures",
         "List every figure name and group"),
    ]
    parser = TableHelpParser(
        prog="postprocess.py",
        description="Single-model evaluation (--config) or intercomparison "
                    "(--configs / --model / --all) of trained configs. Metrics and "
                    "obs/pred series are cached per config in "
                    "postprocessing/results/single/<model>/<conf>/cache/.",
        examples=examples,
        notes="Run from the project root. Use --list-figures to see figure names. "
              "Figure style: configs/plots/default.yaml, unless --plot-config is given (the style actually used is "
              "saved next to the figures in plot_config_used.yaml).",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--config", metavar="YAML",
                      help="Evaluate one config, e.g. configs/train/xgb/conf0.yaml")
    mode.add_argument("--configs", nargs="+", metavar="YAML",
                      help="Intercompare these configs")
    mode.add_argument("--model", metavar="TYPE",
                      help="Intercompare all configs of one model type, e.g. xgb")
    mode.add_argument("--all", action="store_true",
                      help="Intercompare all configs of all model types")
    mode.add_argument("--list-figures", action="store_true",
                      help="List all figure names and groups, then exit")

    parser.add_argument("--figures", nargs="+", default=["all"], metavar="NAME",
                        help="Figures to draw: figure names and/or groups "
                             "(main, extreme for --config; intercomp for the "
                             "intercomparison modes; all)")
    parser.add_argument("--plot-config", metavar="YAML",
                        help="Figure style. Without it: configs/plots/default.yaml. With it: this "
                             "override (e.g. configs/plots/overrides/paper.yaml) merged on top of "
                             "default.yaml")
    cache = parser.add_mutually_exclusive_group()
    cache.add_argument("--plot-only", action="store_true",
                       help="Never compute: draw from the cache, skip configs without a valid one")
    cache.add_argument("--recompute", action="store_true",
                       help="Ignore the cache and recompute it from the predictions pickle")
    return parser


def main():
    parser = _build_parser()
    args   = parser.parse_args()

    if args.list_figures:
        _print_figure_list()
        return
    if not (args.config or args.configs or args.model or args.all):
        parser.error("one of --config, --configs, --model, --all or --list-figures is required")

    registry = SINGLE_FIGURES if args.config else INTERCOMP_FIGURES
    try:
        figures = resolve_figures(args.figures, registry)
        style   = load_plot_config(args.plot_config)
    except (ValueError, PlotConfigError) as e:
        parser.error(str(e))

    if args.config:
        run_single(args.config, figures, style, args.plot_only, args.recompute)
        return

    if args.configs:
        config_paths = args.configs
    elif args.model:
        config_paths = sorted(glob.glob(
            os.path.join("configs", "train", args.model.lower(), "*.yaml")
        ))
    else:
        config_paths = sorted(glob.glob(os.path.join("configs", "train", "*", "*.yaml")))

    if not config_paths:
        print("No config files found.")
        sys.exit(1)

    run_compare(config_paths, figures, style, args.plot_only, args.recompute)


if __name__ == "__main__":
    main()
