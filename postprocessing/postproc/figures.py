"""
postproc/figures.py
===================
Registry of every figure postprocess.py can draw, so figures can be
selected by name or by group on the command line (--figures).

Each entry maps a figure name to (group, description, draw), where draw
takes a context namespace built in postprocess.py (all modes also carry
`style`, the PlotConfig loaded from configs/plots/):

    single mode : df_annual, df_seasonal, obs, pred, lats, lons,
                  gdfs_regions, gdf_morocco, model_name,
                  path_metrics_dist, path_et_dist, path_extreme_metrics
    intercomp   : labels, colors, dfs, dfs_seasonal, obs_list, pred_list,
                  lats_list, lons_list, gdfs_regions, gdf_morocco, path_out
"""

from collections import namedtuple

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
)


Figure = namedtuple("Figure", ["group", "description", "draw"])


# ==========================================================
# Single-model figures  (groups: main, extreme)
# ==========================================================

SINGLE_FIGURES = {
    "spatial_metrics": Figure(
        "main", "Maps of each per-pixel metric (R², KGE, NSE, r, bias, RMSE, MAE)",
        lambda c: plot_spatial_metrics(c.df_annual, c.model_name, c.gdf_morocco,
                                       c.path_metrics_dist, style=c.style)),
    "boxplots_global": Figure(
        "main", "Boxplots of each metric over all pixels",
        lambda c: plot_boxplots_global(c.df_annual, c.model_name, c.path_metrics_dist, style=c.style)),
    "boxplots_by_season": Figure(
        "main", "Metric boxplots split by season",
        lambda c: plot_boxplots_by_season(c.df_seasonal, c.model_name, c.path_metrics_dist, style=c.style)),
    "boxplots_by_region": Figure(
        "main", "Metric boxplots split by region",
        lambda c: plot_boxplots_by_region(c.df_annual, c.model_name, c.path_metrics_dist, style=c.style)),
    "et_distribution": Figure(
        "main", "Obs vs pred ET distribution by region and season",
        lambda c: plot_et_distribution(c.obs, c.pred, c.lats, c.lons, c.gdfs_regions,
                                       c.model_name, c.path_et_dist, style=c.style)),
    "mean_et_map": Figure(
        "main", "Maps of mean ET, obs vs pred",
        lambda c: plot_mean_et_map(c.obs, c.pred, c.lats, c.lons, c.model_name,
                                   c.gdf_morocco, c.path_et_dist, style=c.style)),
    "et_seasonal_maps": Figure(
        "main", "Maps of seasonal mean ET, obs vs pred",
        lambda c: plot_et_seasonal_maps(c.obs, c.pred, c.lats, c.lons, c.model_name,
                                        c.gdf_morocco, c.path_et_dist, style=c.style)),
    "extreme_spatial": Figure(
        "extreme", "Maps of MAE on extremes (Q95, Q99, Q99.5)",
        lambda c: plot_spatial_extreme(c.df_annual, c.model_name, c.gdf_morocco,
                                       c.path_extreme_metrics, style=c.style)),
    "extreme_boxplots": Figure(
        "extreme", "Boxplots of the extreme metrics over all pixels",
        lambda c: plot_boxplots_extreme(c.df_annual, c.model_name, c.path_extreme_metrics, style=c.style)),
    "extreme_boxplots_by_season": Figure(
        "extreme", "Extreme-metric boxplots split by season",
        lambda c: plot_boxplots_extreme_by_season(c.df_seasonal, c.model_name,
                                                  c.path_extreme_metrics, style=c.style)),
    "extreme_boxplots_by_region": Figure(
        "extreme", "Extreme-metric boxplots split by region",
        lambda c: plot_boxplots_extreme_by_region(c.df_annual, c.model_name,
                                                  c.path_extreme_metrics, style=c.style)),
}


# ==========================================================
# Intercomparison figures  (group: intercomp)
# ==========================================================

INTERCOMP_FIGURES = {
    "bars_by_region": Figure(
        "intercomp", "Per-region bars of R², KGE, RMSE, MAE Q95, one bar per model",
        lambda c: plot_intercomparison_by_region(c.dfs, c.labels, c.path_out, c.colors, style=c.style)),
    "spatial_metrics": Figure(
        "intercomp", "Maps of each metric, one column per model",
        lambda c: plot_spatial_metrics_intercomp(c.dfs, c.labels, c.gdf_morocco, c.path_out, style=c.style)),
    "boxplots_global": Figure(
        "intercomp", "Metric boxplots over all pixels, models side by side",
        lambda c: plot_boxplots_global_intercomp(c.dfs, c.labels, c.colors, c.path_out, style=c.style)),
    "boxplots_by_season": Figure(
        "intercomp", "Metric boxplots by season, models side by side",
        lambda c: plot_boxplots_by_season_intercomp(c.dfs_seasonal, c.labels, c.colors,
                                                    c.path_out, style=c.style)),
    "boxplots_by_region": Figure(
        "intercomp", "Metric boxplots by region, models side by side",
        lambda c: plot_boxplots_by_region_intercomp(c.dfs, c.labels, c.colors, c.path_out, style=c.style)),
    "mean_et_map": Figure(
        "intercomp", "Maps of mean ET, obs and every model",
        lambda c: plot_mean_et_map_intercomp(c.obs_list, c.pred_list, c.lats_list,
                                             c.lons_list, c.labels, c.gdf_morocco,
                                             c.path_out, style=c.style)),
    "et_seasonal_maps": Figure(
        "intercomp", "Maps of seasonal mean ET, obs and every model",
        lambda c: plot_et_seasonal_maps_intercomp(c.obs_list, c.pred_list, c.lats_list,
                                                  c.lons_list, c.labels, c.gdf_morocco,
                                                  c.path_out, style=c.style)),
    "et_boxplots": Figure(
        "intercomp", "ET boxplots by region and season, obs and every model",
        lambda c: plot_et_boxplots_intercomp(c.obs_list, c.pred_list, c.lats_list,
                                             c.lons_list, c.labels, c.colors,
                                             c.gdfs_regions, c.path_out, style=c.style)),
}


# ==========================================================
# Selection
# ==========================================================

def resolve_figures(requested, registry):
    """
    Expand figure names and group names ("main", "extreme", "intercomp",
    "all") into an ordered list of figure names from registry.

    Raises ValueError on a name that is neither a figure nor a group of registry.
    """
    groups   = sorted({f.group for f in registry.values()})
    selected = []
    for name in requested:
        if name == "all":
            names = list(registry)
        elif name in registry:
            names = [name]
        else:
            names = [n for n, f in registry.items() if f.group == name]
        if not names:
            raise ValueError(
                f"unknown figure or group '{name}' for this mode. "
                f"Groups: {', '.join(groups + ['all'])}. "
                f"Figures: {', '.join(registry)}"
            )
        selected += [n for n in names if n not in selected]
    return selected
