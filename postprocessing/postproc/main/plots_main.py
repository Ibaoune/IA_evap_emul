"""
src/main/plots_main.py
======================
All plots for the main (general) evaluation of a single model.
Style (sizes, colors, colorbars, output format) comes from the PlotConfig
passed as `style` (configs/plots/, see postproc.plot_config).

Functions:
    plot_spatial_metrics        — spatial map of each metric
    plot_boxplots_global        — boxplots of all metrics (global)
    plot_boxplots_by_season     — boxplots of all metrics split by season
    plot_boxplots_by_region     — boxplots of all metrics split by region
    plot_et_distribution        — ET boxplots by region and season
    plot_mean_et_map            — spatial map of mean ET (obs vs pred)
    plot_et_seasonal_maps       — spatial maps of mean ET per season
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs

from postproc.metrics     import SEASON_DEF, SEASON_LABELS, REGION_ORDER, REGION_LABELS
from postproc.utils       import (points_to_grid, get_region_point_indices, df_to_grid,
                                  months_of, boxplots_by_group, draw_map_frame, add_colorbar)
from postproc.plot_config import default_style


# ==========================================================
# Metrics shown by the plots (labels/colormaps of the maps are
# in configs/plots/default.yaml → colorbars)
# ==========================================================

SPATIAL_METRICS = ["R2", "KGE", "NSE", "Correlation", "Bias_pct", "RMSE", "MAE"]

# Main metrics groups for boxplots:
# Group 1 — performance:  R2, KGE, NSE, Correlation
# Group 2 — error:        RMSE, MAE, Bias_pct
MAIN_METRICS   = ["R2", "KGE", "NSE", "Correlation", "RMSE", "MAE", "Bias_pct"]
MAIN_LABELS    = ["R²", "KGE", "NSE", "r",           "RMSE", "MAE", "Bias (%)"]


# ==========================================================
# 1. Spatial map of each metric
# ==========================================================
def plot_spatial_metrics(df, model_name, gdf_morocco, path_out, style=None):
    """
    One single figure — all metrics as subplots in a grid.
    """
    style = style or default_style()
    st    = style.figure("single", "spatial_metrics")

    metrics_to_plot = [col for col in SPATIAL_METRICS if col in df.columns]
    n_metrics = len(metrics_to_plot)
    ncols     = st.ncols
    nrows     = (n_metrics + ncols - 1) // ncols

    lat_vals = np.array(sorted(df["lat"].unique()))
    lon_vals = np.array(sorted(df["lon"].unique()))

    pw, ph = st.panel_size
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(pw * ncols, ph * nrows), dpi=st.dpi,
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    axes_flat = axes.flatten() if n_metrics > 1 else [axes]

    for ax_idx, col in enumerate(metrics_to_plot):
        ax       = axes_flat[ax_idx]
        all_vals = df[col].dropna()
        cb       = style.colorbar(col, all_vals)

        grid = df_to_grid(df, col, lat_vals, lon_vals)
        lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)

        ax.pcolormesh(lon2d, lat2d, grid, cmap=cb.cmap, norm=cb.norm, shading="auto")
        draw_map_frame(ax, gdf_morocco, st)

        vals = all_vals
        ax.set_title(
            f"{cb.label}\nmed={vals.median():.3f}  min={vals.min():.3f}  max={vals.max():.3f}",
            fontsize=st.title_size, fontweight="bold"
        )

        cax = ax.inset_axes([1.03, 0, 0.05, 1])
        add_colorbar(cax, cb, st)

    # Hide unused axes
    for ax in axes_flat[n_metrics:]:
        ax.set_visible(False)

    fig.suptitle(f"{model_name} — Spatial metric distributions (test period)",
                 fontsize=st.suptitle_size, fontweight="bold", y=1.01)
    plt.tight_layout()

    os.makedirs(path_out, exist_ok=True)
    fname = style.savefig(fig, os.path.join(path_out, "spatial_metrics"))
    print(f"  Spatial metrics map saved: {fname}")


# ==========================================================
# 2. Boxplots — global (all pixels, no split)
# ==========================================================

def plot_boxplots_global(df, model_name, path_out, style=None):
    """
    One figure — boxplot of each main metric across all Morocco pixels.
    """
    style = style or default_style()
    st    = style.figure("single", "boxplots_global")

    fig, axes = plt.subplots(1, len(MAIN_METRICS), figsize=st.figsize, dpi=st.dpi)
    fig.suptitle(f"{model_name} — Metric distributions (all Morocco pixels, test period)",
                 fontsize=st.suptitle_size, fontweight="bold")

    color = style.colors["metric_box"]
    for ax, col, lbl in zip(axes, MAIN_METRICS, MAIN_LABELS):
        data = df[col].dropna().values
        if len(data) == 0:
            ax.set_title(lbl); continue
        med = np.median(data)
        ax.boxplot(data, patch_artist=True, showfliers=st.show_fliers,
                   medianprops=dict(color="black", lw=2),
                   boxprops=dict(facecolor=color, alpha=st.box_alpha),
                   whiskerprops=dict(color=color),
                   capprops=dict(color=color))
        ax.text(1, med, f" {med:.3f}", va="center", ha="left",
                fontsize=st.annotation_size, color=style.colors["median_text"],
                fontweight="bold")
        ax.set_title(lbl, fontsize=st.title_size)
        ax.set_xticklabels([model_name], fontsize=st.tick_label_size)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fname = style.savefig(fig, os.path.join(path_out, "boxplots_global"))
    print(f"  Global boxplots saved: {fname}")


# ==========================================================
# 3. Boxplots — split by season
# ==========================================================

def plot_boxplots_by_season(df_seasonal, model_name, path_out, style=None):
    """
    One figure — each metric as a column, one box per season.
    df_seasonal must have a 'Season' column (DJF, MAM, JJA, SON).
    """
    style = style or default_style()
    st    = style.figure("single", "boxplots_by_season")

    fig, axes = plt.subplots(1, len(MAIN_METRICS), figsize=st.figsize, dpi=st.dpi)
    fig.suptitle(f"{model_name} — Metric distributions by season",
                 fontsize=st.suptitle_size, fontweight="bold")

    boxplots_by_group(axes, df_seasonal, "Season", list(SEASON_DEF.keys()), SEASON_LABELS,
                       style.colors["seasons"], MAIN_METRICS, MAIN_LABELS, st)

    plt.tight_layout()
    fname = style.savefig(fig, os.path.join(path_out, "boxplots_by_season"))
    print(f"  Seasonal boxplots saved: {fname}")


# ==========================================================
# 4. Boxplots — split by region
# ==========================================================

def plot_boxplots_by_region(df_regional, model_name, path_out, style=None):
    """
    One figure — each metric as a column, one box per region.
    df_regional must have a 'region' column.
    """
    style = style or default_style()
    st    = style.figure("single", "boxplots_by_region")

    fig, axes = plt.subplots(1, len(MAIN_METRICS), figsize=st.figsize, dpi=st.dpi)
    fig.suptitle(f"{model_name} — Metric distributions by region",
                 fontsize=st.suptitle_size, fontweight="bold")

    boxplots_by_group(axes, df_regional, "region", REGION_ORDER, REGION_LABELS,
                       style.colors["regions"], MAIN_METRICS, MAIN_LABELS, st)

    plt.tight_layout()
    fname = style.savefig(fig, os.path.join(path_out, "boxplots_by_region"))
    print(f"  Regional boxplots saved: {fname}")


# ==========================================================
# 5. ET distribution by region and season
# ==========================================================

def plot_et_distribution(ref_da, pred_da, valid_lats, valid_lons,
                         gdfs_regions, model_name, path_out, style=None):
    """
    Boxplots of regional mean ET (obs vs pred) per season.
    Spatial mean computed first per region, then temporal boxplot.
    Adapted from plot_et_boxplots_by_region_season in notebook.
    """
    style = style or default_style()
    st    = style.figure("single", "et_distribution")

    region_indices = get_region_point_indices(valid_lats, valid_lons, gdfs_regions)
    months = months_of(ref_da)

    seasons    = list(SEASON_DEF.keys())
    obs_color  = style.colors["obs"]
    pred_color = style.colors["pred"]
    spacing    = 0.9
    gap        = 1.5
    group_w    = 2 * spacing

    fig, axes = plt.subplots(1, 4, figsize=st.figsize, dpi=st.dpi, sharey=False)
    fig.suptitle(
        f"{model_name} — Regional Mean ET · Distribution by Season\n"
        "(spatial mean per region first, then temporal boxplot)",
        fontsize=st.suptitle_size, fontweight="bold"
    )

    for ax, region in zip(axes, REGION_ORDER):
        pt_idx = region_indices.get(region, np.array([]))
        if len(pt_idx) == 0:
            ax.set_title(REGION_LABELS[region]); continue

        tick_centers = []
        for s_idx, season in enumerate(seasons):
            time_mask = np.isin(months, SEASON_DEF[season])
            center    = s_idx * (group_w + gap)
            tick_centers.append(center + spacing / 2)

            for m_idx, (da, color, name) in enumerate([
                (ref_da,  obs_color,  "Obs"),
                (pred_da, pred_color, model_name),
            ]):
                spatial_mean = da.isel(points=pt_idx).mean(dim="points").values
                data = spatial_mean[time_mask]
                data = data[~np.isnan(data)]
                pos  = center + m_idx * spacing
                ax.boxplot(data, positions=[pos], widths=0.7, patch_artist=True,
                           showfliers=st.show_fliers,
                           medianprops=dict(color="black", lw=1.5),
                           boxprops=dict(facecolor=color, alpha=st.box_alpha),
                           whiskerprops=dict(color=color, lw=0.8),
                           capprops=dict(color=color, lw=0.8),
                           manage_ticks=False)

        ax.set_xticks(tick_centers)
        ax.set_xticklabels([SEASON_LABELS[s] for s in seasons],
                           fontsize=st.tick_label_size, rotation=st.tick_rotation, ha="right")
        ax.set_title(REGION_LABELS[region], fontsize=st.title_size, fontweight="bold")
        ax.set_ylabel("ET (mm/day)")
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    handles = [
        mpatches.Patch(facecolor=obs_color,  alpha=st.box_alpha, label="Obs (LMDZ35)"),
        mpatches.Patch(facecolor=pred_color, alpha=st.box_alpha, label=model_name),
    ]
    fig.legend(handles=handles, loc="lower right", fontsize=st.legend_size, framealpha=0.7)
    plt.tight_layout()
    fname = style.savefig(fig, os.path.join(path_out, "et_distribution_region_season"))
    print(f"  ET distribution plot saved: {fname}")


# ==========================================================
# 6. Mean ET spatial map  (obs vs pred)
# ==========================================================

def plot_mean_et_map(ref_da, pred_da, valid_lats, valid_lons,
                     model_name, gdf_morocco, path_out, style=None):
    """
    Two-panel spatial map of mean ET: obs and pred.
    Adapted from _mean_et_maps in notebook.
    """
    style = style or default_style()
    st    = style.figure("single", "mean_et_map")

    obs_mean  = ref_da.mean(dim="time").values
    pred_mean = pred_da.mean(dim="time").values
    cb = style.colorbar("et_mean", np.concatenate([obs_mean, pred_mean]))

    def _panel(ax, vals_1d, title):
        grid, lat_u, lon_u = points_to_grid(valid_lats, valid_lons, vals_1d)
        lon2d, lat2d = np.meshgrid(lon_u, lat_u)
        ax.pcolormesh(lon2d, lat2d, grid, cmap=cb.cmap, norm=cb.norm, shading="auto")
        draw_map_frame(ax, gdf_morocco, st)
        mu = np.nanmean(grid)
        ax.set_title(f"{title}\nmean={mu:.3f}  min={np.nanmin(grid):.3f}  "
                     f"max={np.nanmax(grid):.3f}", fontsize=st.title_size, fontweight="bold")

    fig = plt.figure(figsize=st.figsize, dpi=st.dpi)
    gs  = gridspec.GridSpec(1, 3, figure=fig,
                             left=0.04, right=0.88, top=0.88, bottom=0.06,
                             width_ratios=[1, 1, 0.05], wspace=0.28)

    ax0 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax1 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())

    _panel(ax0, obs_mean,  "Obs — LMDZ35")
    _panel(ax1, pred_mean, f"Pred — {model_name}")

    add_colorbar(fig.add_subplot(gs[0, 2]), cb, st)

    fig.suptitle(f"{model_name} — Mean ET (2009–2014) · Morocco",
                 fontsize=st.suptitle_size, fontweight="bold")
    fname = style.savefig(fig, os.path.join(path_out, "mean_et_map"))
    print(f"  Mean ET map saved: {fname}")


# ==========================================================
# 7. ET seasonal spatial maps
# ==========================================================

def plot_et_seasonal_maps(ref_da, pred_da, valid_lats, valid_lons,
                           model_name, gdf_morocco, path_out, style=None):
    """
    One figure — 2 rows (obs / pred) × 4 columns (seasons).
    Adapted from _et_seasonal_maps in notebook.
    """
    style = style or default_style()
    st    = style.figure("single", "et_seasonal_maps")

    months  = months_of(ref_da)
    seasons = list(SEASON_DEF.keys())

    # Seasonal means first, so bounds: auto can cover all panels
    means = {
        (season, row_idx): da.isel(time=np.isin(months, SEASON_DEF[season])).mean(dim="time").values
        for season in seasons
        for row_idx, da in enumerate([ref_da, pred_da])
    }
    cb = style.colorbar("et_mean", np.concatenate(list(means.values())))

    fig, axes = plt.subplots(
        2, 4, figsize=st.figsize, dpi=st.dpi,
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    row_labels = ["Obs (LMDZ35)", f"Pred ({model_name})"]

    for col_idx, season in enumerate(seasons):
        for row_idx, row_lbl in enumerate(row_labels):
            ax  = axes[row_idx, col_idx]
            grid, lat_u, lon_u = points_to_grid(valid_lats, valid_lons,
                                                means[(season, row_idx)])
            lon2d, lat2d = np.meshgrid(lon_u, lat_u)

            ax.pcolormesh(lon2d, lat2d, grid, cmap=cb.cmap, norm=cb.norm, shading="auto")
            draw_map_frame(ax, gdf_morocco, st)

            title = f"{SEASON_LABELS[season]}" if row_idx == 0 else ""
            if col_idx == 0:
                title = f"{row_lbl}\n{title}" if title else row_lbl
            ax.set_title(title, fontsize=st.title_size, fontweight="bold")

    # Shared colorbar
    fig.subplots_adjust(right=0.90, hspace=0.15, wspace=0.1)
    add_colorbar(fig.add_axes([0.92, 0.12, 0.018, 0.76]), cb, st)

    fig.suptitle(f"{model_name} — Seasonal Mean ET (2009–2014) · Morocco",
                 fontsize=st.suptitle_size, fontweight="bold", y=0.98)
    fname = style.savefig(fig, os.path.join(path_out, "et_seasonal_maps"))
    print(f"  Seasonal ET maps saved: {fname}")
