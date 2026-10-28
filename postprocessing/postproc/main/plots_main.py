"""
src/main/plots_main.py
======================
All plots for the main (general) evaluation of a single model.

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
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import cartopy.crs as ccrs
from matplotlib.colors import BoundaryNorm

from postproc.metrics  import SEASON_DEF, SEASON_LABELS, REGION_ORDER, REGION_LABELS
from postproc.utils    import points_to_grid, compute_smart_bounds, get_region_point_indices


# ==========================================================
# Shared metric display config
# ==========================================================

METRIC_CFG = {
    "R2"         : ("R²",               mpl.cm.RdYlGn),
    "KGE"        : ("KGE",              mpl.cm.RdYlGn),
    "NSE"        : ("NSE",              mpl.cm.RdYlGn),
    "Correlation": ("Pearson r",        mpl.cm.RdBu),
    "Bias_pct"   : ("Bias (%)",         mpl.cm.RdBu_r),
    "RMSE"       : ("RMSE (mm/day)",    mpl.cm.YlOrRd),
    "MAE"        : ("MAE (mm/day)",     mpl.cm.YlOrRd),
}

# Main metrics groups for boxplots:
# Group 1 — performance:  R2, KGE, NSE, Correlation
# Group 2 — error:        RMSE, MAE, Bias_pct
MAIN_METRICS   = ["R2", "KGE", "NSE", "Correlation", "RMSE", "MAE", "Bias_pct"]
MAIN_LABELS    = ["R²", "KGE", "NSE", "r",           "RMSE", "MAE", "Bias (%)"]


# ==========================================================
# 1. Spatial map of each metric
# ==========================================================
def plot_spatial_metrics(df, model_name, gdf_morocco, path_out):
    """
    One single figure — all metrics as subplots in a grid.
    """
    metrics_to_plot = [col for col in METRIC_CFG if col in df.columns]
    n_metrics = len(metrics_to_plot)
    ncols     = 4
    nrows     = (n_metrics + ncols - 1) // ncols

    lat_vals = np.array(sorted(df["lat"].unique()))
    lon_vals = np.array(sorted(df["lon"].unique()))
    minx, miny, maxx, maxy = gdf_morocco.total_bounds

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4.5 * nrows), dpi=150,
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    axes_flat = axes.flatten() if n_metrics > 1 else [axes]

    for ax_idx, col in enumerate(metrics_to_plot):
        ax       = axes_flat[ax_idx]
        lbl, cmap_base = METRIC_CFG[col]

        all_vals = df[col].dropna()
        bounds   = compute_smart_bounds(all_vals)
        if col == "KGE":
            bounds = [b for b in bounds if b <= 1.0]
            if len(bounds) == 0:
                bounds = [-1.0, 0.0, 1.0]
            elif bounds[-1] < 1.0:
                bounds.append(1.0)
            if len(bounds) == 1:
                bounds.insert(0, bounds[0] - 1.0)

        ncolors = len(bounds) - 1
        cmap    = mpl.colors.ListedColormap(cmap_base(np.linspace(0, 1, ncolors)))
        norm    = BoundaryNorm(boundaries=bounds, ncolors=ncolors)

        grid = np.full((len(lat_vals), len(lon_vals)), np.nan)
        for _, row in df.iterrows():
            li = np.searchsorted(lat_vals, row["lat"])
            lo = np.searchsorted(lon_vals, row["lon"])
            if li < len(lat_vals) and lo < len(lon_vals):
                grid[li, lo] = row[col]
        lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)

        ax.pcolormesh(lon2d, lat2d, grid, cmap=cmap, norm=norm, shading="auto")
        gdf_morocco.boundary.plot(ax=ax, edgecolor="black", linewidth=1.0)
        ax.set_extent([minx - 0.3, maxx + 0.3, miny - 0.3, maxy + 0.3])
        gl = ax.gridlines(draw_labels=True, linewidth=0.4,
                          color="gray", alpha=0.4, linestyle="--")
        gl.top_labels = False; gl.right_labels = False
        gl.xlabel_style = {"size": 5}; gl.ylabel_style = {"size": 5}

        vals = all_vals
        ax.set_title(
            f"{lbl}\nmed={vals.median():.3f}  min={vals.min():.3f}  max={vals.max():.3f}",
            fontsize=7.5, fontweight="bold"
        )

        cax = ax.inset_axes([1.03, 0, 0.05, 1])
        mpl.colorbar.ColorbarBase(
            cax, cmap=cmap, norm=norm, boundaries=bounds,
            ticks=bounds, orientation="vertical"
        ).set_label(lbl, fontsize=6)
        cax.tick_params(labelsize=5)

    # Hide unused axes
    for ax in axes_flat[n_metrics:]:
        ax.set_visible(False)

    fig.suptitle(f"{model_name} — Spatial metric distributions (test period)",
                 fontsize=12, fontweight="bold", y=1.01)
    plt.tight_layout()

    os.makedirs(path_out, exist_ok=True)
    fname = os.path.join(path_out, "spatial_metrics.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Spatial metrics map saved: {fname}")


# ==========================================================
# 2. Boxplots — global (all pixels, no split)
# ==========================================================

def plot_boxplots_global(df, model_name, path_out):
    """
    One figure — boxplot of each main metric across all Morocco pixels.
    """
    fig, axes = plt.subplots(1, len(MAIN_METRICS), figsize=(22, 5), dpi=130)
    fig.suptitle(f"{model_name} — Metric distributions (all Morocco pixels, test period)",
                 fontsize=11, fontweight="bold")

    color = "#3498DB"
    for ax, col, lbl in zip(axes, MAIN_METRICS, MAIN_LABELS):
        data = df[col].dropna().values
        if len(data) == 0:
            ax.set_title(lbl); continue
        med = np.median(data)
        ax.boxplot(data, patch_artist=True, showfliers=False,
                   medianprops=dict(color="black", lw=2),
                   boxprops=dict(facecolor=color, alpha=0.75),
                   whiskerprops=dict(color=color),
                   capprops=dict(color=color))
        ax.text(1, med, f" {med:.3f}", va="center", ha="left",
                fontsize=9, color="darkred", fontweight="bold")
        ax.set_title(lbl, fontsize=9)
        ax.set_xticklabels([model_name], fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fname = os.path.join(path_out, "boxplots_global.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Global boxplots saved: {fname}")


# ==========================================================
# 3. Boxplots — split by season
# ==========================================================

def plot_boxplots_by_season(df_seasonal, model_name, path_out):
    """
    One figure — each metric as a column, one box per season.
    df_seasonal must have a 'Season' column (DJF, MAM, JJA, SON).
    """
    seasons  = list(SEASON_DEF.keys())
    n_metrics = len(MAIN_METRICS)
    color_map = {"DJF": "#3498DB", "MAM": "#2ECC71", "JJA": "#E74C3C", "SON": "#E67E22"}

    fig, axes = plt.subplots(1, n_metrics, figsize=(24, 5), dpi=130)
    fig.suptitle(f"{model_name} — Metric distributions by season",
                 fontsize=11, fontweight="bold")

    for ax, col, lbl in zip(axes, MAIN_METRICS, MAIN_LABELS):
        data_list  = [df_seasonal[df_seasonal["Season"] == s][col].dropna().values
                      for s in seasons]
        colors     = [color_map[s] for s in seasons]
        bp = ax.boxplot(data_list, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=2))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(0.75)
        for i, (d, c) in enumerate(zip(data_list, colors)):
            if len(d):
                med = np.median(d)
                ax.text(i + 1, med, f"{med:.2f}", ha="center", va="bottom",
                        fontsize=7, fontweight="bold", color=c)
        ax.set_xticks(range(1, len(seasons) + 1))
        ax.set_xticklabels([SEASON_LABELS[s] for s in seasons],
                           fontsize=7, rotation=20, ha="right")
        ax.set_title(lbl, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fname = os.path.join(path_out, "boxplots_by_season.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Seasonal boxplots saved: {fname}")


# ==========================================================
# 4. Boxplots — split by region
# ==========================================================

def plot_boxplots_by_region(df_regional, model_name, path_out):
    """
    One figure — each metric as a column, one box per region.
    df_regional must have a 'region' column.
    """
    color_map = {
        "north"     : "#3498DB",
        "north_east": "#2ECC71",
        "east"      : "#E67E22",
        "south"     : "#E74C3C",
    }

    fig, axes = plt.subplots(1, len(MAIN_METRICS), figsize=(24, 5), dpi=130)
    fig.suptitle(f"{model_name} — Metric distributions by region",
                 fontsize=11, fontweight="bold")

    for ax, col, lbl in zip(axes, MAIN_METRICS, MAIN_LABELS):
        data_list = [df_regional[df_regional["region"] == r][col].dropna().values
                     for r in REGION_ORDER]
        colors    = [color_map[r] for r in REGION_ORDER]
        bp = ax.boxplot(data_list, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=2))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(0.75)
        for i, (d, c) in enumerate(zip(data_list, colors)):
            if len(d):
                med = np.median(d)
                ax.text(i + 1, med, f"{med:.2f}", ha="center", va="bottom",
                        fontsize=7, fontweight="bold", color=c)
        ax.set_xticks(range(1, len(REGION_ORDER) + 1))
        ax.set_xticklabels([REGION_LABELS[r] for r in REGION_ORDER],
                           fontsize=8, rotation=15, ha="right")
        ax.set_title(lbl, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fname = os.path.join(path_out, "boxplots_by_region.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Regional boxplots saved: {fname}")


# ==========================================================
# 5. ET distribution by region and season
# ==========================================================

def plot_et_distribution(ref_da, pred_da, valid_lats, valid_lons,
                         gdfs_regions, model_name, path_out):
    """
    Boxplots of regional mean ET (obs vs pred) per season.
    Spatial mean computed first per region, then temporal boxplot.
    Adapted from plot_et_boxplots_by_region_season in notebook.
    """
    region_indices = get_region_point_indices(valid_lats, valid_lons, gdfs_regions)
    try:
        months = pd.DatetimeIndex(ref_da.time.values).month.values
    except Exception:
        months = np.array([t.month for t in ref_da.time.values])

    seasons    = list(SEASON_DEF.keys())
    obs_color  = "#2C2C2C"
    pred_color = "#3498DB"
    spacing    = 0.9
    gap        = 1.5
    group_w    = 2 * spacing

    fig, axes = plt.subplots(1, 4, figsize=(22, 5), sharey=False)
    fig.suptitle(
        f"{model_name} — Regional Mean ET · Distribution by Season\n"
        "(spatial mean per region first, then temporal boxplot)",
        fontsize=11, fontweight="bold"
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
                           showfliers=False,
                           medianprops=dict(color="black", lw=1.5),
                           boxprops=dict(facecolor=color, alpha=0.75),
                           whiskerprops=dict(color=color, lw=0.8),
                           capprops=dict(color=color, lw=0.8),
                           manage_ticks=False)

        ax.set_xticks(tick_centers)
        ax.set_xticklabels([SEASON_LABELS[s] for s in seasons],
                           fontsize=8, rotation=15, ha="right")
        ax.set_title(REGION_LABELS[region], fontsize=9, fontweight="bold")
        ax.set_ylabel("ET (mm/day)")
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    handles = [
        mpatches.Patch(facecolor=obs_color,  alpha=0.75, label="Obs (LMDZ35)"),
        mpatches.Patch(facecolor=pred_color, alpha=0.75, label=model_name),
    ]
    fig.legend(handles=handles, loc="lower right", fontsize=9, framealpha=0.7)
    plt.tight_layout()
    fname = os.path.join(path_out, "et_distribution_region_season.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ET distribution plot saved: {fname}")


# ==========================================================
# 6. Mean ET spatial map  (obs vs pred)
# ==========================================================

def plot_mean_et_map(ref_da, pred_da, valid_lats, valid_lons,
                     model_name, gdf_morocco, path_out):
    """
    Two-panel spatial map of mean ET: obs and pred.
    Adapted from _mean_et_maps in notebook.
    """
    et_bounds = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2,
                           1.4, 1.6, 1.8, 2.0, 2.2, 2.4])
    N_et    = len(et_bounds) - 1
    cmap_et = mpl.colors.ListedColormap(
        plt.get_cmap("RdYlGn", N_et)(np.linspace(0, 1, N_et))
    )
    norm_et = mpl.colors.BoundaryNorm(boundaries=et_bounds, ncolors=N_et)
    minx, miny, maxx, maxy = gdf_morocco.total_bounds

    def _extend(grid):
        mn, mx = np.nanmin(grid), np.nanmax(grid)
        if   mn < et_bounds[0]  and mx > et_bounds[-1]: return "both"
        elif mn < et_bounds[0]:                          return "min"
        elif mx > et_bounds[-1]:                         return "max"
        else:                                            return "neither"

    def _panel(ax, vals_1d, title):
        grid, lat_u, lon_u = points_to_grid(valid_lats, valid_lons, vals_1d)
        lon2d, lat2d = np.meshgrid(lon_u, lat_u)
        ax.pcolormesh(lon2d, lat2d, grid, cmap=cmap_et, norm=norm_et, shading="auto")
        gdf_morocco.boundary.plot(ax=ax, edgecolor="black", linewidth=1.0)
        ax.set_extent([minx - 0.3, maxx + 0.3, miny - 0.3, maxy + 0.3])
        gl = ax.gridlines(draw_labels=True, linewidth=0.4,
                          color="gray", alpha=0.4, linestyle="--")
        gl.top_labels = False; gl.right_labels = False
        gl.xlabel_style = {"size": 6}; gl.ylabel_style = {"size": 6}
        mu = np.nanmean(grid)
        ax.set_title(f"{title}\nmean={mu:.3f}  min={np.nanmin(grid):.3f}  "
                     f"max={np.nanmax(grid):.3f}", fontsize=8.5, fontweight="bold")

    obs_mean  = ref_da.mean(dim="time").values
    pred_mean = pred_da.mean(dim="time").values

    fig = plt.figure(figsize=(13, 5.5), dpi=150)
    gs  = gridspec.GridSpec(1, 3, figure=fig,
                             left=0.04, right=0.88, top=0.88, bottom=0.06,
                             width_ratios=[1, 1, 0.05], wspace=0.28)

    ax0 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    ax1 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())

    _panel(ax0, obs_mean,  "Obs — LMDZ35")
    _panel(ax1, pred_mean, f"Pred — {model_name}")

    cax = fig.add_subplot(gs[0, 2])
    sm  = plt.cm.ScalarMappable(cmap=cmap_et, norm=norm_et)
    cb  = fig.colorbar(sm, cax=cax, ticks=et_bounds)
    cb.ax.set_yticklabels([f"{b:.1f}" for b in et_bounds], fontsize=7)
    cb.set_label("Mean ET (mm/day)", fontsize=9)

    fig.suptitle(f"{model_name} — Mean ET (2009–2014) · Morocco",
                 fontsize=11, fontweight="bold")
    fname = os.path.join(path_out, "mean_et_map.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Mean ET map saved: {fname}")


# ==========================================================
# 7. ET seasonal spatial maps
# ==========================================================

def plot_et_seasonal_maps(ref_da, pred_da, valid_lats, valid_lons,
                           model_name, gdf_morocco, path_out):
    """
    One figure — 2 rows (obs / pred) × 4 columns (seasons).
    Adapted from _et_seasonal_maps in notebook.
    """
    et_bounds = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2,
                           1.4, 1.6, 1.8, 2.0, 2.2, 2.4])
    N_et    = len(et_bounds) - 1
    cmap_et = mpl.colors.ListedColormap(
        plt.get_cmap("RdYlGn", N_et)(np.linspace(0, 1, N_et))
    )
    norm_et = mpl.colors.BoundaryNorm(boundaries=et_bounds, ncolors=N_et)
    minx, miny, maxx, maxy = gdf_morocco.total_bounds

    try:
        months = pd.DatetimeIndex(ref_da.time.values).month.values
    except Exception:
        months = np.array([t.month for t in ref_da.time.values])

    seasons = list(SEASON_DEF.keys())

    fig, axes = plt.subplots(
        2, 4, figsize=(20, 8), dpi=150,
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    row_labels = ["Obs (LMDZ35)", f"Pred ({model_name})"]

    for col_idx, season in enumerate(seasons):
        mask = np.isin(months, SEASON_DEF[season])
        for row_idx, (da, row_lbl) in enumerate([
            (ref_da,  row_labels[0]),
            (pred_da, row_labels[1]),
        ]):
            ax  = axes[row_idx, col_idx]
            vals = da.isel(time=mask).mean(dim="time").values
            grid, lat_u, lon_u = points_to_grid(valid_lats, valid_lons, vals)
            lon2d, lat2d = np.meshgrid(lon_u, lat_u)

            ax.pcolormesh(lon2d, lat2d, grid, cmap=cmap_et, norm=norm_et, shading="auto")
            gdf_morocco.boundary.plot(ax=ax, edgecolor="black", linewidth=0.8)
            ax.set_extent([minx - 0.2, maxx + 0.2, miny - 0.2, maxy + 0.2])

            title = f"{SEASON_LABELS[season]}" if row_idx == 0 else ""
            if col_idx == 0:
                title = f"{row_lbl}\n{title}" if title else row_lbl
            ax.set_title(title, fontsize=8.5, fontweight="bold")

    # Shared colorbar
    fig.subplots_adjust(right=0.90, hspace=0.15, wspace=0.1)
    cax = fig.add_axes([0.92, 0.12, 0.018, 0.76])
    sm  = plt.cm.ScalarMappable(cmap=cmap_et, norm=norm_et)
    cb  = fig.colorbar(sm, cax=cax, ticks=et_bounds)
    cb.ax.set_yticklabels([f"{b:.1f}" for b in et_bounds], fontsize=7)
    cb.set_label("Mean ET (mm/day)", fontsize=9)

    fig.suptitle(f"{model_name} — Seasonal Mean ET (2009–2014) · Morocco",
                 fontsize=11, fontweight="bold", y=0.98)
    fname = os.path.join(path_out, "et_seasonal_maps.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Seasonal ET maps saved: {fname}")
