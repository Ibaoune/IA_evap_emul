"""
src/extreme/plots_extreme.py
============================
Evaluation plots for extreme events only.
Metrics: MAE_Q95, MAE_Q99, MAE_Q995

Functions:
    plot_spatial_extreme     — spatial maps of MAE Q95/Q99/Q99.5
    plot_boxplots_extreme    — boxplot distributions of extreme metrics
"""

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib.colors import BoundaryNorm

from postproc.utils import points_to_grid, compute_smart_bounds


EXTREME_METRICS = ["MAE_Q95", "MAE_Q99", "MAE_Q995"]
EXTREME_LABELS  = ["MAE Q95 (mm/day)", "MAE Q99 (mm/day)", "MAE Q99.5 (mm/day)"]


# ==========================================================
# 1. Spatial maps of extreme metrics
# ==========================================================

def plot_spatial_extreme(df, model_name, gdf_morocco, path_out):
    """
    One figure — 3 panels (one per extreme metric) spatial map.

    Args:
        df          : pd.DataFrame with lat, lon, MAE_Q95, MAE_Q99, MAE_Q995
        model_name  : str
        gdf_morocco : GeoDataFrame
        path_out    : output directory
    """

    save_dir = path_out

    lat_vals = np.array(sorted(df["lat"].unique()))
    lon_vals = np.array(sorted(df["lon"].unique()))
    minx, miny, maxx, maxy = gdf_morocco.total_bounds

    fig, axes = plt.subplots(
        1, 3, figsize=(18, 5), dpi=150,
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    fig.suptitle(f"{model_name} — Extreme ET Errors (Spatial Distribution)",
                 fontsize=11, fontweight="bold")

    for ax, col, lbl in zip(axes, EXTREME_METRICS, EXTREME_LABELS):
        if col not in df.columns:
            ax.set_title(f"{lbl}\n(not available)"); continue

        all_vals = df[col].dropna()
        bounds   = compute_smart_bounds(all_vals)
        ncolors  = len(bounds) - 1
        cmap     = mpl.colors.ListedColormap(
            mpl.cm.YlOrRd(np.linspace(0, 1, ncolors))
        )
        norm = BoundaryNorm(boundaries=bounds, ncolors=ncolors)

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
        gl.xlabel_style = {"size": 6}; gl.ylabel_style = {"size": 6}

        vals = all_vals
        ax.set_title(
            f"{lbl}\nmin={vals.min():.3f}  med={vals.median():.3f}  max={vals.max():.3f}",
            fontsize=8.5, fontweight="bold"
        )

        # Inset colorbar
        cax = ax.inset_axes([1.04, 0, 0.05, 1])
        mpl.colorbar.ColorbarBase(
            cax, cmap=cmap, norm=norm, boundaries=bounds,
            ticks=bounds, orientation="vertical"
        ).set_label(lbl, fontsize=7)
        cax.tick_params(labelsize=6)

    plt.tight_layout()
    fname = os.path.join(save_dir, "spatial_extreme_metrics.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Extreme spatial maps saved: {fname}")


# ==========================================================
# 2. Boxplots of extreme metrics
# ==========================================================

def plot_boxplots_extreme(df, model_name, path_out):
    """
    One figure — boxplot of MAE Q95 / Q99 / Q99.5 across all Morocco pixels.
    """
    save_dir = path_out

    fig, axes = plt.subplots(1, 3, figsize=(13, 5), dpi=130)
    fig.suptitle(
        f"{model_name} — Extreme Error Distributions (MAE at high ET quantiles)",
        fontsize=11, fontweight="bold"
    )

    color = "#E74C3C"
    for ax, col, lbl in zip(axes, EXTREME_METRICS, EXTREME_LABELS):
        data = df[col].dropna().values if col in df.columns else np.array([])
        if len(data) == 0:
            ax.set_title(f"{lbl}\n(not available)"); continue

        med = np.median(data)
        ax.boxplot(data, patch_artist=True, showfliers=False,
                   medianprops=dict(color="black", lw=2),
                   boxprops=dict(facecolor=color, alpha=0.75),
                   whiskerprops=dict(color=color),
                   capprops=dict(color=color))
        ax.text(1, med, f" {med:.3f}", va="center", ha="left",
                fontsize=9, color="darkred", fontweight="bold")
        ax.set_title(lbl, fontsize=9, fontweight="bold")
        ax.set_xticklabels([model_name], fontsize=8)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fname = os.path.join(save_dir, "boxplots_extreme.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Extreme boxplots saved: {fname}")



def plot_boxplots_extreme_by_season(df_seasonal, model_name, path_out):
    """Boxplots of MAE Q95/Q99/Q99.5 split by season."""
    from postproc.metrics import SEASON_DEF, SEASON_LABELS
    color_map = {"DJF": "#3498DB", "MAM": "#2ECC71",
                 "JJA": "#E74C3C", "SON": "#E67E22"}
    seasons = list(SEASON_DEF.keys())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=130)
    fig.suptitle(f"{model_name} — Extreme errors by season", fontsize=11, fontweight="bold")

    for ax, col, lbl in zip(axes, EXTREME_METRICS, EXTREME_LABELS):
        data_list = [df_seasonal[df_seasonal["Season"] == s][col].dropna().values
                     for s in seasons]
        colors    = [color_map[s] for s in seasons]
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
    fname = os.path.join(path_out, "boxplots_extreme_by_season.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Extreme seasonal boxplots saved: {fname}")


def plot_boxplots_extreme_by_region(df_regional, model_name, path_out):
    """Boxplots of MAE Q95/Q99/Q99.5 split by region."""
    from postproc.metrics import REGION_ORDER, REGION_LABELS
    color_map = {"north": "#3498DB", "north_east": "#2ECC71",
                 "east": "#E67E22",  "south": "#E74C3C"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=130)
    fig.suptitle(f"{model_name} — Extreme errors by region", fontsize=11, fontweight="bold")

    for ax, col, lbl in zip(axes, EXTREME_METRICS, EXTREME_LABELS):
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
    fname = os.path.join(path_out, "boxplots_extreme_by_region.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Extreme regional boxplots saved: {fname}")