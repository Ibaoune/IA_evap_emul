"""
src/extreme/plots_extreme.py
============================
Evaluation plots for extreme events only.
Metrics: MAE_Q95, MAE_Q99, MAE_Q995
Style comes from the PlotConfig passed as `style` (configs/plots/).

Functions:
    plot_spatial_extreme     — spatial maps of MAE Q95/Q99/Q99.5
    plot_boxplots_extreme    — boxplot distributions of extreme metrics
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs

from postproc.metrics         import SEASON_DEF, SEASON_LABELS, REGION_ORDER, REGION_LABELS
from postproc.utils           import df_to_grid, boxplots_by_group, draw_map_frame, add_colorbar
from postproc.plot_config     import default_style


EXTREME_METRICS = ["MAE_Q95", "MAE_Q99", "MAE_Q995"]
EXTREME_LABELS  = ["MAE Q95 (mm/day)", "MAE Q99 (mm/day)", "MAE Q99.5 (mm/day)"]


# ==========================================================
# 1. Spatial maps of extreme metrics
# ==========================================================

def plot_spatial_extreme(df, model_name, gdf_morocco, path_out, style=None):
    """
    One figure — 3 panels (one per extreme metric) spatial map.

    Args:
        df          : pd.DataFrame with lat, lon, MAE_Q95, MAE_Q99, MAE_Q995
        model_name  : str
        gdf_morocco : GeoDataFrame
        path_out    : output directory
        style       : PlotConfig (default: configs/plots/default.yaml)
    """
    style = style or default_style()
    st    = style.figure("single", "extreme_spatial")

    lat_vals = np.array(sorted(df["lat"].unique()))
    lon_vals = np.array(sorted(df["lon"].unique()))

    fig, axes = plt.subplots(
        1, 3, figsize=st.figsize, dpi=st.dpi,
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    fig.suptitle(f"{model_name} — Extreme ET Errors (Spatial Distribution)",
                 fontsize=st.suptitle_size, fontweight="bold")

    for ax, col, lbl in zip(axes, EXTREME_METRICS, EXTREME_LABELS):
        if col not in df.columns:
            ax.set_title(f"{lbl}\n(not available)"); continue

        all_vals = df[col].dropna()
        cb       = style.colorbar(col, all_vals)

        grid = df_to_grid(df, col, lat_vals, lon_vals)
        lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)

        ax.pcolormesh(lon2d, lat2d, grid, cmap=cb.cmap, norm=cb.norm, shading="auto")
        draw_map_frame(ax, gdf_morocco, st)

        vals = all_vals
        ax.set_title(
            f"{cb.label}\nmin={vals.min():.3f}  med={vals.median():.3f}  max={vals.max():.3f}",
            fontsize=st.title_size, fontweight="bold"
        )

        # Inset colorbar
        add_colorbar(ax.inset_axes([1.04, 0, 0.05, 1]), cb, st)

    plt.tight_layout()
    fname = style.savefig(fig, os.path.join(path_out, "spatial_extreme_metrics"))
    print(f"  Extreme spatial maps saved: {fname}")


# ==========================================================
# 2. Boxplots of extreme metrics
# ==========================================================

def plot_boxplots_extreme(df, model_name, path_out, style=None):
    """
    One figure — boxplot of MAE Q95 / Q99 / Q99.5 across all Morocco pixels.
    """
    style = style or default_style()
    st    = style.figure("single", "extreme_boxplots")

    fig, axes = plt.subplots(1, 3, figsize=st.figsize, dpi=st.dpi)
    fig.suptitle(
        f"{model_name} — Extreme Error Distributions (MAE at high ET quantiles)",
        fontsize=st.suptitle_size, fontweight="bold"
    )

    color = style.colors["extreme_box"]
    for ax, col, lbl in zip(axes, EXTREME_METRICS, EXTREME_LABELS):
        data = df[col].dropna().values if col in df.columns else np.array([])
        if len(data) == 0:
            ax.set_title(f"{lbl}\n(not available)"); continue

        med = np.median(data)
        ax.boxplot(data, patch_artist=True, showfliers=st.show_fliers,
                   medianprops=dict(color="black", lw=2),
                   boxprops=dict(facecolor=color, alpha=st.box_alpha),
                   whiskerprops=dict(color=color),
                   capprops=dict(color=color))
        ax.text(1, med, f" {med:.3f}", va="center", ha="left",
                fontsize=st.annotation_size, color=style.colors["median_text"],
                fontweight="bold")
        ax.set_title(lbl, fontsize=st.title_size, fontweight="bold")
        ax.set_xticklabels([model_name], fontsize=st.tick_label_size)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fname = style.savefig(fig, os.path.join(path_out, "boxplots_extreme"))
    print(f"  Extreme boxplots saved: {fname}")


def plot_boxplots_extreme_by_season(df_seasonal, model_name, path_out, style=None):
    """Boxplots of MAE Q95/Q99/Q99.5 split by season."""
    style = style or default_style()
    st    = style.figure("single", "extreme_boxplots_by_season")

    fig, axes = plt.subplots(1, 3, figsize=st.figsize, dpi=st.dpi)
    fig.suptitle(f"{model_name} — Extreme errors by season",
                 fontsize=st.suptitle_size, fontweight="bold")

    boxplots_by_group(axes, df_seasonal, "Season", list(SEASON_DEF.keys()), SEASON_LABELS,
                       style.colors["seasons"], EXTREME_METRICS, EXTREME_LABELS, st)

    plt.tight_layout()
    fname = style.savefig(fig, os.path.join(path_out, "boxplots_extreme_by_season"))
    print(f"  Extreme seasonal boxplots saved: {fname}")


def plot_boxplots_extreme_by_region(df_regional, model_name, path_out, style=None):
    """Boxplots of MAE Q95/Q99/Q99.5 split by region."""
    style = style or default_style()
    st    = style.figure("single", "extreme_boxplots_by_region")

    fig, axes = plt.subplots(1, 3, figsize=st.figsize, dpi=st.dpi)
    fig.suptitle(f"{model_name} — Extreme errors by region",
                 fontsize=st.suptitle_size, fontweight="bold")

    boxplots_by_group(axes, df_regional, "region", REGION_ORDER, REGION_LABELS,
                       style.colors["regions"], EXTREME_METRICS, EXTREME_LABELS, st)

    plt.tight_layout()
    fname = style.savefig(fig, os.path.join(path_out, "boxplots_extreme_by_region"))
    print(f"  Extreme regional boxplots saved: {fname}")
