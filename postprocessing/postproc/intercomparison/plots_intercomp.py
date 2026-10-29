"""
src/intercomparison/plots_intercomp.py
=======================================
Intercomparison plots — comparing multiple trained model configs.
One figure per region, 4 subplots (R², KGE, RMSE, MAE Q95).
Models appear on the x-axis, one bar per model.
Style comes from the PlotConfig passed as `style` (configs/plots/); model
colors come from its colors.models list.

Adapted from plot_all_lmdz35_barfigs_by_region in the notebook.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
import numpy as np
import pandas as pd

from postproc.utils       import (points_to_grid, get_region_point_indices, df_to_grid,
                                  months_of, draw_map_frame, add_colorbar)
from postproc.metrics     import SEASON_DEF, SEASON_LABELS, REGION_ORDER, REGION_LABELS
from postproc.plot_config import default_style
from postproc.main.plots_main import SPATIAL_METRICS, MAIN_METRICS, MAIN_LABELS


INTERCOMP_METRICS = ["R2",  "KGE",  "RMSE",          "MAE_Q95"]
INTERCOMP_LABELS  = ["R²",  "KGE",  "RMSE (mm/day)", "MAE Q95 (mm/day)"]


def _box(ax, data, pos, width, color, st, lw):
    ax.boxplot(data, positions=[pos], widths=width, patch_artist=True,
               showfliers=st.show_fliers,
               medianprops=dict(color="black", lw=1.5),
               boxprops=dict(facecolor=color, alpha=st.box_alpha),
               whiskerprops=dict(color=color, lw=lw),
               capprops=dict(color=color, lw=lw),
               manage_ticks=False)


# ==========================================================
# Main intercomparison function
# ==========================================================

def plot_intercomparison_by_region(dfs, model_labels, path_out, colors=None, style=None):
    """
    4 figures (one per region), each with 4 subplots (R², KGE, RMSE, MAE Q95).
    Models appear on the x-axis.

    Args:
        dfs          : list of pd.DataFrame — one per model config,
                       each with columns: lat, lon, region, R2, KGE, RMSE, MAE_Q95
        model_labels : list of str — label for each model (e.g. "XGB conf0")
        path_out     : output directory
        colors       : optional list of colors (one per model; default: colors.models)
        style        : PlotConfig (default: configs/plots/default.yaml)
    """
    style = style or default_style()
    st    = style.figure("intercomp", "bars_by_region")
    if colors is None:
        colors = style.model_colors(len(model_labels))

    os.makedirs(path_out, exist_ok=True)

    bar_w = 0.6
    x     = np.arange(len(model_labels))
    width = max(st.min_width, len(model_labels) * st.width_per_model)

    for region in REGION_ORDER:
        region_lbl = REGION_LABELS[region]

        # Median per model for this region
        def _median(df, metric):
            sub = df[df["region"] == region]
            return sub[metric].median() if metric in sub.columns else np.nan

        fig, axes = plt.subplots(2, 2, figsize=(width, st.height), dpi=st.dpi)
        axes = axes.ravel()
        fig.suptitle(
            f"Model Intercomparison — {region_lbl}\n"
            "Median of per-pixel metrics (test period)",
            fontsize=st.suptitle_size, fontweight="bold"
        )

        for ax, metric, mlbl in zip(axes, INTERCOMP_METRICS, INTERCOMP_LABELS):
            vals   = [_median(df, metric) for df in dfs]
            bars   = ax.bar(x, vals, bar_w, color=colors, alpha=st.bar_alpha,
                            edgecolor="white", linewidth=0.6)

            for bar, v in zip(bars, vals):
                if np.isnan(v):
                    continue
                offset = max(abs(v) * 0.025, 0.003)
                ypos   = v + offset if v >= 0 else v - offset
                ax.text(
                    bar.get_x() + bar.get_width() / 2, ypos,
                    f"{v:.3f}", ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=st.annotation_size, fontweight="bold",
                    color=style.colors["bar_text"]
                )

            ax.set_xticks(x)
            ax.set_xticklabels(model_labels, fontsize=st.tick_label_size, fontweight="bold",
                               rotation=st.tick_rotation, ha="right")
            ax.set_title(mlbl, fontsize=st.title_size, fontweight="bold", pad=10, color="#1A2940")
            ax.axhline(0, color="#444", lw=0.8)
            ax.grid(axis="y", alpha=0.25, lw=0.5, linestyle="--")
            ax.spines[["top", "right"]].set_visible(False)

        handles = [
            mpatches.Patch(facecolor=c, alpha=0.85, label=lbl)
            for c, lbl in zip(colors, model_labels)
        ]
        fig.legend(handles=handles, loc="lower center", ncol=min(len(model_labels), 6),
                   fontsize=st.legend_size, framealpha=0.7, bbox_to_anchor=(0.5, -0.02))

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        fname = style.savefig(fig, os.path.join(path_out, f"intercomp_{region}"))
        print(f"  Intercomparison figure saved: {fname}")


# ==========================================================
# Spatial metrics — one figure per metric, one panel per model
# ==========================================================

def plot_spatial_metrics_intercomp(dfs, model_labels, gdf_morocco, path_out, style=None):
    """
    One figure per metric — each model as a panel side by side.
    """
    style = style or default_style()
    st    = style.figure("intercomp", "spatial_metrics")

    save_dir = os.path.join(path_out, "metric_distribution")
    os.makedirs(save_dir, exist_ok=True)

    n_models = len(model_labels)
    pw, ph   = st.panel_size

    for col in SPATIAL_METRICS:
        # Check at least one df has this column
        if not any(col in df.columns for df in dfs):
            continue

        # Shared colorbar across all models for this metric
        all_vals = pd.concat([df[col].dropna() for df in dfs if col in df.columns])
        cb       = style.colorbar(col, all_vals)

        fig, axes = plt.subplots(
            1, n_models,
            figsize=(pw * n_models, ph),
            dpi=st.dpi,
            subplot_kw={"projection": ccrs.PlateCarree()}
        )
        if n_models == 1:
            axes = [axes]

        for ax, df, label in zip(axes, dfs, model_labels):
            if col not in df.columns:
                ax.set_visible(False); continue

            lat_vals = np.array(sorted(df["lat"].unique()))
            lon_vals = np.array(sorted(df["lon"].unique()))
            grid = df_to_grid(df, col, lat_vals, lon_vals)
            lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)

            ax.pcolormesh(lon2d, lat2d, grid, cmap=cb.cmap, norm=cb.norm, shading="auto")
            draw_map_frame(ax, gdf_morocco, st)

            vals = df[col].dropna()
            ax.set_title(
                f"{label}\nmed={vals.median():.3f}  min={vals.min():.3f}  max={vals.max():.3f}",
                fontsize=st.title_size, fontweight="bold"
            )

        fig.subplots_adjust(right=0.88, wspace=0.3)
        add_colorbar(fig.add_axes([0.90, 0.12, 0.018, 0.76]), cb, st)

        fig.suptitle(f"Intercomparison — {cb.label} spatial distribution",
                     fontsize=st.suptitle_size, fontweight="bold")
        style.savefig(fig, os.path.join(save_dir, f"spatial_{col}"))

    print(f"  Intercomp spatial metrics saved: {save_dir}")


# ==========================================================
# Boxplots — global
# ==========================================================

def plot_boxplots_global_intercomp(dfs, model_labels, colors, path_out, style=None):
    """One figure — all metrics, one box per model."""
    style = style or default_style()
    st    = style.figure("intercomp", "boxplots_global")

    save_dir = os.path.join(path_out, "metric_distribution")
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, len(MAIN_METRICS), figsize=st.figsize, dpi=st.dpi)
    fig.suptitle("Intercomparison — Metric distributions (all Morocco pixels)",
                 fontsize=st.suptitle_size, fontweight="bold")

    for ax, col, lbl in zip(axes, MAIN_METRICS, MAIN_LABELS):
        data_list = [df[col].dropna().values for df in dfs]
        bp = ax.boxplot(data_list, patch_artist=True, showfliers=st.show_fliers,
                        medianprops=dict(color="black", lw=2))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(st.box_alpha)
        for i, (d, c) in enumerate(zip(data_list, colors)):
            if len(d):
                med = np.median(d)
                ax.text(i + 1, med, f"{med:.2f}", ha="center", va="bottom",
                        fontsize=st.annotation_size, fontweight="bold", color=c)
        ax.set_xticks(range(1, len(model_labels) + 1))
        ax.set_xticklabels(model_labels, fontsize=st.tick_label_size,
                           rotation=st.tick_rotation, ha="right")
        ax.set_title(lbl, fontsize=st.title_size)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fname = style.savefig(fig, os.path.join(save_dir, "boxplots_global"))
    print(f"  Intercomp global boxplots saved: {fname}")


# ==========================================================
# Boxplots — by season / by region
# ==========================================================

def _grouped_intercomp_boxplots(st, style, dfs, group_col, groups, group_labels,
                                model_labels, colors, title, path_no_ext):
    """Metrics × groups (seasons or regions), one box per model per group."""
    n_models = len(model_labels)
    box_w    = 0.8 / n_models

    fig, axes = plt.subplots(1, len(MAIN_METRICS), figsize=st.figsize, dpi=st.dpi)
    fig.suptitle(title, fontsize=st.suptitle_size, fontweight="bold")

    for ax, col, lbl in zip(axes, MAIN_METRICS, MAIN_LABELS):
        tick_pos, tick_lbl = [], []
        for g_idx, group in enumerate(groups):
            center = g_idx * (n_models * box_w + 0.4)
            tick_pos.append(center + (n_models - 1) * box_w / 2)
            tick_lbl.append(group_labels[group])
            for m_idx, (df, color) in enumerate(zip(dfs, colors)):
                data = df[df[group_col] == group][col].dropna().values
                _box(ax, data, center + m_idx * box_w, box_w * 0.85, color, st, lw=0.7)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lbl, fontsize=st.tick_label_size,
                           rotation=st.tick_rotation, ha="right")
        ax.set_title(lbl, fontsize=st.title_size)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    handles = [mpatches.Patch(facecolor=c, alpha=st.box_alpha, label=l)
               for c, l in zip(colors, model_labels)]
    fig.legend(handles=handles, loc="lower center", ncol=n_models,
               fontsize=st.legend_size, bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    return style.savefig(fig, path_no_ext)


def plot_boxplots_by_season_intercomp(dfs_seasonal, model_labels, colors, path_out, style=None):
    """One figure — metrics × seasons, one box per model per season."""
    style = style or default_style()
    st    = style.figure("intercomp", "boxplots_by_season")

    save_dir = os.path.join(path_out, "metric_distribution")
    os.makedirs(save_dir, exist_ok=True)

    fname = _grouped_intercomp_boxplots(
        st, style, dfs_seasonal, "Season", list(SEASON_DEF.keys()), SEASON_LABELS,
        model_labels, colors, "Intercomparison — Metrics by season",
        os.path.join(save_dir, "boxplots_by_season"))
    print(f"  Intercomp seasonal boxplots saved: {fname}")


def plot_boxplots_by_region_intercomp(dfs, model_labels, colors, path_out, style=None):
    """One figure — metrics × regions, one box per model per region."""
    style = style or default_style()
    st    = style.figure("intercomp", "boxplots_by_region")

    save_dir = os.path.join(path_out, "metric_distribution")
    os.makedirs(save_dir, exist_ok=True)

    fname = _grouped_intercomp_boxplots(
        st, style, dfs, "region", REGION_ORDER, REGION_LABELS,
        model_labels, colors, "Intercomparison — Metrics by region",
        os.path.join(save_dir, "boxplots_by_region"))
    print(f"  Intercomp regional boxplots saved: {fname}")


# ==========================================================
# ET spatial maps — obs panel + one panel per model
# ==========================================================

def _et_map_figure(st, style, gdf_morocco, panels, title, path_no_ext):
    """
    One row of ET maps sharing the et_mean colorbar.
    panels: list of (valid_lats, valid_lons, values_1d, panel_title).
    """
    cb = style.colorbar("et_mean", np.concatenate([p[2] for p in panels]))

    pw, ph = st.panel_size
    fig, axes = plt.subplots(
        1, len(panels), figsize=(pw * len(panels), ph), dpi=st.dpi,
        subplot_kw={"projection": ccrs.PlateCarree()}
    )
    for ax, (valid_lats, valid_lons, vals_1d, panel_title) in zip(axes, panels):
        grid, lat_u, lon_u = points_to_grid(valid_lats, valid_lons, vals_1d)
        lon2d, lat2d = np.meshgrid(lon_u, lat_u)
        ax.pcolormesh(lon2d, lat2d, grid, cmap=cb.cmap, norm=cb.norm, shading="auto")
        draw_map_frame(ax, gdf_morocco, st)
        mu = np.nanmean(grid)
        ax.set_title(f"{panel_title}\nmean={mu:.3f}", fontsize=st.title_size, fontweight="bold")

    fig.subplots_adjust(right=0.90, wspace=0.28)
    add_colorbar(fig.add_axes([0.92, 0.12, 0.016, 0.76]), cb, st)

    fig.suptitle(title, fontsize=st.suptitle_size, fontweight="bold")
    return style.savefig(fig, path_no_ext)


def plot_mean_et_map_intercomp(obs_list, pred_list, lats_list, lons_list,
                                model_labels, gdf_morocco, path_out, style=None):
    """
    One figure — obs panel + one pred panel per model.
    """
    style = style or default_style()
    st    = style.figure("intercomp", "mean_et_map")

    save_dir = os.path.join(path_out, "et_distribution")
    os.makedirs(save_dir, exist_ok=True)

    # Obs panel (use first model's obs)
    panels = [(lats_list[0], lons_list[0], obs_list[0].mean(dim="time").values, "Obs (LMDZ35)")]
    panels += [(la, lo, pred_da.mean(dim="time").values, label)
               for pred_da, la, lo, label in zip(pred_list, lats_list, lons_list, model_labels)]

    fname = _et_map_figure(st, style, gdf_morocco, panels,
                           "Intercomparison — Mean ET (test period)",
                           os.path.join(save_dir, "mean_et_map"))
    print(f"  Intercomp mean ET map saved: {fname}")


def plot_et_seasonal_maps_intercomp(obs_list, pred_list, lats_list, lons_list,
                                     model_labels, gdf_morocco, path_out, style=None):
    """
    One figure per season — obs panel + one pred panel per model.
    """
    style = style or default_style()
    st    = style.figure("intercomp", "et_seasonal_maps")

    save_dir = os.path.join(path_out, "et_distribution")
    os.makedirs(save_dir, exist_ok=True)

    def _season_mean(da, month_list):
        return da.isel(time=np.isin(months_of(da), month_list)).mean(dim="time").values

    for season, month_list in SEASON_DEF.items():
        panels = [(lats_list[0], lons_list[0], _season_mean(obs_list[0], month_list),
                   "Obs (LMDZ35)")]
        panels += [(la, lo, _season_mean(pred_da, month_list), label)
                   for pred_da, la, lo, label in zip(pred_list, lats_list, lons_list,
                                                     model_labels)]
        fname = _et_map_figure(st, style, gdf_morocco, panels,
                               f"Intercomparison — {SEASON_LABELS[season]} Mean ET",
                               os.path.join(save_dir, f"et_seasonal_{season}"))
        print(f"  Intercomp seasonal ET map saved: {fname}")


# ==========================================================
# ET boxplots by season and region
# ==========================================================

def plot_et_boxplots_intercomp(obs_list, pred_list, lats_list, lons_list,
                                model_labels, colors, gdfs_regions, path_out, style=None):
    """
    Two figures:
      - ET boxplots by season (one panel per region)
      - ET boxplots by region (one panel per season)
    """
    style = style or default_style()
    st    = style.figure("intercomp", "et_boxplots")

    save_dir = os.path.join(path_out, "et_distribution")
    os.makedirs(save_dir, exist_ok=True)

    obs_color  = style.colors["obs"]
    seasons    = list(SEASON_DEF.keys())
    n_models   = len(model_labels)
    spacing    = 0.9
    gap        = 1.5
    group_w    = (n_models + 1) * spacing   # +1 for obs

    region_indices = get_region_point_indices(lats_list[0], lons_list[0], gdfs_regions)

    def _region_season(da, pt_idx, season):
        """Regional spatial mean of da, restricted to one season, NaNs dropped."""
        mask = np.isin(months_of(da), SEASON_DEF[season])
        data = da.isel(points=pt_idx).mean(dim="points").values[mask]
        return data[~np.isnan(data)]

    def _group(ax, center, pt_idx, season):
        _box(ax, _region_season(obs_list[0], pt_idx, season), center, 0.7,
             obs_color, st, lw=0.8)
        for m_idx, (pred_da, color) in enumerate(zip(pred_list, colors)):
            _box(ax, _region_season(pred_da, pt_idx, season),
                 center + (m_idx + 1) * spacing, 0.7, color, st, lw=0.8)

    def _finish(fig, axes_titles, path_no_ext):
        for ax, ax_title in axes_titles:
            ax.set_title(ax_title, fontsize=st.title_size, fontweight="bold")
            ax.set_ylabel("ET (mm/day)")
            ax.grid(axis="y", alpha=0.3)
            ax.spines[["top", "right"]].set_visible(False)
        handles = [mpatches.Patch(facecolor=obs_color, alpha=st.box_alpha, label="Obs")] + \
                  [mpatches.Patch(facecolor=c, alpha=st.box_alpha, label=l)
                   for c, l in zip(colors, model_labels)]
        fig.legend(handles=handles, loc="lower center", ncol=n_models + 1,
                   fontsize=st.legend_size, bbox_to_anchor=(0.5, -0.05))
        plt.tight_layout(rect=[0, 0.08, 1, 1])
        return style.savefig(fig, path_no_ext)

    # ── Figure 1: by season (one panel per region) ────────────────────
    fig, axes = plt.subplots(1, 4, figsize=st.figsize, dpi=st.dpi, sharey=False)
    fig.suptitle("Intercomparison — Regional Mean ET by Season",
                 fontsize=st.suptitle_size, fontweight="bold")

    titled = []
    for ax, region in zip(axes, REGION_ORDER):
        pt_idx = region_indices.get(region, np.array([]))
        if len(pt_idx) == 0:
            ax.set_title(REGION_LABELS[region]); continue

        tick_centers = []
        for s_idx, season in enumerate(seasons):
            center = s_idx * (group_w + gap)
            tick_centers.append(center + group_w / 2)
            _group(ax, center, pt_idx, season)

        ax.set_xticks(tick_centers)
        ax.set_xticklabels([SEASON_LABELS[s] for s in seasons],
                           fontsize=st.tick_label_size, rotation=st.tick_rotation, ha="right")
        titled.append((ax, REGION_LABELS[region]))

    fname = _finish(fig, titled, os.path.join(save_dir, "et_boxplots_by_season"))
    print(f"  Intercomp ET by season saved: {fname}")

    # ── Figure 2: by region (one panel per season) ─────────────────────
    fig, axes = plt.subplots(1, 4, figsize=st.figsize, dpi=st.dpi, sharey=False)
    fig.suptitle("Intercomparison — Regional Mean ET by Region",
                 fontsize=st.suptitle_size, fontweight="bold")

    titled = []
    for ax, season in zip(axes, seasons):
        tick_centers = []
        for r_idx, region in enumerate(REGION_ORDER):
            pt_idx = region_indices.get(region, np.array([]))
            if len(pt_idx) == 0:
                continue
            center = r_idx * (group_w + gap)
            tick_centers.append(center + group_w / 2)
            _group(ax, center, pt_idx, season)

        ax.set_xticks(tick_centers)
        ax.set_xticklabels([REGION_LABELS[r] for r in REGION_ORDER],
                           fontsize=st.tick_label_size, rotation=st.tick_rotation, ha="right")
        titled.append((ax, SEASON_LABELS[season]))

    fname = _finish(fig, titled, os.path.join(save_dir, "et_boxplots_by_region"))
    print(f"  Intercomp ET by region saved: {fname}")
