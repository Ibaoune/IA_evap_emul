"""
src/intercomparison/plots_intercomp.py
=======================================
Intercomparison plots — comparing multiple trained model configs.
One figure per region, 4 subplots (R², KGE, RMSE, MAE Q95).
Models appear on the x-axis, one bar per model.

Adapted from plot_all_lmdz35_barfigs_by_region in the notebook.
"""

import os
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import cartopy.crs as ccrs
from matplotlib.colors import BoundaryNorm
import numpy as np
import pandas as pd

from postproc.utils   import points_to_grid, compute_smart_bounds
from postproc.metrics import SEASON_DEF, SEASON_LABELS, REGION_ORDER, REGION_LABELS
from postproc.main.plots_main import METRIC_CFG, MAIN_METRICS, MAIN_LABELS


INTERCOMP_METRICS = ["R2",  "KGE",  "RMSE",          "MAE_Q95"]
INTERCOMP_LABELS  = ["R²",  "KGE",  "RMSE (mm/day)", "MAE Q95 (mm/day)"]

# Default color palette — one color per model config label
DEFAULT_COLORS = [
    "#2980B9", "#E67E22", "#C0392B", "#8E44AD",
    "#27AE60", "#16A085", "#2C3E50", "#F39C12",
    "#D35400", "#1ABC9C", "#7F8C8D", "#BDC3C7",
]


# ==========================================================
# Main intercomparison function
# ==========================================================

def plot_intercomparison_by_region(dfs, model_labels, path_out, colors=None):
    """
    4 figures (one per region), each with 4 subplots (R², KGE, RMSE, MAE Q95).
    Models appear on the x-axis.

    Args:
        dfs          : list of pd.DataFrame — one per model config,
                       each with columns: lat, lon, region, R2, KGE, RMSE, MAE_Q95
        model_labels : list of str — label for each model (e.g. "XGB conf0")
        path_out     : output directory
        colors       : optional list of hex colors (one per model)
    """
    if colors is None:
        colors = DEFAULT_COLORS[:len(model_labels)]

    os.makedirs(path_out, exist_ok=True)

    bar_w = 0.6
    x     = np.arange(len(model_labels))

    for region in REGION_ORDER:
        region_lbl = REGION_LABELS[region]

        # Median per model for this region
        def _median(df, metric):
            sub = df[df["region"] == region]
            return sub[metric].median() if metric in sub.columns else np.nan

        fig, axes = plt.subplots(2, 2, figsize=(max(12, len(model_labels) * 1.8), 9), dpi=140)
        axes = axes.ravel()
        fig.suptitle(
            f"Model Intercomparison — {region_lbl}\n"
            "Median of per-pixel metrics (test period)",
            fontsize=12, fontweight="bold"
        )

        for ax, metric, mlbl in zip(axes, INTERCOMP_METRICS, INTERCOMP_LABELS):
            vals   = [_median(df, metric) for df in dfs]
            bars   = ax.bar(x, vals, bar_w, color=colors, alpha=0.88,
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
                    fontsize=8, fontweight="bold", color="#2C3E50"
                )

            ax.set_xticks(x)
            ax.set_xticklabels(model_labels, fontsize=9, fontweight="bold",
                               rotation=20, ha="right")
            ax.set_title(mlbl, fontsize=12, fontweight="bold", pad=10, color="#1A2940")
            ax.axhline(0, color="#444", lw=0.8)
            ax.grid(axis="y", alpha=0.25, lw=0.5, linestyle="--")
            ax.spines[["top", "right"]].set_visible(False)

        handles = [
            mpatches.Patch(facecolor=c, alpha=0.85, label=lbl)
            for c, lbl in zip(colors, model_labels)
        ]
        fig.legend(handles=handles, loc="lower center", ncol=min(len(model_labels), 6),
                   fontsize=9, framealpha=0.7, bbox_to_anchor=(0.5, -0.02))

        plt.tight_layout(rect=[0, 0.05, 1, 1])
        fname = os.path.join(path_out, f"intercomp_{region}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Intercomparison figure saved: {fname}")




# ==========================================================
# Spatial metrics — one figure per metric, one panel per model
# ==========================================================

def plot_spatial_metrics_intercomp(dfs, model_labels, gdf_morocco, path_out):
    """
    One figure per metric — each model as a panel side by side.
    """
    save_dir = os.path.join(path_out, "metric_distribution")
    os.makedirs(save_dir, exist_ok=True)

    minx, miny, maxx, maxy = gdf_morocco.total_bounds
    n_models = len(model_labels)

    for col, (lbl, cmap_base) in METRIC_CFG.items():
        # Check at least one df has this column
        if not any(col in df.columns for df in dfs):
            continue

        # Shared bounds across all models for this metric
        all_vals = pd.concat([df[col].dropna() for df in dfs if col in df.columns])
        bounds   = compute_smart_bounds(all_vals)
        if col == "KGE":
            bounds = [b for b in bounds if b <= 1.0]
            if bounds[-1] < 1.0:
                bounds.append(1.0)

        ncolors = len(bounds) - 1
        cmap    = mpl.colors.ListedColormap(cmap_base(np.linspace(0, 1, ncolors)))
        norm    = BoundaryNorm(boundaries=bounds, ncolors=ncolors)

        fig, axes = plt.subplots(
            1, n_models,
            figsize=(5 * n_models, 5),
            dpi=150,
            subplot_kw={"projection": ccrs.PlateCarree()}
        )
        if n_models == 1:
            axes = [axes]

        for ax, df, label in zip(axes, dfs, model_labels):
            if col not in df.columns:
                ax.set_visible(False); continue

            lat_vals = np.array(sorted(df["lat"].unique()))
            lon_vals = np.array(sorted(df["lon"].unique()))
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

            vals = df[col].dropna()
            ax.set_title(
                f"{label}\nmed={vals.median():.3f}  min={vals.min():.3f}  max={vals.max():.3f}",
                fontsize=7.5, fontweight="bold"
            )

        fig.subplots_adjust(right=0.88, wspace=0.3)
        cax = fig.add_axes([0.90, 0.12, 0.018, 0.76])
        mpl.colorbar.ColorbarBase(
            cax, cmap=cmap, norm=norm, boundaries=bounds,
            ticks=bounds, orientation="vertical"
        ).set_label(lbl, fontsize=8)
        cax.tick_params(labelsize=7)

        fig.suptitle(f"Intercomparison — {lbl} spatial distribution",
                     fontsize=11, fontweight="bold")
        fname = os.path.join(save_dir, f"spatial_{col}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)

    print(f"  Intercomp spatial metrics saved: {save_dir}")


# ==========================================================
# Boxplots — global
# ==========================================================

def plot_boxplots_global_intercomp(dfs, model_labels, colors, path_out):
    """One figure — all metrics, one box per model."""
    save_dir = os.path.join(path_out, "metric_distribution")
    os.makedirs(save_dir, exist_ok=True)

    n_metrics = len(MAIN_METRICS)
    fig, axes = plt.subplots(1, n_metrics, figsize=(24, 5), dpi=130)
    fig.suptitle("Intercomparison — Metric distributions (all Morocco pixels)",
                 fontsize=11, fontweight="bold")

    for ax, col, lbl in zip(axes, MAIN_METRICS, MAIN_LABELS):
        data_list = [df[col].dropna().values for df in dfs]
        bp = ax.boxplot(data_list, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", lw=2))
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color); patch.set_alpha(0.75)
        for i, (d, c) in enumerate(zip(data_list, colors)):
            if len(d):
                med = np.median(d)
                ax.text(i + 1, med, f"{med:.2f}", ha="center", va="bottom",
                        fontsize=7, fontweight="bold", color=c)
        ax.set_xticks(range(1, len(model_labels) + 1))
        ax.set_xticklabels(model_labels, fontsize=7, rotation=20, ha="right")
        ax.set_title(lbl, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    plt.tight_layout()
    fname = os.path.join(save_dir, "boxplots_global.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Intercomp global boxplots saved: {fname}")


# ==========================================================
# Boxplots — by season
# ==========================================================

def plot_boxplots_by_season_intercomp(dfs_seasonal, model_labels, colors, path_out):
    """One figure — metrics × seasons, one box per model per season."""
    save_dir = os.path.join(path_out, "metric_distribution")
    os.makedirs(save_dir, exist_ok=True)

    seasons   = list(SEASON_DEF.keys())
    n_metrics = len(MAIN_METRICS)
    n_models  = len(model_labels)
    box_w     = 0.8 / n_models

    fig, axes = plt.subplots(1, n_metrics, figsize=(28, 5), dpi=130)
    fig.suptitle("Intercomparison — Metrics by season", fontsize=11, fontweight="bold")

    for ax, col, lbl in zip(axes, MAIN_METRICS, MAIN_LABELS):
        tick_pos, tick_lbl = [], []
        for s_idx, season in enumerate(seasons):
            center = s_idx * (n_models * box_w + 0.4)
            tick_pos.append(center + (n_models - 1) * box_w / 2)
            tick_lbl.append(SEASON_LABELS[season])
            for m_idx, (df_s, color) in enumerate(zip(dfs_seasonal, colors)):
                data = df_s[df_s["Season"] == season][col].dropna().values
                pos  = center + m_idx * box_w
                ax.boxplot(data, positions=[pos], widths=box_w * 0.85,
                           patch_artist=True, showfliers=False,
                           medianprops=dict(color="black", lw=1.5),
                           boxprops=dict(facecolor=color, alpha=0.75),
                           whiskerprops=dict(color=color, lw=0.7),
                           capprops=dict(color=color, lw=0.7),
                           manage_ticks=False)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lbl, fontsize=7, rotation=15, ha="right")
        ax.set_title(lbl, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    handles = [mpatches.Patch(facecolor=c, alpha=0.75, label=l)
               for c, l in zip(colors, model_labels)]
    fig.legend(handles=handles, loc="lower center", ncol=n_models,
               fontsize=8, bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fname = os.path.join(save_dir, "boxplots_by_season.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Intercomp seasonal boxplots saved: {fname}")


# ==========================================================
# Boxplots — by region
# ==========================================================

def plot_boxplots_by_region_intercomp(dfs, model_labels, colors, path_out):
    """One figure — metrics × regions, one box per model per region."""
    save_dir = os.path.join(path_out, "metric_distribution")
    os.makedirs(save_dir, exist_ok=True)

    n_metrics = len(MAIN_METRICS)
    n_models  = len(model_labels)
    box_w     = 0.8 / n_models

    fig, axes = plt.subplots(1, n_metrics, figsize=(28, 5), dpi=130)
    fig.suptitle("Intercomparison — Metrics by region", fontsize=11, fontweight="bold")

    for ax, col, lbl in zip(axes, MAIN_METRICS, MAIN_LABELS):
        tick_pos, tick_lbl = [], []
        for r_idx, region in enumerate(REGION_ORDER):
            center = r_idx * (n_models * box_w + 0.4)
            tick_pos.append(center + (n_models - 1) * box_w / 2)
            tick_lbl.append(REGION_LABELS[region])
            for m_idx, (df, color) in enumerate(zip(dfs, colors)):
                data = df[df["region"] == region][col].dropna().values
                pos  = center + m_idx * box_w
                ax.boxplot(data, positions=[pos], widths=box_w * 0.85,
                           patch_artist=True, showfliers=False,
                           medianprops=dict(color="black", lw=1.5),
                           boxprops=dict(facecolor=color, alpha=0.75),
                           whiskerprops=dict(color=color, lw=0.7),
                           capprops=dict(color=color, lw=0.7),
                           manage_ticks=False)
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lbl, fontsize=7, rotation=15, ha="right")
        ax.set_title(lbl, fontsize=9)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    handles = [mpatches.Patch(facecolor=c, alpha=0.75, label=l)
               for c, l in zip(colors, model_labels)]
    fig.legend(handles=handles, loc="lower center", ncol=n_models,
               fontsize=8, bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fname = os.path.join(save_dir, "boxplots_by_region.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Intercomp regional boxplots saved: {fname}")


# ==========================================================
# ET mean spatial map — one panel per model
# ==========================================================

def plot_mean_et_map_intercomp(obs_list, pred_list, lats_list, lons_list,
                                model_labels, gdf_morocco, path_out):
    """
    One figure — obs panel + one pred panel per model.
    """
    save_dir = os.path.join(path_out, "et_distribution")
    os.makedirs(save_dir, exist_ok=True)

    et_bounds = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2,
                           1.4, 1.6, 1.8, 2.0, 2.2, 2.4])
    N_et    = len(et_bounds) - 1
    cmap_et = mpl.colors.ListedColormap(
        plt.cm.get_cmap("RdYlGn", N_et)(np.linspace(0, 1, N_et))
    )
    norm_et = mpl.colors.BoundaryNorm(boundaries=et_bounds, ncolors=N_et)
    minx, miny, maxx, maxy = gdf_morocco.total_bounds

    # obs + one panel per model
    n_panels = 1 + len(model_labels)
    fig, axes = plt.subplots(
        1, n_panels, figsize=(5 * n_panels, 5), dpi=150,
        subplot_kw={"projection": ccrs.PlateCarree()}
    )

    def _draw(ax, valid_lats, valid_lons, vals_1d, title):
        grid, lat_u, lon_u = points_to_grid(valid_lats, valid_lons, vals_1d)
        lon2d, lat2d = np.meshgrid(lon_u, lat_u)
        ax.pcolormesh(lon2d, lat2d, grid, cmap=cmap_et, norm=norm_et, shading="auto")
        gdf_morocco.boundary.plot(ax=ax, edgecolor="black", linewidth=1.0)
        ax.set_extent([minx - 0.3, maxx + 0.3, miny - 0.3, maxy + 0.3])
        gl = ax.gridlines(draw_labels=True, linewidth=0.4,
                          color="gray", alpha=0.4, linestyle="--")
        gl.top_labels = False; gl.right_labels = False
        gl.xlabel_style = {"size": 5}; gl.ylabel_style = {"size": 5}
        mu = np.nanmean(grid)
        ax.set_title(f"{title}\nmean={mu:.3f}", fontsize=8, fontweight="bold")

    # Obs panel (use first model's obs)
    obs_mean = obs_list[0].mean(dim="time").values
    _draw(axes[0], lats_list[0], lons_list[0], obs_mean, "Obs (LMDZ35)")

    for ax, pred_da, valid_lats, valid_lons, label in zip(
        axes[1:], pred_list, lats_list, lons_list, model_labels
    ):
        pred_mean = pred_da.mean(dim="time").values
        _draw(ax, valid_lats, valid_lons, pred_mean, label)

    fig.subplots_adjust(right=0.90, wspace=0.28)
    cax = fig.add_axes([0.92, 0.12, 0.016, 0.76])
    sm  = plt.cm.ScalarMappable(cmap=cmap_et, norm=norm_et)
    cb  = fig.colorbar(sm, cax=cax, ticks=et_bounds)
    cb.ax.set_yticklabels([f"{b:.1f}" for b in et_bounds], fontsize=7)
    cb.set_label("Mean ET (mm/day)", fontsize=9)

    fig.suptitle("Intercomparison — Mean ET (test period)",
                 fontsize=11, fontweight="bold")
    fname = os.path.join(save_dir, "mean_et_map.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Intercomp mean ET map saved: {fname}")


# ==========================================================
# ET seasonal spatial maps — one figure per season
# ==========================================================

def plot_et_seasonal_maps_intercomp(obs_list, pred_list, lats_list, lons_list,
                                     model_labels, gdf_morocco, path_out):
    """
    One figure per season — obs panel + one pred panel per model.
    """
    save_dir = os.path.join(path_out, "et_distribution")
    os.makedirs(save_dir, exist_ok=True)

    et_bounds = np.array([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2,
                           1.4, 1.6, 1.8, 2.0, 2.2, 2.4])
    N_et    = len(et_bounds) - 1
    cmap_et = mpl.colors.ListedColormap(
        plt.cm.get_cmap("RdYlGn", N_et)(np.linspace(0, 1, N_et))
    )
    norm_et = mpl.colors.BoundaryNorm(boundaries=et_bounds, ncolors=N_et)
    minx, miny, maxx, maxy = gdf_morocco.total_bounds

    # Get months from first obs
    try:
        months = pd.DatetimeIndex(obs_list[0].time.values).month.values
    except Exception:
        months = np.array([t.month for t in obs_list[0].time.values])

    n_panels = 1 + len(model_labels)

    for season, month_list in SEASON_DEF.items():
        mask = np.isin(months, month_list)

        fig, axes = plt.subplots(
            1, n_panels, figsize=(5 * n_panels, 5), dpi=150,
            subplot_kw={"projection": ccrs.PlateCarree()}
        )

        def _draw(ax, valid_lats, valid_lons, da, title):
            try:
                da_months = pd.DatetimeIndex(da.time.values).month.values
            except Exception:
                da_months = np.array([t.month for t in da.time.values])
            da_mask = np.isin(da_months, month_list)
            vals = da.isel(time=da_mask).mean(dim="time").values
            grid, lat_u, lon_u = points_to_grid(valid_lats, valid_lons, vals)
            lon2d, lat2d = np.meshgrid(lon_u, lat_u)
            ax.pcolormesh(lon2d, lat2d, grid, cmap=cmap_et, norm=norm_et, shading="auto")
            gdf_morocco.boundary.plot(ax=ax, edgecolor="black", linewidth=1.0)
            ax.set_extent([minx - 0.3, maxx + 0.3, miny - 0.3, maxy + 0.3])
            gl = ax.gridlines(draw_labels=True, linewidth=0.4,
                            color="gray", alpha=0.4, linestyle="--")
            gl.top_labels = False; gl.right_labels = False
            gl.xlabel_style = {"size": 5}; gl.ylabel_style = {"size": 5}
            mu = np.nanmean(grid)
            ax.set_title(f"{title}\nmean={mu:.3f}", fontsize=8, fontweight="bold")

        _draw(axes[0], lats_list[0], lons_list[0], obs_list[0], "Obs (LMDZ35)")
        for ax, pred_da, valid_lats, valid_lons, label in zip(
            axes[1:], pred_list, lats_list, lons_list, model_labels
        ):
            _draw(ax, valid_lats, valid_lons, pred_da, label)

        fig.subplots_adjust(right=0.90, wspace=0.28)
        cax = fig.add_axes([0.92, 0.12, 0.016, 0.76])
        sm  = plt.cm.ScalarMappable(cmap=cmap_et, norm=norm_et)
        cb  = fig.colorbar(sm, cax=cax, ticks=et_bounds)
        cb.ax.set_yticklabels([f"{b:.1f}" for b in et_bounds], fontsize=7)
        cb.set_label("Mean ET (mm/day)", fontsize=9)

        fig.suptitle(f"Intercomparison — {SEASON_LABELS[season]} Mean ET",
                     fontsize=11, fontweight="bold")
        fname = os.path.join(save_dir, f"et_seasonal_{season}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Intercomp seasonal ET map saved: {fname}")


# ==========================================================
# ET boxplots by season and region
# ==========================================================

def plot_et_boxplots_intercomp(obs_list, pred_list, lats_list, lons_list,
                                model_labels, colors, gdfs_regions, path_out):
    """
    Two figures:
      - ET boxplots by season (one panel per region)
      - ET boxplots by region (one panel per season)
    """
    from postproc.utils import get_region_point_indices
    save_dir = os.path.join(path_out, "et_distribution")
    os.makedirs(save_dir, exist_ok=True)

    obs_color = "#2C2C2C"


    seasons    = list(SEASON_DEF.keys())
    n_models   = len(model_labels)
    spacing    = 0.9
    gap        = 1.5
    group_w    = (n_models + 1) * spacing   # +1 for obs

    # ── Figure 1: by season (one panel per region) ────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(24, 5), sharey=False)
    fig.suptitle("Intercomparison — Regional Mean ET by Season",
                 fontsize=11, fontweight="bold")

    for ax, region in zip(axes, REGION_ORDER):
        region_indices = get_region_point_indices(
            lats_list[0], lons_list[0], gdfs_regions
        )
        pt_idx = region_indices.get(region, np.array([]))
        if len(pt_idx) == 0:
            ax.set_title(REGION_LABELS[region]); continue

        tick_centers = []

        for s_idx, season in enumerate(seasons):
            center = s_idx * (group_w + gap)
            tick_centers.append(center + group_w / 2)

            try:
                obs_months = pd.DatetimeIndex(obs_list[0].time.values).month.values
            except Exception:
                obs_months = np.array([t.month for t in obs_list[0].time.values])
            obs_mask = np.isin(obs_months, SEASON_DEF[season])
            obs_mean = obs_list[0].isel(points=pt_idx).mean(dim="points").values
            data = obs_mean[obs_mask]; data = data[~np.isnan(data)]
            ax.boxplot(data, positions=[center], widths=0.7, patch_artist=True,
                       showfliers=False,
                       medianprops=dict(color="black", lw=1.5),
                       boxprops=dict(facecolor=obs_color, alpha=0.75),
                       whiskerprops=dict(color=obs_color, lw=0.8),
                       capprops=dict(color=obs_color, lw=0.8),
                       manage_ticks=False)

            for m_idx, (pred_da, color) in enumerate(zip(pred_list, colors)):
                try:
                    pred_months = pd.DatetimeIndex(pred_da.time.values).month.values
                except Exception:
                    pred_months = np.array([t.month for t in pred_da.time.values])
                pred_mask = np.isin(pred_months, SEASON_DEF[season])
                pred_mean = pred_da.isel(points=pt_idx).mean(dim="points").values
                data = pred_mean[pred_mask]; data = data[~np.isnan(data)]
                pos  = center + (m_idx + 1) * spacing
                ax.boxplot(data, positions=[pos], widths=0.7, patch_artist=True,
                           showfliers=False,
                           medianprops=dict(color="black", lw=1.5),
                           boxprops=dict(facecolor=color, alpha=0.75),
                           whiskerprops=dict(color=color, lw=0.8),
                           capprops=dict(color=color, lw=0.8),
                           manage_ticks=False)

        ax.set_xticks(tick_centers)
        ax.set_xticklabels([SEASON_LABELS[s] for s in seasons],
                           fontsize=7, rotation=15, ha="right")
        ax.set_title(REGION_LABELS[region], fontsize=9, fontweight="bold")
        ax.set_ylabel("ET (mm/day)")
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    handles = [mpatches.Patch(facecolor=obs_color, alpha=0.75, label="Obs")] + \
              [mpatches.Patch(facecolor=c, alpha=0.75, label=l)
               for c, l in zip(colors, model_labels)]
    fig.legend(handles=handles, loc="lower center", ncol=n_models + 1,
               fontsize=8, bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fname = os.path.join(save_dir, "et_boxplots_by_season.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Intercomp ET by season saved: {fname}")

    # ── Figure 2: by region (one panel per season) ─────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(24, 5), sharey=False)
    fig.suptitle("Intercomparison — Regional Mean ET by Region",
                 fontsize=11, fontweight="bold")

    for ax, season in zip(axes, seasons):
        tick_centers = []
        try:
            obs_months = pd.DatetimeIndex(obs_list[0].time.values).month.values
        except Exception:
            obs_months = np.array([t.month for t in obs_list[0].time.values])
        obs_mask = np.isin(obs_months, SEASON_DEF[season])

        for r_idx, region in enumerate(REGION_ORDER):
            region_indices = get_region_point_indices(
                lats_list[0], lons_list[0], gdfs_regions
            )
            pt_idx = region_indices.get(region, np.array([]))
            if len(pt_idx) == 0:
                continue

            center = r_idx * (group_w + gap)
            tick_centers.append(center + group_w / 2)

            obs_mean = obs_list[0].isel(points=pt_idx).mean(dim="points").values
            data = obs_mean[obs_mask]; data = data[~np.isnan(data)]
            ax.boxplot(data, positions=[center], widths=0.7, patch_artist=True,
                       showfliers=False,
                       medianprops=dict(color="black", lw=1.5),
                       boxprops=dict(facecolor=obs_color, alpha=0.75),
                       whiskerprops=dict(color=obs_color, lw=0.8),
                       capprops=dict(color=obs_color, lw=0.8),
                       manage_ticks=False)

            for m_idx, (pred_da, color) in enumerate(zip(pred_list, colors)):
                try:
                    pred_months = pd.DatetimeIndex(pred_da.time.values).month.values
                except Exception:
                    pred_months = np.array([t.month for t in pred_da.time.values])
                pred_mask = np.isin(pred_months, SEASON_DEF[season])
                pred_mean = pred_da.isel(points=pt_idx).mean(dim="points").values
                data = pred_mean[pred_mask]; data = data[~np.isnan(data)]
                pos  = center + (m_idx + 1) * spacing
                ax.boxplot(data, positions=[pos], widths=0.7, patch_artist=True,
                           showfliers=False,
                           medianprops=dict(color="black", lw=1.5),
                           boxprops=dict(facecolor=color, alpha=0.75),
                           whiskerprops=dict(color=color, lw=0.8),
                           capprops=dict(color=color, lw=0.8),
                           manage_ticks=False)

        ax.set_xticks(tick_centers)
        ax.set_xticklabels([REGION_LABELS[r] for r in REGION_ORDER],
                           fontsize=7, rotation=15, ha="right")
        ax.set_title(SEASON_LABELS[season], fontsize=9, fontweight="bold")
        ax.set_ylabel("ET (mm/day)")
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)

    handles = [mpatches.Patch(facecolor=obs_color, alpha=0.75, label="Obs")] + \
              [mpatches.Patch(facecolor=c, alpha=0.75, label=l)
               for c, l in zip(colors, model_labels)]
    fig.legend(handles=handles, loc="lower center", ncol=n_models + 1,
               fontsize=8, bbox_to_anchor=(0.5, -0.05))
    plt.tight_layout(rect=[0, 0.08, 1, 1])
    fname = os.path.join(save_dir, "et_boxplots_by_region.png")
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Intercomp ET by region saved: {fname}")