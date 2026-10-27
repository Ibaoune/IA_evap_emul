"""
plots.py
========
Quick-look figures written next to the predictions in results/<model_type>/<conf_name>/.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")   # no display on HPC nodes
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import geopandas as gpd

# ET colour scale: discrete brown -> green classes of 0.25 mm/day from 0 to 2,
# with one extra class for values above 2
ET_LEVELS = np.arange(0, 2.01, 0.25)


def plot_mean_maps(cfg, model_preds, lats, lons, nlat, nlon):
    """
    Three maps of Morocco over the test period:
    mean observed ET | mean predicted ET | bias (predicted - observed).

    Works for every model: model_preds holds one entry per Morocco pixel
    (same format for point-wise and spatial models).
    Saved as results/<model_type>/<conf_name>/<model_type>_mean_maps.png.
    """
    obs  = np.full((nlat, nlon), np.nan)
    pred = np.full((nlat, nlon), np.nan)
    for e in model_preds:
        li, lj = e["lat_idx"], e["lon_idx"]
        obs[li, lj]  = np.nanmean(e["te_df"]["target"].values)
        pred[li, lj] = np.nanmean(e["pred"])
    bias = pred - obs

    # observed and predicted share the fixed ET scale, so they can be compared
    et_cmap = plt.get_cmap("BrBG", len(ET_LEVELS))
    et_norm = BoundaryNorm(ET_LEVELS, et_cmap.N, extend="max")

    # bias: discrete classes symmetric around 0, limit rounded up to 0.1 mm/day
    blim = np.ceil(np.nanpercentile(np.abs(bias), 98) * 10) / 10
    blim = blim if blim > 0 else 0.1
    b_levels = np.linspace(-blim, blim, 9)
    b_cmap   = plt.get_cmap("RdBu_r", len(b_levels) + 1)
    b_norm   = BoundaryNorm(b_levels, b_cmap.N, extend="both")

    shp_files = [f for f in os.listdir(cfg.shp_path) if f.endswith(".shp")]
    border    = gpd.read_file(os.path.join(cfg.shp_path, shp_files[0])) if shp_files else None

    rows, cols = np.where(~np.isnan(obs))
    pad  = 0.5
    xlim = (lons[cols].min() - pad, lons[cols].max() + pad)
    ylim = (lats[rows].min() - pad, lats[rows].max() + pad)

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), constrained_layout=True)
    panels = [
        (obs,  "Observed ET (LMDZ35)",                     et_cmap, et_norm, ET_LEVELS, "ET (mm/day)"),
        (pred, f"Predicted ET ({cfg.model_type.upper()})", et_cmap, et_norm, ET_LEVELS, "ET (mm/day)"),
        (bias, "Bias (predicted − observed)",              b_cmap,  b_norm,  b_levels,  "Bias (mm/day)"),
    ]
    for ax, (grid, title, cmap, norm, levels, label) in zip(axes, panels):
        im = ax.pcolormesh(lons, lats, grid, cmap=cmap, norm=norm, shading="auto")
        if border is not None:
            border.boundary.plot(ax=ax, color="#333333", linewidth=0.8)
        ax.set_facecolor("#e6e6e6")         # outside Morocco, so low ET (near white) stays visible
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)
        ax.set_aspect("equal")
        stats = (f"min = {np.nanmin(grid):.3f}   mean = {np.nanmean(grid):.3f}   "
                 f"max = {np.nanmax(grid):.3f}")
        ax.set_title(f"{title}\n{stats}", fontsize=11)
        ax.set_xlabel("Longitude (°)")
        ax.set_ylabel("Latitude (°)")
        ax.grid(color="#cccccc", linewidth=0.4, linestyle="--")
        ax.set_axisbelow(True)
        cb = fig.colorbar(im, ax=ax, orientation="vertical", ticks=levels[::2],
                          shrink=0.8, pad=0.02, aspect=25)
        cb.set_label(label)

    y0, y1 = cfg.test_years[0], cfg.test_years[-1]
    period = f"{y0}" if y0 == y1 else f"{y0}–{y1}"
    fig.suptitle(f"{cfg.model_type.upper()} {cfg.conf_name} — mean ET over the test period {period}",
                 fontsize=13)

    path = os.path.join(cfg.path_out, f"{cfg.model_type}_mean_maps.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
