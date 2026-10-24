"""
postprocessing.py
=================
Reads the predictions pickle saved by train.py and produces:
  - NetCDF file with predictions + observations
  - All evaluation figures (spatial maps, metrics, boxplots, time series, loss curves)
  - Metrics CSV

Usage:
    python postprocessing.py --config configs/XGB/conf0.yaml
    python postprocessing.py --config configs/LSTM/conf1.yaml
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import BoundaryNorm
from shapely.ops import unary_union
from shapely.geometry import Point
from sklearn.metrics import r2_score
import cartopy.crs as ccrs

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import load_config
from utils  import vprint, set_verbose, build_morocco_mask


# ==========================================================
# Metrics
# ==========================================================

def compute_metrics(obs, pred):
    """Point-wise metrics for a single pixel time series."""
    r = np.corrcoef(pred, obs)[0, 1] if (np.std(obs) > 0 and np.std(pred) > 0) else 0.0

    beta  = np.mean(pred) / np.mean(obs) if np.mean(obs) > 0 else np.nan
    gamma = (
        (np.std(pred) / np.mean(pred)) / (np.std(obs) / np.mean(obs))
        if not np.isclose(np.mean(pred), 0) and not np.isclose(np.std(obs), 0)
        else 1.0
    )
    kge = (
        1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)
        if not np.isnan(gamma) else np.nan
    )

    ss_res = np.sum((obs - pred) ** 2)
    ss_tot = np.sum((obs - np.mean(obs)) ** 2)
    nse    = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    q95     = np.percentile(obs, 95)
    mask_q  = obs >= q95
    mae_q95 = np.mean(np.abs(pred[mask_q] - obs[mask_q])) if mask_q.sum() > 0 else np.nan

    ubrmse = np.sqrt(
        np.mean(((pred - np.mean(pred)) - (obs - np.mean(obs))) ** 2)
    )

    return {
        "R²":            r2_score(obs, pred),
        "Correlation":   r,
        "KGE":           kge,
        "Bias (%)":      100 * (np.mean(pred) - np.mean(obs)) / np.mean(obs),
        "Bias (mm/day)": float(np.mean(pred) - np.mean(obs)),
        "RMSE":          np.sqrt(np.mean((pred - obs) ** 2)),
        "MAE":           np.mean(np.abs(pred - obs)),
        "MAE_Q95":       mae_q95,
        "NSE":           nse,
        "ubRMSE":        ubrmse,
    }


def build_metrics_dataframe(obs_stacked, pred_stacked, valid_lats, valid_lons, model_name):
    rows = []
    for lat_val, lon_val in zip(valid_lats, valid_lons):
        pt   = (lat_val, lon_val)
        obs  = obs_stacked.sel(points=pt).values
        pred = pred_stacked.sel(points=pt).values
        m    = compute_metrics(obs, pred)
        m["Model"] = model_name
        m["lat"]   = lat_val
        m["lon"]   = lon_val
        rows.append(m)
    return pd.DataFrame(rows)


# ==========================================================
# Plots
# ==========================================================

def plot_spatial_check(Y_test_np, Y_pred, lat_vals, lon_vals, model_name, path_out):
    """Mean spatial map: obs vs pred vs bias."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), subplot_kw={'aspect': 'equal'})

    mean_obs  = np.nanmean(Y_test_np, axis=0)
    mean_pred = np.nanmean(Y_pred,    axis=0)
    diff      = mean_pred - mean_obs
    vmin, vmax = np.nanpercentile(mean_obs, [2, 98])

    im0 = axes[0].pcolormesh(lon_vals, lat_vals, mean_obs,  cmap='YlGnBu', vmin=vmin, vmax=vmax)
    axes[0].set_title('Mean ET — obs (LMDZ35)')
    plt.colorbar(im0, ax=axes[0], label='mm/day')

    im1 = axes[1].pcolormesh(lon_vals, lat_vals, mean_pred, cmap='YlGnBu', vmin=vmin, vmax=vmax)
    axes[1].set_title(f'Mean ET — {model_name}')
    plt.colorbar(im1, ax=axes[1], label='mm/day')

    lim = np.nanpercentile(np.abs(diff), 95)
    im2 = axes[2].pcolormesh(lon_vals, lat_vals, diff, cmap='RdBu_r', vmin=-lim, vmax=lim)
    axes[2].set_title('Bias (pred − obs)')
    plt.colorbar(im2, ax=axes[2], label='mm/day')

    for ax in axes:
        ax.set_xlabel('Lon'); ax.set_ylabel('Lat')

    plt.suptitle(f'{model_name} ET downscaling — test period mean', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(path_out, 'spatial_check.png'), dpi=150, bbox_inches='tight')
    plt.close(fig)
    vprint("  Spatial check plot saved.")


def plot_spatial_metrics(df_metrics, model_name, morocco_gdf, path_out):
    """Spatial maps of all evaluation metrics."""
    METRIC_CFG = {
        "KGE":           {"bounds": [-1,-0.5,0,0.2,0.4,0.5,0.6,0.7,0.8,0.9,1.0],  "cmap": mpl.cm.RdYlGn,  "label": "KGE"},
        "R²":            {"bounds": [-1,-0.5,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], "cmap": mpl.cm.YlGn, "label": "R²"},
        "Correlation":   {"bounds": [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],  "cmap": mpl.cm.YlGn,  "label": "Pearson r"},
        "Bias (%)":      {"bounds": [-60,-30,-20,-10,-5,-2,0,2,5,10,20,30,60],      "cmap": mpl.cm.RdBu_r, "label": "Bias (%)"},
        "Bias (mm/day)": {"bounds": [-5,-3,-2,-1,-0.5,-0.2,0,0.2,0.5,1,2,3,5],     "cmap": mpl.cm.RdBu_r, "label": "Bias (mm/day)"},
        "RMSE":          {"bounds": [0,0.2,0.5,1.0,1.5,2.0,2.5,3.0,4.0,5.0],       "cmap": mpl.cm.YlOrRd, "label": "RMSE (mm/day)"},
        "MAE":           {"bounds": [0,0.1,0.3,0.5,0.8,1.0,1.5,2.0,3.0,4.0],       "cmap": mpl.cm.YlOrRd, "label": "MAE (mm/day)"},
        "MAE_Q95":       {"bounds": [0,0.5,1.0,1.5,2.0,2.5,3.0,4.0,5.0,6.0],       "cmap": mpl.cm.YlOrRd, "label": "MAE Q95 (mm/day)"},
        "NSE":           {"bounds": [-1,-0.5,0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0], "cmap": mpl.cm.RdYlGn, "label": "NSE"},
        "ubRMSE":        {"bounds": [0,0.2,0.5,1.0,1.5,2.0,2.5,3.0,4.0,5.0],       "cmap": mpl.cm.YlOrRd, "label": "ubRMSE (mm/day)"},
    }

    lat_vals = np.array(sorted(df_metrics["lat"].unique()))
    lon_vals = np.array(sorted(df_metrics["lon"].unique()))
    minx, miny, maxx, maxy = morocco_gdf.total_bounds

    fig, axes = plt.subplots(2, 5, figsize=(25, 10),
                             subplot_kw={"projection": ccrs.PlateCarree()}, dpi=150)
    axes_flat = axes.flatten()

    for ax, (metric, cfg_m) in zip(axes_flat, METRIC_CFG.items()):
        grid = np.full((len(lat_vals), len(lon_vals)), np.nan)
        for _, row in df_metrics.iterrows():
            li = np.searchsorted(lat_vals, row["lat"])
            lo = np.searchsorted(lon_vals, row["lon"])
            grid[li, lo] = row[metric]

        bounds  = cfg_m["bounds"]
        ncolors = len(bounds) - 1
        cmap    = mpl.colors.ListedColormap(cfg_m["cmap"](np.linspace(0, 1, ncolors)))
        norm    = BoundaryNorm(boundaries=bounds, ncolors=ncolors)

        ax.pcolormesh(lon_vals, lat_vals, grid, cmap=cmap, norm=norm, shading="auto")
        morocco_gdf.boundary.plot(ax=ax, edgecolor="black", linewidth=1.5)
        ax.set_extent([minx, maxx, miny, maxy])

        vmin_val = np.nanmin(df_metrics[metric].values)
        vmax_val = np.nanmax(df_metrics[metric].values)
        med_val  = np.nanmedian(df_metrics[metric].values)
        ax.set_title(f"{model_name} — {metric}\n"
                     f"min={vmin_val:.3f}  med={med_val:.3f}  max={vmax_val:.3f}", fontsize=9)

        cbar_ax = ax.inset_axes([1.05, 0, 0.05, 1])
        mpl.colorbar.ColorbarBase(
            cbar_ax, cmap=cmap, norm=norm, boundaries=bounds,
            ticks=bounds, orientation="vertical"
        ).set_label(cfg_m["label"], fontsize=8)

    plt.suptitle(f"Spatial evaluation — {model_name} ET downscaling", y=1.02)
    plt.subplots_adjust(left=0.03, right=0.92, wspace=0.3)
    plt.savefig(os.path.join(path_out, "spatial_metrics.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    vprint("  Spatial metrics map saved.")


def plot_boxplots(df_metrics, model_name, path_out):
    """Boxplot of each metric distribution."""
    METRICS = ["KGE", "R²", "Correlation", "Bias (%)", "Bias (mm/day)",
               "RMSE", "MAE", "MAE_Q95", "NSE", "ubRMSE"]

    fig, axes = plt.subplots(2, 5, figsize=(25, 8))
    for ax, metric in zip(axes.flatten(), METRICS):
        data    = df_metrics[metric].dropna()
        med_val = data.median()
        ax.boxplot(data, patch_artist=True,
                   boxprops=dict(facecolor='steelblue', alpha=0.7),
                   medianprops=dict(color='green', linewidth=2))
        ax.text(1, med_val, f' {med_val:.3f}', va='center', ha='left',
                fontsize=9, color='red', fontweight='bold')
        ax.set_title(f"{metric}\nmed={med_val:.3f}", fontsize=9)
        ax.set_xticklabels([model_name])
        ax.grid(alpha=0.3)

    plt.suptitle(f"{model_name} — metric distributions (test period)")
    plt.tight_layout()
    plt.savefig(os.path.join(path_out, "boxplots_metrics.png"), dpi=150)
    plt.close(fig)
    vprint("  Boxplots saved.")


def plot_losses_median(all_train_losses, all_val_losses, model_name, path_out):
    """Median train/val loss across all grid points with IQR band."""
    max_len = max(len(l) for l in all_train_losses)

    def _pad(losses):
        return np.array([
            np.pad(l, (0, max_len - len(l)), constant_values=np.nan)
            for l in losses
        ])

    train_arr = _pad(all_train_losses)
    val_arr   = _pad(all_val_losses)
    epochs    = np.arange(1, max_len + 1)

    median_train = np.nanmedian(train_arr, axis=0)
    median_val   = np.nanmedian(val_arr,   axis=0)
    p25_train    = np.nanpercentile(train_arr, 25, axis=0)
    p75_train    = np.nanpercentile(train_arr, 75, axis=0)
    p25_val      = np.nanpercentile(val_arr,   25, axis=0)
    p75_val      = np.nanpercentile(val_arr,   75, axis=0)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(epochs, median_train, label="Median Train", color="steelblue", lw=2)
    ax.fill_between(epochs, p25_train, p75_train, alpha=0.2, color="steelblue", label="Train IQR")
    ax.plot(epochs, median_val,   label="Median Val",   color="coral",     lw=2)
    ax.fill_between(epochs, p25_val,   p75_val,   alpha=0.2, color="coral",     label="Val IQR")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{model_name} — Median train/val loss across all grid points")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(path_out, "loss_curves_median.png"), dpi=150)
    plt.close(fig)
    vprint("  Median loss plot saved.")


# ==========================================================
# Region helpers
# ==========================================================

REGION_NAMES = {0: "east", 1: "north", 2: "north_east", 3: "south"}

COLORS = {
    "LMDZ35": "#2c2c2c",
    "RCAN"  : "#ff6b6b",
    "LSTM"  : "#3a86ff",
}

REGION_COLORS = {
    "east"      : "#e07b39",
    "north"     : "#3a86ff",
    "north_east": "#06d6a0",
    "south"     : "#ef233c",
}


def build_cluster_lookup(valid_lats, valid_lons, shp_regions_path):
    """
    Assign each valid Morocco point to a region.
    Unassigned points are snapped to the nearest region.
    """
    region_names = list(REGION_NAMES.values())
    id_map       = {v: k for k, v in REGION_NAMES.items()}

    gdfs = {}
    for name in region_names:
        shp = os.path.join(shp_regions_path, f"{name}.shp")
        if os.path.exists(shp):
            gdfs[name] = gpd.read_file(shp)

    coords_df  = pd.DataFrame({"lat": valid_lats, "lon": valid_lons})
    gdf_coords = gpd.GeoDataFrame(
        coords_df, geometry=gpd.points_from_xy(coords_df.lon, coords_df.lat)
    )
    gdf_coords["region"] = np.nan

    for rname, gdf in gdfs.items():
        region_geom = unary_union(gdf.geometry)
        mask = gdf_coords.geometry.within(region_geom)
        gdf_coords.loc[mask, "region"] = id_map[rname]

    # Snap unassigned points to nearest region
    unassigned = gdf_coords[gdf_coords["region"].isna()].copy()
    if len(unassigned) > 0:
        region_geoms = {rname: unary_union(gdf.geometry) for rname, gdf in gdfs.items()}
        for idx, row in unassigned.iterrows():
            pt      = Point(row.geometry)
            nearest = min(region_geoms, key=lambda r: region_geoms[r].distance(pt))
            gdf_coords.loc[idx, "region"] = id_map[nearest]

    cluster_lookup = (
        gdf_coords[["lat", "lon", "region"]]
        .dropna(subset=["region"])
        .copy()
    )
    cluster_lookup["region"] = cluster_lookup["region"].astype(int)

    for cid, cname in REGION_NAMES.items():
        n = (cluster_lookup["region"] == cid).sum()
        vprint(f"  {cname:12s} → {n} points")
    return cluster_lookup


def build_cluster_ts(cluster_lookup, obs_stacked, pred_stacked, model_name, n_time):
    """Build per-region time series arrays for obs and predictions."""
    _preds     = {"LMDZ35": obs_stacked, model_name: pred_stacked}
    cluster_ts = {}

    for cid, cname in REGION_NAMES.items():
        sub = cluster_lookup[cluster_lookup["region"] == cid]
        pts = list(zip(sub["lat"], sub["lon"]))
        cluster_ts[cname] = {}

        for mname, da in _preds.items():
            cols = []
            for pt in pts:
                try:
                    v = da.sel(points=pt).values
                    if not np.any(np.isnan(v)):
                        cols.append(v)
                except Exception:
                    pass
            cluster_ts[cname][mname] = (
                np.column_stack(cols) if cols else np.empty((n_time, 0))
            )

        n = cluster_ts[cname]["LMDZ35"].shape[1]
        vprint(f"  {cname:12s} → {n} clean points")

    return cluster_ts


def plot_timeseries_regional_mean(test_years, cluster_ts, model_name, path_out):
    """One figure per test year — regional mean ET: obs vs prediction."""
    regions = list(REGION_NAMES.values())
    models  = [model_name]

    for year in test_years:
        time_idx = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31', freq='D')
        n_days   = len(time_idx)

        obs_data  = cluster_ts[regions[0]]["LMDZ35"]
        n_actual  = min(n_days, obs_data.shape[0])
        time_idx  = time_idx[:n_actual]
        n_days    = n_actual 

        fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
        fig.suptitle(
            f"Regional Mean Evaporation — {year}\n(LMDZ35 vs {model_name})",
            fontsize=13
        )

        for ax, region in zip(axes, regions):
            obs_data  = cluster_ts[region]["LMDZ35"]
            era5_mean = np.nanmean(obs_data, axis=1)[:n_days]

            ax.fill_between(time_idx, era5_mean, alpha=0.15, color=COLORS["LMDZ35"])
            ax.plot(time_idx, era5_mean, color=COLORS["LMDZ35"], lw=2, label="LMDZ35")

            for m in models:
                pred_data = cluster_ts[region].get(m)
                if pred_data is not None and pred_data.shape[1] > 0:
                    pred_mean = np.nanmean(pred_data, axis=1)[:n_days]
                    ax.plot(time_idx, pred_mean,
                            color=COLORS.get(m, "#999999"), lw=1.4, ls="--",
                            label=m, alpha=0.9)

            mu = np.nanmean(era5_mean)
            ax.axhline(mu, color="gray", lw=1, ls=":", label=f"LMDZ35 mean ({mu:.2f})")
            ax.set_ylabel("mm/day")
            ax.set_title(region.replace("_", " ").title(), fontsize=10, loc="left")
            ax.legend(fontsize=8, ncol=6, loc="upper right")
            ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
            ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%b"))

        plt.tight_layout()
        fname = os.path.join(path_out, f"timeseries_regional_mean_{year}.png")
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close(fig)
        vprint(f"  Time-series plot saved: {fname}")


def plot_metrics_by_region(df_metrics, model_name, path_out):
    """Bar chart of median metrics per region."""
    metrics_to_plot = ["KGE", "NSE", "R²", "Correlation",
                       "Bias (%)", "Bias (mm/day)",
                       "RMSE", "MAE", "MAE_Q95", "ubRMSE"]
    region_list = list(REGION_NAMES.values())

    fig, axes = plt.subplots(2, 5, figsize=(28, 10))
    fig.suptitle(f"{model_name} — Median metrics by region", fontsize=12)

    for ax, metric in zip(axes.flatten(), metrics_to_plot):
        vals   = []
        colors = []
        for region in region_list:
            cid  = [k for k, v in REGION_NAMES.items() if v == region][0]
            data = df_metrics.loc[df_metrics["region"] == cid, metric].dropna()
            vals.append(data.median() if not data.empty else np.nan)
            colors.append(REGION_COLORS[region])

        bars = ax.bar(region_list, vals, color=colors,
                      alpha=0.85, edgecolor="black", linewidth=0.5)

        for bar, val in zip(bars, vals):
            if not np.isnan(val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01 * (1 if val >= 0 else -1),
                    f"{val:.3f}",
                    ha="center", va="bottom" if val >= 0 else "top",
                    fontsize=8, fontweight="bold"
                )

        if metric in ["KGE", "R²", "Correlation", "NSE"]:
            ax.axhline(1, color="green", lw=1, ls="--", alpha=0.5)
            ax.axhline(0, color="gray",  lw=0.8, ls=":")
        if metric in ["Bias (%)", "Bias (mm/day)"]:
            ax.axhline(0, color="gray", lw=1, ls="--")
        if metric in ["RMSE", "MAE", "MAE_Q95", "ubRMSE"]:
            ax.axhline(0, color="gray", lw=0.8, ls=":")

        ax.set_title(metric, fontsize=10)
        ax.set_ylabel(metric)
        ax.set_xticklabels([r.replace("_", "\n") for r in region_list], fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(path_out, "metrics_by_region.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    vprint("  Metrics by region plot saved.")


# ==========================================================
# Save NetCDF
# ==========================================================

def save_netcdf(Y_pred, Y_test_np, time_test, lat_vals, lon_vals, model_name, path_out):
    """Save predictions and observations to NetCDF."""
    ds_out = xr.Dataset(
        {
            "evap_pred": (["time", "lat", "lon"], Y_pred),
            "evap_obs":  (["time", "lat", "lon"], Y_test_np),
        },
        coords={"time": time_test, "lat": lat_vals, "lon": lon_vals},
    )
    ds_out["evap_pred"].attrs["units"] = "mm/day"
    ds_out["evap_obs"].attrs["units"]  = "mm/day"
    ds_out.attrs["model"] = model_name

    nc_path = os.path.join(path_out, f"{model_name}_ET_predictions.nc")
    ds_out.to_netcdf(nc_path)
    vprint(f"  NetCDF saved: {nc_path}")


# ==========================================================
# Main
# ==========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="Same config yaml used for training, e.g. configs/XGB/conf0.yaml")
    args = parser.parse_args()

    cfg = load_config(config_path=args.config, train_mode=False)
    set_verbose(cfg.verbose)

    model_name = cfg.model_type.upper()
    path_out   = cfg.path_out   # results/<model>/<conf>/
    os.makedirs(path_out, exist_ok=True)

    vprint(f"=== Postprocessing  |  model={model_name}  conf={cfg.conf_name} ===")
    vprint(f"Reading results from: {path_out}")

    # ------------------------------------------------------------------
    # Load predictions pickle
    # ------------------------------------------------------------------
    preds_path = os.path.join(path_out, f"{cfg.model_type}_preds.pkl")
    with open(preds_path, "rb") as f:
        saved = pickle.load(f)

    model_preds      = saved["model_preds"]
    all_train_losses = saved["all_train_losses"]
    all_val_losses   = saved["all_val_losses"]
    nlat, nlon       = saved["nlat"], saved["nlon"]
    lat_fine         = saved["lats"]
    lon_fine         = saved["lons"]

    # ------------------------------------------------------------------
    # Reconstruct Y_pred and Y_test_np on the fine grid
    # ------------------------------------------------------------------
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

        # align lengths in case of seq_len offset
        n = min(len(ti), len(pred))
        Y_pred[ti[:n], li, lo]    = pred[:n]
        Y_test_np[ti[:n], li, lo] = te["target"].values[:n]
    
    morocco_mask_fine = build_morocco_mask(lat_fine, lon_fine, cfg.shp_regions_path)
    Y_pred[:,    ~morocco_mask_fine] = np.nan
    Y_test_np[:, ~morocco_mask_fine] = np.nan

    # ------------------------------------------------------------------
    # Loss curves  (LSTM / MLP only)
    # ------------------------------------------------------------------
    if all_train_losses:
        plot_losses_median(all_train_losses, all_val_losses, model_name, path_out)

    # ------------------------------------------------------------------
    # Save NetCDF
    # ------------------------------------------------------------------
    save_netcdf(Y_pred, Y_test_np, time_test, lat_fine, lon_fine, model_name, path_out)

    # ------------------------------------------------------------------
    # Spatial check plot
    # ------------------------------------------------------------------
    plot_spatial_check(Y_test_np, Y_pred, lat_fine, lon_fine, model_name, path_out)

    # ------------------------------------------------------------------
    # Point-wise metrics
    # ------------------------------------------------------------------
    obs_xr  = xr.DataArray(Y_test_np, dims=["time","lat","lon"],
                            coords={"time": time_test, "lat": lat_fine, "lon": lon_fine})
    pred_xr = xr.DataArray(Y_pred,    dims=["time","lat","lon"],
                            coords={"time": time_test, "lat": lat_fine, "lon": lon_fine})

    lon2d, lat2d = np.meshgrid(lon_fine, lat_fine)
    valid_mask   = morocco_mask_fine.ravel()
    valid_lats   = lat2d.ravel()[valid_mask]
    valid_lons   = lon2d.ravel()[valid_mask]

    obs_stacked  = obs_xr.stack(points=("lat","lon"))
    pred_stacked = pred_xr.stack(points=("lat","lon"))

    valid_midx   = pd.MultiIndex.from_arrays([valid_lats, valid_lons], names=["lat","lon"])
    obs_stacked  = obs_stacked.sel(points=valid_midx)
    pred_stacked = pred_stacked.sel(points=valid_midx)

    df_metrics = build_metrics_dataframe(
        obs_stacked, pred_stacked, valid_lats, valid_lons, model_name
    )

    # ------------------------------------------------------------------
    # Spatial metrics maps + boxplots
    # ------------------------------------------------------------------
    morocco_gdf = gpd.read_file(
        next(
            os.path.join(cfg.shp_regions_path, n)
            for n in os.listdir(cfg.shp_regions_path) if n.endswith(".shp")
        )
    )
    plot_spatial_metrics(df_metrics, model_name, morocco_gdf, path_out)
    plot_boxplots(df_metrics, model_name, path_out)

    # ------------------------------------------------------------------
    # Region time-series + bar charts
    # ------------------------------------------------------------------
    vprint("Building region cluster lookup...")
    cluster_lookup = build_cluster_lookup(valid_lats, valid_lons, cfg.shp_regions_path)

    df_metrics = df_metrics.merge(
        cluster_lookup[["lat", "lon", "region"]], on=["lat", "lon"], how="left"
    ).dropna(subset=["region"])
    df_metrics["region"] = df_metrics["region"].astype(int)

    vprint("Building cluster time series...")
    cluster_ts = build_cluster_ts(
        cluster_lookup, obs_stacked, pred_stacked,
        model_name=model_name,
        n_time=n_time,
    )

    plot_timeseries_regional_mean(cfg.test_years, cluster_ts, model_name, path_out)
    plot_metrics_by_region(df_metrics, model_name, path_out)

    # ------------------------------------------------------------------
    # Save metrics CSV
    # ------------------------------------------------------------------
    metrics_dir = os.path.join(path_out, "Metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    df_metrics.to_csv(os.path.join(metrics_dir, "metrics_per_point.csv"), index=False)

    vprint(f"All outputs saved in: {path_out}")
    vprint("=== Postprocessing completed ===")


if __name__ == "__main__":
    main()
