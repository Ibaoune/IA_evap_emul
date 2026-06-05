import os
import numpy as np
import torch
import geopandas as gpd
from shapely.ops import unary_union

from config import load_config
from preprocessing import preprocess_data
from evaluation import (
    build_cluster_lookup, build_cluster_ts,
    plot_timeseries_regional_mean, plot_metrics_by_region,
    REGION_NAMES,
    build_metrics_dataframe, plot_spatial_check,
    plot_spatial_metrics, plot_boxplots,
    save_netcdf,
)
from utils import vprint, set_verbose, load_model



def main():
    cfg = load_config(train_mode=False, path="config.yaml")
    set_verbose(cfg.verbose) 
    vprint(f"=== Starting evaluation ({cfg.model_type.upper()}) ===")

    path_out = os.path.join(cfg.path_out, cfg.model_type)
    os.makedirs(path_out, exist_ok=True)
    
    if cfg.model_type in ("lstm", "mlp", "xgb", "mlr_rf"):
        import pickle
        from evaluation import plot_losses_median
        from utils import build_morocco_mask

        preds_path = os.path.join(cfg.model_save_dir, cfg.model_type, f"{cfg.model_type}_preds.pkl")
        with open(preds_path, "rb") as f:
            saved = pickle.load(f)

        model_preds      = saved["model_preds"]
        all_train_losses = saved["all_train_losses"]
        all_val_losses   = saved["all_val_losses"]
        nlat, nlon       = saved["nlat"], saved["nlon"]
        lat_fine         = saved["lats"]
        lon_fine         = saved["lons"]

        # Reconstruct Y_pred and Y_test_np on fine grid from point-wise results
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
            li   = te["lat_idx"].values
            lo   = te["lon_idx"].values
            Y_pred[ti, li, lo]    = pred
            Y_test_np[ti, li, lo] = te["target"].values

        morocco_mask_fine = build_morocco_mask(lat_fine, lon_fine, cfg.shp_path)
        Y_pred[:,    ~morocco_mask_fine] = np.nan
        Y_test_np[:, ~morocco_mask_fine] = np.nan

        if all_train_losses:
            plot_losses_median(all_train_losses, all_val_losses, cfg.model_type.upper(), path_out)

    else:
        raise NotImplementedError(f"model_type '{cfg.model_type}' not supported.")

    # ------------------------------------------------------------------
    # Save NetCDF
    # ------------------------------------------------------------------
    save_netcdf(Y_pred, Y_test_np, time_test, lat_fine, lon_fine,
                cfg.model_type.upper(), path_out)

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------
    plot_spatial_check(Y_test_np, Y_pred, lat_fine, lon_fine,
                       cfg.model_type.upper(), path_out)

    # Build Morocco GeoDataFrame for metric maps
    morocco_gdf = gpd.read_file(
        next(f for f in [
            os.path.join(cfg.shp_path, n)
            for n in os.listdir(cfg.shp_path) if n.endswith(".shp")
        ])
    )

    # Point-wise metrics (stack spatial dims)
    import xarray as xr, pandas as pd
    from shapely.geometry import Point

    lon2d, lat2d = np.meshgrid(lon_fine, lat_fine)
    valid_mask   = morocco_mask_fine.ravel()
    valid_lats   = lat2d.ravel()[valid_mask]
    valid_lons   = lon2d.ravel()[valid_mask]

    obs_xr = xr.DataArray(Y_test_np, dims=["time","lat","lon"],
                           coords={"time": time_test, "lat": lat_fine, "lon": lon_fine})
    pred_xr = xr.DataArray(Y_pred,   dims=["time","lat","lon"],
                            coords={"time": time_test, "lat": lat_fine, "lon": lon_fine})

    obs_stacked  = obs_xr.stack(points=("lat","lon"))
    pred_stacked = pred_xr.stack(points=("lat","lon"))

    valid_midx = pd.MultiIndex.from_arrays([valid_lats, valid_lons], names=["lat","lon"])
    obs_stacked  = obs_stacked.sel(points=valid_midx)
    pred_stacked = pred_stacked.sel(points=valid_midx)

    df_metrics = build_metrics_dataframe(
        obs_stacked, pred_stacked, valid_lats, valid_lons, cfg.model_type.upper()
    )

    plot_spatial_metrics(df_metrics, cfg.model_type.upper(), morocco_gdf, path_out)
    plot_boxplots(df_metrics, cfg.model_type.upper(), path_out)

    # ------------------------------------------------------------------
    # Region-based plots (time series + median metrics per region)
    # ------------------------------------------------------------------
    vprint("Building region cluster lookup...")
    cluster_lookup = build_cluster_lookup(valid_lats, valid_lons, cfg.shp_regions_path)

    # Attach region to df_metrics for the bar chart
    df_metrics = df_metrics.merge(
        cluster_lookup[["lat", "lon", "region"]], on=["lat", "lon"], how="left"
    ).dropna(subset=["region"])
    df_metrics["region"] = df_metrics["region"].astype(int)

    vprint("Building cluster time series...")
    cluster_ts = build_cluster_ts(
        cluster_lookup, obs_stacked, pred_stacked,
        model_name=cfg.model_type.upper(),
        n_time=len(time_test),
    )

    plot_timeseries_regional_mean(cfg.test_years, cluster_ts, cfg.model_type.upper(), path_out)
    plot_metrics_by_region(df_metrics, cfg.model_type.upper(), path_out)

    # Save metrics CSV
    metrics_dir = os.path.join(path_out, "Metrics")
    os.makedirs(metrics_dir, exist_ok=True)
    df_metrics.to_csv(os.path.join(metrics_dir, "metrics_per_point.csv"), index=False)

    vprint(f"All outputs saved in: {path_out}")
    vprint("=== Evaluation completed ===")


if __name__ == "__main__":
    main()