"""
src/metrics.py
==============
All metric computation functions, adapted from the notebook.
Used by both main/ and extreme/ evaluations.
"""

import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.ops import unary_union
from sklearn.metrics import r2_score


# ==========================================================
# Season definitions
# ==========================================================
SEASON_DEF = {
    "DJF": [12, 1, 2],
    "MAM": [3, 4, 5],
    "JJA": [6, 7, 8],
    "SON": [9, 10, 11],
}

SEASON_LABELS = {
    "DJF": "Winter (Dec–Feb)",
    "MAM": "Spring (Mar–May)",
    "JJA": "Summer (Jun–Aug)",
    "SON": "Autumn (Sep–Nov)",
}

REGION_ORDER  = ["north", "north_east", "east", "south"]
REGION_LABELS = {
    "north"     : "North",
    "north_east": "North-East",
    "east"      : "East",
    "south"     : "South",
}


# ==========================================================
# Per-pixel metric computation
# ==========================================================

def compute_pixel_metrics(ref, pred):
    """
    Compute all metrics for one pixel time series.
    Returns dict or None if data is invalid.
    """
    if len(ref) < 10 or np.any(np.isnan(ref)) or np.any(np.isnan(pred)):
        return None
    if np.isclose(np.mean(ref), 0):
        return None

    r = np.corrcoef(pred, ref)[0, 1] if (np.std(ref) > 0 and np.std(pred) > 0) else 0.0

    beta  = np.mean(pred) / np.mean(ref) if np.mean(ref) > 0 else np.nan
    gamma = (
        (np.std(pred) / np.mean(pred)) / (np.std(ref) / np.mean(ref))
        if not np.isclose(np.mean(pred), 0) and not np.isclose(np.std(ref), 0)
        else 1.0
    )
    kge = (
        1 - np.sqrt((r - 1)**2 + (beta - 1)**2 + (gamma - 1)**2)
        if not (np.isnan(beta) or np.isnan(gamma)) else np.nan
    )

    ss_res = np.sum((ref - pred) ** 2)
    ss_tot = np.sum((ref - np.mean(ref)) ** 2)
    nse    = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    bias_pct = (np.mean(pred) - np.mean(ref)) / np.mean(ref)
    rmse     = np.sqrt(np.mean((pred - ref) ** 2))
    mae      = np.mean(np.abs(pred - ref))
    ubrmse   = np.sqrt(np.mean(
        ((pred - np.mean(pred)) - (ref - np.mean(ref))) ** 2
    ))

    def _mae_q(q):
        thr  = np.percentile(ref, q)
        mask = ref >= thr
        return np.mean(np.abs(pred[mask] - ref[mask])) if mask.sum() > 0 else np.nan

    return {
        "R2":          r2_score(ref, pred),
        "KGE":         kge,
        "NSE":         nse,
        "Correlation": r,
        "Bias_pct":    bias_pct,
        "RMSE":        rmse,
        "MAE":         mae,
        "ubRMSE":      ubrmse,
        "MAE_Q95":     _mae_q(95),
        "MAE_Q99":     _mae_q(99),
        "MAE_Q995":    _mae_q(99.5),
    }


def compute_metrics_df(ref_da, pred_da, valid_lats, valid_lons,
                       model_name, season_mask=None):
    """
    Loop over all valid pixels and compute per-pixel metrics.

    Args:
        ref_da      : xr.DataArray stacked on points dim
        pred_da     : xr.DataArray stacked on points dim
        valid_lats  : 1-D array of valid lat coordinates
        valid_lons  : 1-D array of valid lon coordinates
        model_name  : str label for the model
        season_mask : optional boolean array for seasonal filtering

    Returns:
        pd.DataFrame with one row per pixel
    """
    records = []
    for pt_idx in range(len(valid_lats)):
        ref_full  = ref_da.isel(points=pt_idx).values
        pred_full = pred_da.isel(points=pt_idx).values

        ref  = ref_full[season_mask]  if season_mask is not None else ref_full
        pred = pred_full[season_mask] if season_mask is not None else pred_full

        m = compute_pixel_metrics(ref, pred)
        if m is None:
            continue
        m["Model"] = model_name
        m["lat"]   = valid_lats[pt_idx]
        m["lon"]   = valid_lons[pt_idx]
        records.append(m)

    return pd.DataFrame(records)


def compute_seasonal_metrics_df(ref_da, pred_da, valid_lats, valid_lons, model_name):
    """
    Compute per-pixel metrics for each season separately.

    Returns:
        pd.DataFrame with a 'Season' column added
    """
    try:
        months = pd.DatetimeIndex(ref_da.time.values).month.values
    except Exception:
        months = np.array([t.month for t in ref_da.time.values])

    parts = []
    for season, month_list in SEASON_DEF.items():
        mask  = np.isin(months, month_list)
        df_s  = compute_metrics_df(ref_da, pred_da, valid_lats, valid_lons,
                                   model_name, season_mask=mask)
        df_s["Season"] = season
        parts.append(df_s)
    return pd.concat(parts, ignore_index=True)


# ==========================================================
# Region labelling
# ==========================================================

def add_region_labels(df, gdfs_regions):
    """
    Add a 'region' string column to df based on lat/lon.

    Args:
        df          : DataFrame with 'lat' and 'lon' columns
        gdfs_regions: dict {region_name: GeoDataFrame}

    Returns:
        df with 'region' column added (rows without region dropped)
    """
    id_map = {v: i for i, v in enumerate(REGION_ORDER)}

    if df.empty or "lat" not in df.columns or "lon" not in df.columns:
        return df.copy()

    coords = pd.DataFrame({"lat": df["lat"].values, "lon": df["lon"].values})
    gdf    = gpd.GeoDataFrame(coords,
                               geometry=gpd.points_from_xy(coords.lon, coords.lat))
    gdf["region"] = None

    for rname, gdf_r in gdfs_regions.items():
        region_geom = unary_union(gdf_r.geometry)
        mask = gdf.geometry.within(region_geom)
        gdf.loc[mask, "region"] = rname

    df = df.copy()
    df["region"] = gdf["region"].values
    return df.dropna(subset=["region"])
