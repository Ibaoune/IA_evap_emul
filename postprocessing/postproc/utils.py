"""
src/utils.py
============
Shared helpers: load NC predictions, build valid Morocco points,
stack DataArrays, build region point indices.
"""

import os
import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import Point


# ==========================================================
# Load predictions NC
# ==========================================================

def load_predictions(nc_path, pred_var="evap_pred", obs_var="evap_obs"):
    """
    Load a NetCDF file saved by train.py / postprocessing.py.

    Returns:
        pred_da : xr.DataArray  (time, lat, lon)
        obs_da  : xr.DataArray  (time, lat, lon)
    """
    ds      = xr.open_dataset(nc_path)
    pred_da = ds[pred_var]
    obs_da  = ds[obs_var]
    return pred_da, obs_da


# ==========================================================
# Valid Morocco points
# ==========================================================

def get_valid_morocco_points(da, shp_path):
    """
    Return valid (lat, lon) arrays — points inside Morocco.

    Args:
        da       : xr.DataArray with lat/lon dims
        shp_path : path to Morocco shapefile directory

    Returns:
        valid_lats, valid_lons : 1-D np.ndarray
    """
    shp_files = [f for f in os.listdir(shp_path) if f.endswith(".shp")]
    gdf       = gpd.read_file(os.path.join(shp_path, shp_files[0]))
    morocco   = unary_union(gdf.geometry)

    lat_vals = da.lat.values
    lon_vals = da.lon.values
    lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)
    coords = pd.DataFrame({"lat": lat2d.ravel(), "lon": lon2d.ravel()})
    gdf_pts = gpd.GeoDataFrame(
        coords, geometry=gpd.points_from_xy(coords.lon, coords.lat)
    )
    valid = coords[gdf_pts.geometry.within(morocco)].reset_index(drop=True)
    return valid["lat"].values, valid["lon"].values


# ==========================================================
# Stack DataArray on valid points
# ==========================================================

def stack_on_points(da, valid_lats, valid_lons):
    """
    Stack a (time, lat, lon) DataArray on the valid Morocco points.

    Returns:
        xr.DataArray with dim 'points' (MultiIndex of lat, lon)
    """
    stacked = da.stack(points=("lat", "lon"))
    midx    = pd.MultiIndex.from_arrays([valid_lats, valid_lons],
                                         names=["lat", "lon"])
    return stacked.sel(points=midx)


# ==========================================================
# Region point indices
# ==========================================================

def get_region_point_indices(valid_lats, valid_lons, gdfs_regions):
    """
    For each region, return the indices into (valid_lats, valid_lons)
    that fall inside that region.

    Returns:
        dict {region_name: np.ndarray of indices}
    """
    coords  = pd.DataFrame({"lat": valid_lats, "lon": valid_lons})
    gdf_pts = gpd.GeoDataFrame(
        coords, geometry=gpd.points_from_xy(coords.lon, coords.lat)
    )
    result = {}
    for rname, gdf_r in gdfs_regions.items():
        region_geom = unary_union(gdf_r.geometry)
        result[rname] = np.where(gdf_pts.geometry.within(region_geom))[0]
    return result


# ==========================================================
# Build 2-D grid from point data  (for spatial maps)
# ==========================================================

def points_to_grid(valid_lats, valid_lons, values):
    """
    Convert flat point arrays to a 2-D (lat, lon) grid.

    Returns:
        grid     : np.ndarray (nlat, nlon)
        lat_u    : sorted unique lats
        lon_u    : sorted unique lons
    """
    lat_u = np.array(sorted(set(valid_lats)))
    lon_u = np.array(sorted(set(valid_lons)))
    grid  = np.full((len(lat_u), len(lon_u)), np.nan)
    for la, lo, v in zip(valid_lats, valid_lons, values):
        li  = np.searchsorted(lat_u, la)
        loi = np.searchsorted(lon_u, lo)
        if li < len(lat_u) and loi < len(lon_u):
            grid[li, loi] = v
    return grid, lat_u, lon_u


# ==========================================================
# Smart colorbar bounds  (from notebook)
# ==========================================================

def compute_smart_bounds(vals, n_bins=10):
    """Robust colorbar bounds based on IQR."""
    vals = pd.Series(vals).dropna()
    q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
    iqr   = q3 - q1
    vmin  = vals[vals >= q1 - 1.5 * iqr].min()
    vmax  = vals[vals <= q3 + 1.5 * iqr].max()

    symmetric = (vmin < 0) and (vmax > 0)
    if symmetric:
        mx   = max(abs(vmin), abs(vmax))
        vmin, vmax = -mx, mx

    raw_step  = (vmax - vmin) / n_bins
    magnitude = 10 ** np.floor(np.log10(raw_step + 1e-12))
    nice_step = next(
        (s for s in [1, 2, 3, 4, 5, 10] if s >= raw_step / magnitude - 1e-9), 10
    )
    step      = magnitude * nice_step
    vmin_snap = np.floor(vmin / step) * step
    vmax_snap = vmin_snap + n_bins * step

    if symmetric:
        mx        = max(abs(vmin_snap), abs(vmax_snap))
        vmax_snap = np.ceil(mx / step) * step
        vmin_snap = -vmax_snap
        vmax_snap = vmin_snap + n_bins * step

    decimals = min(2, max(0, -int(np.floor(np.log10(step + 1e-12)))))
    return [round(vmin_snap + i * step, decimals) for i in range(n_bins + 1)]
