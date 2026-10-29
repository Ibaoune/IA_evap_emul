"""
src/utils.py
============
Shared helpers: build valid Morocco points,
stack DataArrays, build region point indices,
colorbar bounds and common map/colorbar drawing.
"""

import os
import numpy as np
import matplotlib as mpl
import pandas as pd
import xarray as xr
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import Point


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

def compute_smart_bounds(vals, n_bins=10, quantiles=None, symmetric="auto"):
    """
    Robust colorbar bounds snapped to n_bins "nice" steps.

    Args:
        quantiles : None to span the IQR whiskers (Q1 - 1.5 IQR, Q3 + 1.5 IQR),
                    or [low, high] percentiles, e.g. [2, 98]
        symmetric : "auto" centers on 0 when the range spans 0; True/False force it
    """
    vals = pd.Series(vals).replace([np.inf, -np.inf], np.nan).dropna()
    if len(vals) == 0:
        return [0.0, 1.0]
    if quantiles is None:
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr   = q3 - q1
        vmin  = vals[vals >= q1 - 1.5 * iqr].min()
        vmax  = vals[vals <= q3 + 1.5 * iqr].max()
    else:
        vmin, vmax = np.percentile(vals, quantiles)

    if symmetric == "auto":
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

    decimals = max(0, -int(np.floor(np.log10(step + 1e-12))))
    
    bounds = [round(vmin_snap + i * step, decimals) for i in range(n_bins + 1)]
    # Ensure no duplicates due to rounding
    if len(set(bounds)) < len(bounds):
        bounds = [vmin_snap + i * step for i in range(n_bins + 1)]
    return bounds


# ==========================================================
# Drawing helpers (style settings come from postproc.plot_config)
# ==========================================================

def df_to_grid(df, col, lat_vals, lon_vals):
    """Grid (lat, lon) of column col of a per-pixel DataFrame with lat/lon columns."""
    grid = np.full((len(lat_vals), len(lon_vals)), np.nan)
    for _, row in df.iterrows():
        li = np.searchsorted(lat_vals, row["lat"])
        lo = np.searchsorted(lon_vals, row["lon"])
        if li < len(lat_vals) and lo < len(lon_vals):
            grid[li, lo] = row[col]
    return grid


def months_of(da):
    """Month (1-12) of each time step of a DataArray (numpy or cftime dates)."""
    try:
        return pd.DatetimeIndex(da.time.values).month.values
    except Exception:
        return np.array([t.month for t in da.time.values])


def boxplots_by_group(axes, df_grouped, group_col, groups, group_labels, colors,
                       metrics, labels, st):
    """One panel per metric, one colored box per group (season or region)."""
    for ax, col, lbl in zip(axes, metrics, labels):
        data_list = [df_grouped[df_grouped[group_col] == g][col].dropna().values
                     for g in groups]
        box_colors = [colors[g] for g in groups]
        bp = ax.boxplot(data_list, patch_artist=True, showfliers=st.show_fliers,
                        medianprops=dict(color="black", lw=2))
        for patch, color in zip(bp["boxes"], box_colors):
            patch.set_facecolor(color); patch.set_alpha(st.box_alpha)
        for i, (d, c) in enumerate(zip(data_list, box_colors)):
            if len(d):
                med = np.median(d)
                ax.text(i + 1, med, f"{med:.2f}", ha="center", va="bottom",
                        fontsize=st.annotation_size, fontweight="bold", color=c)
        ax.set_xticks(range(1, len(groups) + 1))
        ax.set_xticklabels([group_labels[g] for g in groups],
                           fontsize=st.tick_label_size, rotation=st.tick_rotation, ha="right")
        ax.set_title(lbl, fontsize=st.title_size)
        ax.grid(axis="y", alpha=0.3)
        ax.spines[["top", "right"]].set_visible(False)


def draw_map_frame(ax, gdf_morocco, st):
    """Morocco outline, map extent and lat/lon gridlines of one map panel."""
    minx, miny, maxx, maxy = gdf_morocco.total_bounds
    m = st.map_margin
    gdf_morocco.boundary.plot(ax=ax, edgecolor=st.border_color, linewidth=st.border_width)
    ax.set_extent([minx - m, maxx + m, miny - m, maxy + m])
    if st.gridlines:
        gl = ax.gridlines(draw_labels=True, linewidth=0.4,
                          color="gray", alpha=0.4, linestyle="--")
        gl.top_labels = False; gl.right_labels = False
        gl.xlabel_style = {"size": st.grid_label_size}
        gl.ylabel_style = {"size": st.grid_label_size}


def add_colorbar(cax, cb, st, orientation="vertical"):
    """Draw the colorbar cb (from PlotConfig.colorbar) in cax."""
    fmt = None
    if cb.tick_format:
        fmt = mpl.ticker.FuncFormatter(lambda x, _: cb.tick_format.format(x))
    bar = mpl.colorbar.ColorbarBase(
        cax, cmap=cb.cmap, norm=cb.norm, ticks=cb.ticks, format=fmt,
        orientation=orientation, extend=cb.extend
    )
    bar.set_label(cb.label, fontsize=st.colorbar_label_size)
    cax.tick_params(labelsize=st.colorbar_tick_size)
    return bar
