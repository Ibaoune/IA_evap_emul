#!/usr/bin/env python
# coding: utf-8

# # Imports

# In[1]:


# cell 1 : install required dependencies 
print("[cell 1] : install required dependencies ")
!pip install -q cftime rioxarray regionmask

# In[2]:


# cell 2: Imports
print("[cell 2] : Imports")
import os
import time
import warnings
import numpy as np
import pandas as pd
import xarray as xr
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import rioxarray 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from pathlib import Path
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.ops import unary_union
from shapely.geometry import Point
import matplotlib as mpl
import cartopy.crs as ccrs
from matplotlib.colors import BoundaryNorm


warnings.filterwarnings('ignore')
print('All imports OK')

# # Config

# In[3]:


# cell 3: DEfine Configs
print("[cell 3] : Define Configs")

# verbosity function 
VERBOSITY = "high"  # "low" | "medium" | "high"
_LEVELS = {"low": 0, "medium": 1, "high": 2}
def vprint(min_level: str, *args, **kwargs):
    """Print only if current VERBOSITY >= min_level."""
    if _LEVELS.get(VERBOSITY, 0) >= _LEVELS.get(min_level, 0):
        print(*args, **kwargs)

# paths 
data_LMDZ35_path  = "/kaggle/input/datasets/hamzakhoti/et-downscaling-morocco/LMDZOR_35/LMDZOR_35"
data_ERA5_path = "/kaggle/input/datasets/hamzakhoti/et-downscaling-morocco/ERA5/ERA5"
data_LMDZ250_path = "/kaggle/input/datasets/hamzakhoti/et-downscaling-morocco/LMDZOR_250/LMDZOR_250"
path_shp          = "/kaggle/input/datasets/hamzakhoti/et-downscaling-morocco/morocco_complet_shapefiles/morocco_complet_shapefiles"
altitude_path = "/kaggle/input/datasets/hamzakhoti/altitude"
path_shp_regions = "/kaggle/input/datasets/hamzakhoti/et-downscaling-morocco/shapefiles_by_region/shapefiles_by_region"
Soil_moisture_path = "/kaggle/input/datasets/hamzakhoti/soil-moisture/soil moisture"
soiltile_path = "/kaggle/input/datasets/hamzakhoti/soiltile"


TRAIN_YEARS = list(range(1979, 2006)) #27y
VAL_YEARS   = list(range(2006, 2009)) #3y
TEST_YEARS  = list(range(2009, 2015)) #6y

msg = (
    f"Train period     : {TRAIN_YEARS}\n"
    f"Validation period: {VAL_YEARS}\n"
    f"Test period      : {TEST_YEARS}\n"
    f"{'='*50}"
)

vprint("low", msg)

BATCH_SIZE  = 128
EPOCHS      = 300
LR          = 1e-4
Model_name = 'RCAN'
is_commit = False
is_early_stopping = True
is_lr_scheduler = True
SCHEDULER_PATIENCE = 15
ES_PATIENCE = 50 # early stopping patience
loss_name = 'BernoulliGamma' # options 'Huber_loss', 'l1'
WEIGHT_DECAY = 5e-3

path_out    = os.path.join("results", Model_name)
path_models = os.path.join("models", Model_name)

os.makedirs(path_out, exist_ok=True)
os.makedirs(path_models, exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

msg = (
    f"Training config:\n"
    f"Model          : {Model_name}\n"
    f"Batch size     : {BATCH_SIZE}\n"
    f"Epochs         : {EPOCHS}\n"
    f"Learning rate  : {LR}\n"
    f"Loss           : {loss_name}\n"
    f"Early stopping : {is_early_stopping} (patience={ES_PATIENCE})\n"
    f"LR scheduler   : {is_lr_scheduler} (patience={SCHEDULER_PATIENCE})\n"
    f"Commit mode    : {is_commit}\n"
    f"WEIGHT_DECAY   : {WEIGHT_DECAY}\n"
    f"Device    : {device}\n"
    f"{'='*50}"
)

vprint("low", msg)

# # Load & Process

# In[4]:


# cell 4 : Utils Functions to load and mask
print("[cell 4] : Utils Functions to load and mask")
def mask_dataset(ds, name):
    if name in ['mrso', 'mrsos']:
        return ds.sel(lon=slice(-18, 0), lat=slice(20, 36))
    else:
        lon_range = slice(-18, 0)
        lat_range = slice(20, 36)
        var_lon = None
        var_lat = None
        variable_names = list(ds.variables)
        for var_name in variable_names:
            if var_name.startswith('lon') or var_name.startswith('longitude'):
                var_lon = var_name
            elif var_name.startswith('lat') or var_name.startswith('latitude'):
                var_lat = var_name
        mask = (ds[var_lon] >= lon_range.start) & (ds[var_lon] <= lon_range.stop) & \
            (ds[var_lat] >= lat_range.start) & (ds[var_lat] <= lat_range.stop)
        # Apply the mask to the dataset
        ds_masked=ds.where(mask, drop=True)
        return ds_masked

def load_lmdz(filename, varname, data_name):
    """Load + mask a single LMDZ variable — same as main.ipynb."""
    path_in = data_LMDZ35_path if data_name == "LMDZOR_35" else data_LMDZ250_path
    var = mask_dataset(xr.open_dataset(os.path.join(path_in, filename)), varname)[varname]
    var = var.sortby("lat")
    return var
print('Utilities OK')

# In[5]:


# cell 5 : Load LMDZ250 variables
print("[cell 5] : LMDZ250 variables")

# Open radiation / q2m file 
path_rad = data_LMDZ250_path
ds_rad = xr.open_dataset(os.path.join(path_rad, "LMDZ_rad_q2m_surface.nc"))
ds_rad = mask_dataset(ds_rad, 'q2m_radiation').sortby("lat")

# Extract variables
q2m     = ds_rad["q2m"]
SWupSFC = ds_rad["SWupSFC"]
SWdnSFC = ds_rad["SWdnSFC"]
LWupSFC = ds_rad["LWupSFC"]
LWdnSFC = ds_rad["LWdnSFC"]

# Other variables
pres_ds = load_lmdz("/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/models/evap_staging/pres-hist2.nc",  "pres",  "LMDZOR_250")
t2m_ds  = load_lmdz("/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/models/evap_staging/t2m-hist1.nc",   "t2m",   "LMDZOR_250")
u10m_ds = load_lmdz("/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/models/evap_staging/u10m-hist1.nc",  "u10m",  "LMDZOR_250")
v10m_ds = load_lmdz("/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/models/evap_staging/v10m-hist1.nc",  "v10m",  "LMDZOR_250")

# Evap 
evap_lmdz250 = mask_dataset(
    xr.open_dataset(os.path.join(data_LMDZ250_path, "evap-hist1.nc")), "evap"
)["evap"] * 86400
evap_lmdz250 = evap_lmdz250.sortby("lat").clip(min=0) 

# Derived radiation 
swnet_ds = (SWdnSFC - SWupSFC).rename("SWnet")
lwnet_ds = (LWdnSFC - LWupSFC).rename("LWnet")


print('coarse LMDZ250 variables loaded')

# In[6]:


from scipy.ndimage import distance_transform_edt

def fill_nan_nearest_xr(da):
    """Fill NaNs spatially using nearest valid neighbour, per time step."""
    values = da.values.copy()  # shape: (time, lat, lon)
    for t in range(values.shape[0]):
        sl = values[t]
        nan_mask = np.isnan(sl)
        if not nan_mask.any():
            continue
        _, indices = distance_transform_edt(nan_mask, return_indices=True)
        sl[nan_mask] = sl[indices[0][nan_mask], indices[1][nan_mask]]
        values[t] = sl
    return da.copy(data=values)


# load and process soil moisture files 
mrso  = xr.open_dataset(os.path.join(Soil_moisture_path, "mrso_day_IPSL-CM6A-LR_amip_r1i1p1f1_gr_19790101-20141231.nc"))["mrso"]
mrsos = xr.open_dataset(os.path.join(Soil_moisture_path, "mrsos_day_IPSL-CM6A-LR_amip_r1i1p1f1_gr_19790101-20141231.nc"))["mrsos"]

mrso  = mrso.assign_coords(lon=(((mrso.lon + 180) % 360) - 180)).sortby("lon")
mrsos = mrsos.assign_coords(lon=(((mrsos.lon + 180) % 360) - 180)).sortby("lon")

mrso  = mask_dataset(mrso,  "mrso")
mrsos = mask_dataset(mrsos, "mrsos")

# Fill NaNs here, while still xarray, before flattening
mrso  = fill_nan_nearest_xr(mrso)
mrsos = fill_nan_nearest_xr(mrsos)

mrso  = mrso.rename({"time": "time_counter"})
mrsos = mrsos.rename({"time": "time_counter"})

# Verify
print("mrso  NaNs:", np.isnan(mrso.values).sum())
print("mrsos NaNs:", np.isnan(mrsos.values).sum())

# In[7]:


# cell 6 : load LMDZ_35 variables
print("[cell 6] : load LMDZ_35 variables")

# Load LMDZ35 target (high-resolution ET)
evap_lmdz35 = mask_dataset(
    xr.open_dataset(os.path.join(data_LMDZ35_path, "evap_1979_2014_MA.nc")), "evap"
)["evap"] * 86400

evap_lmdz35 = evap_lmdz35.sortby("lat")

# Remove negatives
evap_lmdz35 = evap_lmdz35.clip(min=0)

# lat/lon are stored here for later use (Cell 15, 23)
lat_values = evap_lmdz35["lat"].values
lon_values = evap_lmdz35["lon"].values

# Load other LMDZ35 variables
albedo_ds   = load_lmdz("/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/models/evap_staging/albedo_glob-hist1.nc", "albedo_glob", "LMDZOR_35").fillna(0)
mean_lai_ds = load_lmdz("/home/mohammad.elaabaribao/lustre/climat-um6p-st-iwri-7ksifkvwkuy/users/mohammad.elaabaribao/data/models/evap_staging/LAImean-hist1.nc",     "LAImean",     "LMDZOR_35").fillna(0)

# Reference grid: use coarse LMDZ250 grid (t2m_ds as an exemple) as the target for regridding
# This ensures all coarse-resolution inputs share the same spatial grid
ref_coarse = t2m_ds

albedo_ds   = albedo_ds.reindex_like(ref_coarse,   method="nearest").fillna(0)
mean_lai_ds = mean_lai_ds.reindex_like(ref_coarse, method="nearest").fillna(0)

print('coarse LMDZ35 variables loaded')

# In[8]:


# cell 7: free up memory 
print("[cell 7] : free up memory")
if is_commit:
    # Free raw radiation components — only derived swnet/lwnet needed
    del SWupSFC, SWdnSFC, LWupSFC, LWdnSFC
    gc.collect()
    print("Raw radiation arrays freed")

# In[9]:


# cell 8: Utils functions to build features and inspect
print("[cell 8] : Utils functions to build features and inspect")
def build_features(datasets, target_name="evap_era5"):
    predictors = []

    for name, da in datasets.items():
        if name == target_name:
            continue  # skip target variable
        da = da.transpose("time_counter", "lat", "lon")
        da = da.expand_dims(feature=[name])
        predictors.append(da)

    X = xr.concat(predictors, dim="feature")

    # enforce correct order
    X = X.transpose("time_counter", "lat", "lon", "feature")
    return X


def inspect(datasets):
    ''''
    inspect the datasets by printing out key information about each one, including : variable name, shape,
      dimensions, time steps, grid info (if lat/lon), NaN count and percentage, and value range/mean.
    '''
    print("\nDATASET INSPECTION\n" + "="*50)

    for name, da in datasets.items():
        print(f"\n{name}")
        print(f"Var: {da.name}")
        print(f"Shape: {da.shape} | Dims: {da.dims}")

        t = "time_counter" if "time_counter" in da.dims else "time" if "time" in da.dims else None
        print(f"Time steps: {len(da[t]) if t else '❌'}")

        if "lat" in da.dims and "lon" in da.dims:
            lat, lon = da.lat.values, da.lon.values
            dlat, dlon = np.diff(lat), np.diff(lon)

            print(f"Grid: {len(lat)}x{len(lon)}")
            print(f"Lat: {lat.min():.2f}→{lat.max():.2f} | inc:{np.all(dlat>0)} dec:{np.all(dlat<0)}")
            print(f"Lon: {lon.min():.2f}→{lon.max():.2f}")
            print(f"Res: {np.abs(dlat).mean():.3f}, {np.abs(dlon).mean():.3f}")

        v = da.values
        nan = np.isnan(v)
        print(f"NaN: {nan.sum()} ({nan.mean()*100:.2f}%)")

        v = v[~nan]
        print(f"Range: [{v.min():.3f}, {v.max():.3f}] | Mean: {v.mean():.3f}" if v.size else "All NaN")

        print("-"*50)


predictor_vars = {
    "t2m":          t2m_ds,
    "q2m":          q2m,
    "swnet":        swnet_ds,
    "lwnet":        lwnet_ds,
    "pres":         pres_ds,
    "u10m":         u10m_ds,
    "v10m":         v10m_ds,
    "evap_lmdz250": evap_lmdz250,
    "albedo":       albedo_ds,
    "lai":          mean_lai_ds,
    "mrso": mrso,
    "mrsos": mrsos
}


# Sanity check
print(f"{len(predictor_vars)} predictors loaded")
for name, da in predictor_vars.items():
    print(f"  {name:15s} → shape: {da.shape}")

# In[10]:


# cell 9 : free up memory
print("[cell 9] : free up memory")
if is_commit:
    # Free all individual xarray variables, X contains everything
    del altitude_da, altitude_coarse, altitude_2d, altitude_raw
    del q2m, pres_ds, t2m_ds, u10m_ds, v10m_ds, evap_lmdz250
    del albedo_ds, mean_lai_ds, swnet_ds, lwnet_ds
    gc.collect()
    print("All xarray predictors freed")

# In[11]:


# cell 10 : load Altitude
print("[cell 10] : load Altitude")

# load and process Altitude 
altitude_raw = rioxarray.open_rasterio(os.path.join(altitude_path, "elevation.tif"))
altitude_2d  = altitude_raw.squeeze("band", drop=True).rename({"x": "lon", "y": "lat"})

altitude_coarse = altitude_2d.interp(
    lat=t2m_ds.lat.values,
    lon=t2m_ds.lon.values,
    method="linear"  
).fillna(0)

T_all = len(t2m_ds.time_counter)
altitude_da = xr.DataArray(
    np.broadcast_to(
        altitude_coarse.values[np.newaxis],
        (T_all, *altitude_coarse.shape)
    ).copy().astype(np.float32),
    dims=["time_counter", "lat", "lon"],
    coords={
        "time_counter": t2m_ds.time_counter,
        "lat":          t2m_ds.lat,
        "lon":          t2m_ds.lon,
    },
    name="altitude"
)

predictor_vars["altitude"] = altitude_da

# In[12]:


# cell 10c : Soiltile channels (bare, short, tall vegetation)
print("[cell 10c] : Soiltile channels")

ds_soil = xr.open_dataset(os.path.join(soiltile_path, "soiltile.nc"))
ds_soil = ds_soil.assign_coords(
    lon=(((ds_soil.lon + 180) % 360) - 180)
).sortby("lon")

soiltile = ds_soil["soiltile"]  # (soiltyp, lat, lon)

for soiltyp_val, name in [(1.0, "soiltile_bare"), (2.0, "soiltile_short"), (3.0, "soiltile_tall")]:
    tile_2d = soiltile.sel(soiltyp=soiltyp_val).drop_vars("soiltyp")  # (lat, lon)

    tile_coarse = tile_2d.interp(
        lat=t2m_ds.lat.values,
        lon=t2m_ds.lon.values,
        method="linear"
    )

    predictor_vars[name] = xr.DataArray(
        np.broadcast_to(
            tile_coarse.values[np.newaxis],
            (T_all, *tile_coarse.shape)
        ).copy().astype(np.float32),
        dims=["time_counter", "lat", "lon"],
        coords={
            "time_counter": t2m_ds.time_counter,
            "lat":          t2m_ds.lat,
            "lon":          t2m_ds.lon,
        },
        name=name
    )
    predictor_vars[name]  = fill_nan_nearest_xr(predictor_vars[name])

print("soiltile_bare / soiltile_short / soiltile_tall channels added")

# In[13]:


vprint("medium", "\nRunning dataset inspection...")
dataset = {val:key for val, key in predictor_vars.items()}
dataset['evap_lmdz35'] = evap_lmdz35
inspect(dataset)

# In[14]:


# Cell 11: Stack predictors into (T, C, lat_c, lon_c) tensor
print("[cell 11] : Stack predictors into (T, C, lat_c, lon_c) tensor")
CHANNEL_NAMES = list(predictor_vars.keys())
N_CHANNELS    = len(CHANNEL_NAMES)

X_all = np.stack(
    [da.values for da in predictor_vars.values()], axis=1
).astype(np.float32)                              # (T, 10, lat_c, lon_c)

Y_all = evap_lmdz35.values.astype(np.float32)    # (T, lat_f, lon_f)

_, N_C, LAT_C, LON_C = X_all.shape
_, LAT_F, LON_F      = Y_all.shape

# ADD these lines at the end of Cell 11 (they are currently missing):
INPUT_SHAPE  = (N_C, LAT_C, LON_C)
OUTPUT_SHAPE = (1, LAT_F, LON_F)

vprint("low", f"X_all        : {X_all.shape}")
vprint("low", f"Y_all        : {Y_all.shape}")
vprint("low", f"Input shape  : {INPUT_SHAPE}")
vprint("low", f"Output shape : {OUTPUT_SHAPE}")
vprint("low", f"Channels     : {CHANNEL_NAMES}")

# In[15]:


# Cell 11b: Morocco spatial mask 
print("[cell 11b] : Morocco spatial mask")

morocco_shp   = gpd.read_file(path_shp)
morocco_union = unary_union(morocco_shp.geometry)

def build_morocco_mask(lats, lons, geometry):
    """Bool array (lat, lon) — True = inside Morocco."""
    mask = np.zeros((len(lats), len(lons)), dtype=bool)
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            mask[i, j] = geometry.contains(Point(lon, lat))
    return mask

lat_c_vals = t2m_ds["lat"].values
lon_c_vals = t2m_ds["lon"].values
morocco_mask_fine   = build_morocco_mask(lat_values, lon_values, morocco_union)  # (LAT_F, LON_F)

# Torch mask — used inside the loss
morocco_mask_tensor = torch.BoolTensor(morocco_mask_fine).to(device)

# NOTE: nothing is zeroed here — masking happens per-channel in Cell 14
vprint("low", f"Fine   mask: {morocco_mask_fine.sum()}   / {morocco_mask_fine.size} px in Morocco")

# In[16]:


# Cell 11c: ocean mask from regionmask + land_mask channel
print("[cell 11c] : Land/ocean mask channel")

import regionmask

# ── build global ocean mask on coarse grid ────────────────────────────
land_region       = regionmask.defined_regions.natural_earth_v5_0_0.land_110
land_mask_2d_xr   = land_region.mask(lon_c_vals, lat_c_vals)  # 0=land, NaN=ocean
ocean_mask_coarse = np.isnan(land_mask_2d_xr.values)          # (LAT_C, LON_C) True=ocean
land_mask_2d      = ~ocean_mask_coarse                         # True=land (all countries)

vprint("low", f"Ocean pixels: {ocean_mask_coarse.sum()} / {ocean_mask_coarse.size}")
vprint("low", f"Land  pixels: {land_mask_2d.sum()}")

# ── append land_mask as last channel ─────────────────────────────────
land_mask_arr = np.broadcast_to(
    land_mask_2d[np.newaxis].astype(np.float32),
    (X_all.shape[0], *land_mask_2d.shape)
).copy()

X_all = np.concatenate(
    [X_all, land_mask_arr[:, np.newaxis, :, :]],
    axis=1
)

CHANNEL_NAMES.append("land_mask")
N_CHANNELS   = len(CHANNEL_NAMES)
INPUT_SHAPE  = (N_CHANNELS, LAT_C, LON_C)
OUTPUT_SHAPE = (1, LAT_F, LON_F)

vprint("low", f"X_all       : {X_all.shape}")
vprint("low", f"Input shape : {INPUT_SHAPE}")
vprint("low", f"Channels    : {CHANNEL_NAMES}")

# In[17]:


# cell 10c-viz : Channel visualization utility + RAW call
print("[cell 10c-viz] : Visualize all predictor channels — RAW")

morocco_shp_viz   = gpd.read_file(path_shp)
morocco_union_viz = unary_union(morocco_shp_viz.geometry)
morocco_gdf_viz   = gpd.GeoDataFrame(geometry=[morocco_union_viz], crs="EPSG:4326")
minx, miny, maxx, maxy = morocco_gdf_viz.total_bounds

STD_CHANNELS = {"month_sin", "month_cos"}

def plot_channels(source, lat_1d, lon_1d,
                  channel_names=None,
                  title="Predictor channels",
                  filename="channels.png"):
    """
    source : dict {name: xarray.DataArray}   — use predictor_vars / dataset
           OR np.ndarray (T, C, LAT, LON)    — use X_all after zeroing/norm
    channel_names : required when source is ndarray
    """
    # ── unpack source into list of (name, 2-D data) ──────────────────
    panels = []

    if isinstance(source, np.ndarray):
        # numpy (T, C, LAT, LON) — average over time axis
        for i, name in enumerate(channel_names):
            arr = source[:, i]               # (T, LAT, LON)
            if name in STD_CHANNELS:
                data2d = arr.std(axis=0)
                agg = "std"
            else:
                data2d = np.nanmean(arr, axis=0)
                agg = "mean"
            panels.append((name, data2d, agg, lat_1d, lon_1d))

    else:
        # dict of xarray DataArrays
        for name, da in source.items():
            lats = da.lat.values
            lons = da.lon.values
            if da.ndim == 3:
                if name in STD_CHANNELS:
                    data2d = da.values.std(axis=0)
                    agg = "std"
                else:
                    data2d = np.nanmean(da.values, axis=0)
                    agg = "mean"
            else:
                data2d = da.values
                agg = "static"
            panels.append((name, data2d, agg, lats, lons))

    # ── layout ───────────────────────────────────────────────────────
    N     = len(panels)
    NCOLS = 4
    NROWS = int(np.ceil(N / NCOLS))

    fig, axes = plt.subplots(
        NROWS, NCOLS,
        figsize=(6 * NCOLS, 5 * NROWS),
        subplot_kw={"projection": ccrs.PlateCarree()},
        dpi=110
    )
    axes_flat = axes.flatten()

    for ax, (name, data2d, agg, lats, lons) in zip(axes_flat, panels):
        lon2d, lat2d = np.meshgrid(lons, lats)
        valid = data2d[~np.isnan(data2d)]

        if valid.size == 0:
            ax.set_title(f"{name}\n(all NaN)", fontsize=10)
            ax.set_extent([minx-.5, maxx+.5, miny-.5, maxy+.5])
            continue

        vmin, vmax = np.nanpercentile(valid, [2, 98])
        im = ax.pcolormesh(lon2d, lat2d, data2d,
                           cmap="viridis", vmin=vmin, vmax=vmax,
                           shading="auto", transform=ccrs.PlateCarree())
        morocco_gdf_viz.boundary.plot(ax=ax, edgecolor="black",
                                      linewidth=1.2, transform=ccrs.PlateCarree())
        ax.set_extent([minx-.5, maxx+.5, miny-.5, maxy+.5])
        ax.set_title(f"{name} ({agg})", fontsize=11, fontweight="bold")

        gl = ax.gridlines(draw_labels=True, linewidth=0.4,
                          color="gray", alpha=0.4, linestyle="--")
        gl.top_labels = gl.right_labels = False
        gl.xlabel_style = gl.ylabel_style = {"size": 7}

        plt.colorbar(im, ax=ax, orientation="horizontal", pad=0.04, shrink=0.85)

    for ax in axes_flat[N:]:
        ax.set_visible(False)

    plt.suptitle(title, fontsize=15, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(path_out, filename), dpi=150, bbox_inches="tight")
    plt.show()

# In[18]:


# Cell 11d: Zero ocean pixels for land-only channels
print("[cell 11d] : Ocean zeroing for land-only channels")

# ocean_mask_coarse built in Cell 11c from mrso NaNs
# True = ocean only — Algeria, Mauritania etc. are land → untouched
OCEAN_ZERO_CHANNELS = [
    "mrso", "mrsos",
    "soiltile_bare", "soiltile_short", "soiltile_tall",
    "albedo", "lai",
]

for name in OCEAN_ZERO_CHANNELS:
    if name not in CHANNEL_NAMES:
        print(f"  [skip] {name} not in CHANNEL_NAMES")
        continue
    c = CHANNEL_NAMES.index(name)
    X_all[:, c, ocean_mask_coarse] = 0.0    # ← ocean_mask not ~morocco_mask

vprint("low", f"Ocean pixels zeroed for: {OCEAN_ZERO_CHANNELS}")

plot_channels(
    source        = X_all,
    lat_1d        = lat_c_vals,
    lon_1d        = lon_c_vals,
    channel_names = CHANNEL_NAMES,
    title         = "All channels — after ocean zeroing (pre-normalization)",
    filename      = "channels_zeroed.png"
)

# In[19]:


# Cell 13: Year-based train / val / test split
print("[cell 13] : Year-based train / val / test split")
times = pd.to_datetime(evap_lmdz35.time_counter.values)
years = times.year

train_mask = np.isin(years, TRAIN_YEARS)
val_mask   = np.isin(years, VAL_YEARS)
test_mask  = np.isin(years, TEST_YEARS)

X_train_np, Y_train_np = X_all[train_mask], Y_all[train_mask]
X_val_np,   Y_val_np   = X_all[val_mask],   Y_all[val_mask]
X_test_np,  Y_test_np  = X_all[test_mask],  Y_all[test_mask]
time_test               = times[test_mask]

vprint("low", f"Train : {X_train_np.shape[0]} samples  ({TRAIN_YEARS[0]}–{TRAIN_YEARS[-1]})")
vprint("low", f"Val   : {X_val_np.shape[0]} samples  ({VAL_YEARS[0]}–{VAL_YEARS[-1]})")
vprint("low", f"Test  : {X_test_np.shape[0]} samples  ({TEST_YEARS[0]}–{TEST_YEARS[-1]})")

# In[20]:


# Cell 14: Per-grid-point normalization (fit on train only)
print("[cell 14] : Per-grid normalization")

T_tr = X_train_np.shape[0]
T_va = X_val_np.shape[0]
T_te = X_test_np.shape[0]

X_train_norm = np.zeros_like(X_train_np)
X_val_norm   = np.zeros_like(X_val_np)
X_test_norm  = np.zeros_like(X_test_np)

EPS = 1e-8

SCALE_CHANNELS = [
    "t2m", "q2m", "swnet", "lwnet",
    "pres", "u10m", "v10m", "evap_lmdz250",
    "altitude", "lat_channel", "lon_channel",
    "albedo", "lai",
    "mrso", "mrsos",
]

scalers_X = {}

for name in SCALE_CHANNELS:
    if name not in CHANNEL_NAMES:
        vprint("medium", f"  [skip] {name} not in CHANNEL_NAMES")
        continue
    c = CHANNEL_NAMES.index(name)

    mean_map = X_train_np[:, c].mean(axis=0)          # (LAT_C, LON_C)
    std_map  = X_train_np[:, c].std(axis=0)           # (LAT_C, LON_C)
    std_map  = np.where(std_map < EPS, 1.0, std_map)  # guard zero-std pixels

    for X_in, X_out in [(X_train_np, X_train_norm),
                        (X_val_np,   X_val_norm),
                        (X_test_np,  X_test_norm)]:
        X_out[:, c] = (X_in[:, c] - mean_map) / std_map

    scalers_X[name] = {"mean": mean_map, "std": std_map}

# ── Pass-through: no normalization ───────────────────────────────────
PASSTHROUGH = ["month_sin", "month_cos", "land_mask", "soiltile_bare", "soiltile_short", "soiltile_tall"]

for name in PASSTHROUGH:
    if name not in CHANNEL_NAMES:
        continue
    c = CHANNEL_NAMES.index(name)
    X_train_norm[:, c] = X_train_np[:, c]
    X_val_norm[:,   c] = X_val_np[:,   c]
    X_test_norm[:,  c] = X_test_np[:,  c]

# ── Convert to PyTorch tensors ────────────────────────────────────────
X_train_t = torch.FloatTensor(X_train_norm).to(device)
Y_train_t = torch.FloatTensor(Y_train_np).unsqueeze(1).to(device)
X_val_t   = torch.FloatTensor(X_val_norm).to(device)
Y_val_t   = torch.FloatTensor(Y_val_np).unsqueeze(1).to(device)
X_test_t  = torch.FloatTensor(X_test_norm)
Y_test_t  = torch.FloatTensor(Y_test_np).unsqueeze(1)

vprint("high", "Per-grid normalization done (fit on train only)")
vprint("high", f"  Normalized  : {SCALE_CHANNELS}")
vprint("high", f"  Pass-through: {PASSTHROUGH}")
vprint("high", f"  X_train_t   : {X_train_t.shape}")
vprint("high", f"  X_val_t     : {X_val_t.shape}")
vprint("high", f"  X_test_t    : {X_test_t.shape}")

# In[21]:


# Cell 15: Define EarlyStopper
print("[cell 15] : Define EarlyStopper")
class EarlyStopper:
    def __init__(self, patience=20, min_delta=0.001):
        self.patience      = patience
        self.min_delta     = min_delta
        self.counter       = 0
        self.min_val_loss  = float('inf')

    def early_stop(self, val_loss):
        if val_loss < self.min_val_loss - self.min_delta:
            self.min_val_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience

vprint("high", "EarlyStopper OK")

# In[22]:


# Cell 16: RCAN
print("[cell 16] : RCAN")
class BernoulliGammaLoss(nn.Module):
    def __init__(self, mask=None):
        """
        mask : bool tensor (LAT_F, LON_F) — True inside Morocco.
               If provided, loss is averaged only over Morocco pixels.
        """
        super(BernoulliGammaLoss, self).__init__()
        # register_buffer keeps the mask on the right device automatically
        if mask is not None:
            self.register_buffer('mask', mask.float())   # (H, W)
        else:
            self.mask = None

    def forward(self, true, pred):
        true = true.squeeze(1)                           # (B, H, W)
        ocurrence       = pred[:, 0, :, :]
        shape_parameter = torch.exp(pred[:, 1, :, :])
        scale_parameter = torch.exp(pred[:, 2, :, :])
        bool_rain = (true > 0).float()
        epsilon = 1e-6

        loss_map = -(                                    # (B, H, W)
            (1 - bool_rain) * torch.log(1 - ocurrence + epsilon) +
            bool_rain * (
                torch.log(ocurrence + epsilon) +
                (shape_parameter - 1) * torch.log(true + epsilon) -
                shape_parameter * torch.log(scale_parameter + epsilon) -
                torch.lgamma(shape_parameter + epsilon) -
                true / (scale_parameter + epsilon)
            )
        )

        if self.mask is not None:
            # Average only over Morocco pixels (mask=1), ignore ocean/outside (mask=0)
            return (loss_map * self.mask).sum() / (self.mask.sum() * loss_map.shape[0])
        
        return loss_map.mean()

class ChannelAttention(nn.Module):
    def __init__(self, n_feat, reduction=16):
        super().__init__()
        self.body = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # global avg pool → (B, C, 1, 1)
            nn.Conv2d(n_feat, n_feat // reduction, 1),  # Conv 1×1, 4
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat // reduction, n_feat, 1),  # Conv 1×1, 64
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.body(x)              # element-wise product

class RCAB(nn.Module):
    def __init__(self, n_feat=64, reduction=16, dropout=0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout), 
            nn.Conv2d(n_feat, n_feat, 3, padding=1),
            ChannelAttention(n_feat, reduction)
        )
    def forward(self, x):
        return x + self.body(x)              # RCAB skip connection


class ResidualGroup(nn.Module):
    def __init__(self, n_feat=64, n_rcab=20, reduction=16):
        super().__init__()
        self.body = nn.Sequential(
            *[RCAB(n_feat, reduction) for _ in range(n_rcab)],
            nn.Conv2d(n_feat, n_feat, 3, padding=1)
        )
    def forward(self, x):
        return x + self.body(x)              # RG skip connection


class RCAN(nn.Module):
    def __init__(self, input_shape, output_shape,
                 n_feat=64, n_groups=10, n_rcab=20, reduction=16):
        """
        input_shape  : (C, lat_coarse, lon_coarse)
        output_shape : (1, lat_fine,   lon_fine)
        """
        super().__init__()
        self.output_shape = output_shape

        # Head
        self.head = nn.Conv2d(input_shape[0], n_feat, 3, padding=1)

        # 10 Residual Groups + long skip connection
        self.body = nn.Sequential(*[
            ResidualGroup(n_feat, n_rcab, reduction)
            for _ in range(n_groups)
        ])

        
        self.body_end = nn.Conv2d(n_feat, n_feat, 3, padding=1)

        # Upscale to fine grid (non-integer scale → bilinear + conv)
        self.upsample = nn.Sequential(
            nn.Upsample(size=(output_shape[1], output_shape[2]),
                        mode='bilinear', align_corners=False),
            nn.Conv2d(n_feat, n_feat, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Tail
        self.tail_occ   = nn.Conv2d(n_feat, 1, 3, padding=1)  # occurrence
        self.tail_shape = nn.Conv2d(n_feat, 1, 3, padding=1)  # log-shape
        self.tail_scale = nn.Conv2d(n_feat, 1, 3, padding=1)  # log-scale

    def forward(self, x):
        head = self.head(x)
        body = self.body_end(self.body(head))
        out  = head + body
        out  = self.upsample(out)
        p1 = torch.sigmoid(self.tail_occ(out))   # occurrence in [0,1]
        p2 = self.tail_shape(out)                # log-shape
        p3 = self.tail_scale(out)                # log-scale
        return torch.cat((p1, p2, p3), dim=1)   # (B, 3, lat_f, lon_f)



model = RCAN(
    input_shape  = INPUT_SHAPE,
    output_shape = OUTPUT_SHAPE,
    n_feat       = 32,
    n_groups     = 5,
    n_rcab       = 6,
    reduction    = 8,
)
model  = model.to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
vprint("high", f"Trainable parameters: {n_params:,}")

# In[23]:


# Cell 17: DataLoaders (train 90% / val 10%, same split as reference) 
print("[cell 17] : DataLoaders (train 90% / val 10%, same split as reference)")
# 
train_loader = DataLoader(
    TensorDataset(X_train_t, Y_train_t),
    batch_size=BATCH_SIZE, shuffle=True
)
val_loader = DataLoader(
    TensorDataset(X_val_t, Y_val_t),
    batch_size=BATCH_SIZE, shuffle=False
)

vprint("high", f"Train batches: {len(train_loader)}  |  Val batches: {len(val_loader)}")

# In[24]:


# cell 18 : Model Config
print("[cell 18] : Model Config")

# Select loss
if loss_name == 'Huber_loss':
    criterion = nn.HuberLoss(delta=1.0)
elif loss_name == 'l1':
    criterion = nn.L1Loss(reduction='mean')
    
elif loss_name == 'BernoulliGamma':
    criterion = BernoulliGammaLoss(mask=morocco_mask_tensor)
    
else:
    # default fallback
    criterion = nn.HuberLoss(delta=1.0)
    loss_name = 'Huber_loss'

    vprint("low", f"[Warning] Unknown loss_name. Using default: {loss_name}")


optimizer     = optim.Adam(model.parameters(), lr=LR, betas=(0.9, 0.999), weight_decay=WEIGHT_DECAY)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode='min', factor=0.5, patience=SCHEDULER_PATIENCE, min_lr=1e-6
)
early_stopper = EarlyStopper(patience=ES_PATIENCE, min_delta=0.005)

# In[25]:


# cell 19 : launche training
print("[cell 19] : launche training")

train_losses, val_losses = [], []
best_val_loss = float('inf')
start = time.time()

printing = (
    f"Training {Model_name} on this config:\n"
    f" -epochs: {EPOCHS}\n"
    f" -learning rate: {LR}\n"
    f" -batch size: {BATCH_SIZE}\n"
    f" -loss: {loss_name}\n"
    f" -Train considering: {CHANNEL_NAMES}\n"
    f" -Train period: {TRAIN_YEARS[0]} - {TRAIN_YEARS[-1]}\n"
    f" -Val period: {VAL_YEARS[0]} - {VAL_YEARS[-1]}\n"
    f" -Test period: {TEST_YEARS[0]} - {TEST_YEARS[-1]}\n"
)
vprint("low", printing)
for epoch in range(1, EPOCHS + 1):

    # ── train ──
    model.train()
    running_train = 0.0
    for xb, yb in train_loader:
        xb, yb = xb.to(device), yb.to(device)
        optimizer.zero_grad()
        pred = model(xb)
        loss = criterion(yb, pred)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # ← STILL MISSING
        optimizer.step()
        running_train += loss.item()
    train_loss = running_train / len(train_loader)
    train_losses.append(train_loss)

    # ── validate ──
    model.eval()
    running_val = 0.0
    with torch.no_grad():
        for xb, yb in val_loader:
            xb, yb = xb.to(device), yb.to(device)
            running_val += criterion(yb, model(xb)).item()
    val_loss = running_val / len(val_loader)
    val_losses.append(val_loss)

    scheduler.step(val_loss) 

    # ── checkpoint ──
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        torch.save(model.state_dict(), "/kaggle/working/best_RCAN.pth")

    #if epoch % 10 == 0:
    print(f"Epoch {epoch:4d} | train={train_loss:.4f} | val={val_loss:.4f} | lr={optimizer.param_groups[0]['lr']:.6f}")

    if early_stopper.early_stop(val_loss):
        print(f"\nEarly stop at epoch {epoch}  |  best val loss={best_val_loss:.4f}")
        break

elapsed = time.time() - start
print(f"\nTraining time: {elapsed:.1f}s  ({elapsed/60:.1f} min)")

# ── reload best weights before inference ──
model.load_state_dict(torch.load("/kaggle/working/best_RCAN.pth"))
model.eval()
vprint("meduim", "Best model reloaded")

# In[26]:


# cell 20 : Loss curves
print("[cell 20] : Loss curves")

fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(train_losses, label=f'Train loss {loss_name}',      lw=1.5)
ax.plot(val_losses,   label=f'Validation loss {loss_name}', lw=1.5)
ax.set_xlabel('Epoch')
ax.set_ylabel(f'{loss_name}')
ax.set_title(f'{Model_name} training — ET downscaling')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(path_out, 'loss_curves.png'), dpi=150)
plt.show()

# In[27]:


# cell 21: Save model weights
print("[cell 21] : Save model weights")

save_path = os.path.join(path_models, 'AE_ET.pth')
torch.save(model.state_dict(), save_path)
print(f"Model saved → {save_path}")
vprint("meduim", f"Model saved to {save_path}")

# In[28]:


print(LAT_F, LON_F)

# In[29]:


# cell 22: Predict on test set
print("[cell 22] : Predict on test set")
model.eval()
with torch.no_grad():
    Y_pred_raw = model(X_test_t.to(device)).cpu()  # (T, 3, LAT_F, LON_F)

# Extract expected value: E[BG] = p * alpha * beta
occurrence = Y_pred_raw[:, 0]               # already sigmoid'd in forward
shape      = torch.exp(Y_pred_raw[:, 1])
scale      = torch.exp(Y_pred_raw[:, 2])
Y_pred = (occurrence * shape * scale).numpy().clip(min=0)  # (T, LAT_F, LON_F)

# Mask ocean
Y_pred[:, ~morocco_mask_fine]    = 0.0
Y_test_np[:, ~morocco_mask_fine] = 0.0

vprint("meduim", f"Predictions shape: {Y_pred.shape}")
vprint("meduim", f"  pred range : [{Y_pred.min():.3f}, {Y_pred.max():.3f}] mm/day")
vprint("meduim", f"  obs  range : [{Y_test_np.min():.3f}, {Y_test_np.max():.3f}] mm/day")

# In[30]:


# cell 23: Quick inspect of model performance on test set
print("[cell 23] : Quick inspect of model performance on test set")

obs_flat  = Y_test_np.ravel()
pred_flat = Y_pred.ravel()

valid = ~(np.isnan(obs_flat) | np.isnan(pred_flat))

rmse = np.sqrt(mean_squared_error(obs_flat[valid], pred_flat[valid]))
r2   = r2_score(obs_flat[valid], pred_flat[valid])
bias = float(np.mean(pred_flat[valid] - obs_flat[valid]))

print("───  ───")
msg = (
    f"Test period metrics (global, all pixels)\n"
    f"  R²   : {r2:.4f}\n"
    f"  RMSE : {rmse:.4f} mm/day\n"
    f"  Bias : {bias:+.4f} mm/day\n"
)
vprint("meduim", msg)

# In[31]:


# cell 24: Quick spatial check — mean predicted vs observed ET
print("[cell 24] : Quick spatial check — mean predicted vs observed ET")

lat_vals = evap_lmdz35.lat.values
lon_vals = evap_lmdz35.lon.values

def quick_plot_spatial_check(Y_test_np, Y_pred, lat_vals, lon_vals, path_out):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4),
                            subplot_kw={'aspect': 'equal'})

    mean_obs  = np.nanmean(Y_test_np, axis=0)
    mean_pred = np.nanmean(Y_pred,    axis=0)
    diff      = mean_pred - mean_obs

    vmin, vmax = np.nanpercentile(mean_obs, [2, 98])

    im0 = axes[0].pcolormesh(lon_vals, lat_vals, mean_obs,
                            cmap='YlGnBu', vmin=vmin, vmax=vmax)
    axes[0].set_title('Mean ERA5 ET (obs)')
    plt.colorbar(im0, ax=axes[0], label='mm/day')

    im1 = axes[1].pcolormesh(lon_vals, lat_vals, mean_pred,
                            cmap='YlGnBu', vmin=vmin, vmax=vmax)
    axes[1].set_title('Mean predicted ET (AE)')
    plt.colorbar(im1, ax=axes[1], label='mm/day')

    lim = np.nanpercentile(np.abs(diff), 95)
    im2 = axes[2].pcolormesh(lon_vals, lat_vals, diff,
                            cmap='RdBu_r', vmin=-lim, vmax=lim)
    axes[2].set_title('Bias (pred − obs)')
    plt.colorbar(im2, ax=axes[2], label='mm/day')

    for ax in axes:
        ax.set_xlabel('Lon'); ax.set_ylabel('Lat')

    plt.suptitle('Autoencoder ET downscaling — test period mean', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(path_out, 'spatial_check.png'), dpi=150, bbox_inches='tight')
    plt.show()
quick_plot_spatial_check(Y_test_np, Y_pred, lat_vals, lon_vals, path_out)

# In[32]:


# cell 25: Save NetCDF
print("[cell 25] : Save NetCDF")

ds_out = xr.Dataset(
    {
        "evap_predite": (["time", "lat", "lon"], Y_pred),
        "evap_obs":     (["time", "lat", "lon"], Y_test_np),
    },
    coords={
        "time": time_test,
        "lat":  lat_vals,
        "lon":  lon_vals,
    },
)
ds_out["evap_predite"].attrs["units"] = "mm/day"
ds_out["evap_obs"].attrs["units"]     = "mm/day"

ds_out.attrs["model"] = f"{Model_name} — {loss_name} loss"
nc_path = os.path.join(path_out, f'{Model_name}_ET_2009_2014.nc')

ds_out.to_netcdf(nc_path)
vprint("meduim", f"NetCDF saved to {nc_path}")

# In[33]:


# Cell 26: Align predictions with LMDZ35 reference
print("[cell 26] :  Align predictions with LMDZ35 reference")
gdf_morocco  = gpd.read_file(os.path.join(path_shp, "morocco_unified_fixed_v2.shp"))
morocco_geom = unary_union(gdf_morocco.geometry)

# Build Morocco geometry
morocco_gdf  = gpd.GeoDataFrame(geometry=[morocco_geom], crs="EPSG:4326")

# Wrap Y_pred and Y_test_np as xarray with correct coords
lat_vals  = evap_lmdz35.lat.values
lon_vals  = evap_lmdz35.lon.values

time_vals_eval = time_test
evap_pred_xr = xr.DataArray(
    Y_pred,
    dims=["time_counter", "lat", "lon"],
    coords={"time_counter": time_vals_eval, "lat": lat_vals, "lon": lon_vals}
)
evap_obs_xr = xr.DataArray(
    Y_test_np,
    dims=["time_counter", "lat", "lon"],
    coords={"time_counter": time_vals_eval, "lat": lat_vals, "lon": lon_vals}
)

vprint("high", f"Prediction xarray: {evap_pred_xr.shape}, Observation xarray: {evap_obs_xr.shape}")

# In[34]:


# Cell 27: Build Morocco-only point mask
print("[cell 27] :  Build Morocco-only point mask")

lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)
coords_df    = pd.DataFrame({"lat": lat2d.ravel(), "lon": lon2d.ravel()})
gdf_coords   = gpd.GeoDataFrame(
    coords_df, geometry=gpd.points_from_xy(coords_df.lon, coords_df.lat)
)
valid_coords = coords_df[gdf_coords.geometry.within(morocco_geom)]
valid_lats   = valid_coords["lat"].values
valid_lons   = valid_coords["lon"].values
vprint("high", f"Valid Morocco points: {len(valid_coords)} / {len(coords_df)}")

# In[35]:


# Cell 28: Stack to select Morocco points
print("[cell 28] :  Stack to select Morocco points")

def select_morocco(da):
    stacked   = da.stack(points=("lat", "lon"))
    valid_midx = pd.MultiIndex.from_arrays(
        [valid_lats, valid_lons], names=["lat", "lon"]
    )
    return stacked.sel(points=valid_midx)

pred_stacked = select_morocco(evap_pred_xr)
obs_stacked  = select_morocco(evap_obs_xr)

vprint("high", f"{Model_name} | shape : {pred_stacked.shape} | NaN: {np.isnan(pred_stacked.values).sum()}")

# In[36]:


# Cell 29: Compute per-point metrics (KGE, R², Bias, Correlation)
print("[cell 29] :  Compute per-point metrics (KGE, R², Bias, Correlation)")
metrics = {
    "Model": [], "KGE": [], "Bias (%)": [],
    "Correlation": [], "R²": [], "Bias (mm/day)": [],
    "RMSE": [], "MAE": [], "MAE_Q95": [], "NSE": [], "ubRMSE": [],
    "lat": [], "lon": []
}

for lat_val, lon_val in zip(valid_lats, valid_lons):
    pt  = (lat_val, lon_val)
    ref = obs_stacked.sel(points=pt).values
    pred = pred_stacked.sel(points=pt).values
    metrics["Model"].append(f"{Model_name}")
    metrics["lat"].append(lat_val)
    metrics["lon"].append(lon_val)
    metrics["R²"].append(r2_score(ref, pred))

    
    #=====correlation=======
    r = (
        np.corrcoef(pred, ref)[0, 1]
        if np.std(ref) > 0 and np.std(pred) > 0
        else 0.0   # flat series → no correlation → 0
    )
    metrics["Correlation"].append(r)

    #=====KGE=======
    # beta
    beta = np.mean(pred) / np.mean(ref) if np.mean(ref) > 0 else np.nan

    # gamma
    gamma = (
        (np.std(pred) / np.mean(pred)) / (np.std(ref) / np.mean(ref))
        if not np.isclose(np.mean(pred), 0) and not np.isclose(np.std(ref), 0)
        else 1.0   # flat series → assume perfect variability ratio → gamma=1
    )
    kge = (
        1 - np.sqrt((r-1)**2 + (beta-1)**2 + (gamma-1)**2)
        if not np.isnan(gamma) else np.nan
    )
    metrics["KGE"].append(kge)

    #=====Bias %=====
    bias_pct = 100 * (np.mean(pred) - np.mean(ref)) / np.mean(ref)
    metrics["Bias (%)"].append(bias_pct)

    #=====absolute Bias=====
    bias_abs = float(np.mean(pred) - np.mean(ref))
    metrics["Bias (mm/day)"].append(bias_abs)

    #=====rmse=====
    rmse = np.sqrt(np.mean((pred - ref) ** 2))
    metrics["RMSE"].append(rmse)


    #=====MAE=====
    mae  = np.mean(np.abs(pred - ref))
    metrics["MAE"].append(mae)
    
    #=====q95=====
    q95  = np.percentile(ref, 95)
    mask = ref >= q95
    mae_q95 = np.mean(np.abs(pred[mask] - ref[mask])) if mask.sum() > 0 else np.nan
    metrics["MAE_Q95"].append(mae_q95)

    #=====NSE=====
    ss_res = np.sum((ref - pred) ** 2)
    ss_tot = np.sum((ref - np.mean(ref)) ** 2)
    nse = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    metrics["NSE"].append(nse)

    #=====ubRMSE=====
    bias_val = np.mean(pred) - np.mean(ref)
    ubrmse = np.sqrt(np.mean(((pred - np.mean(pred)) - (ref - np.mean(ref))) ** 2))
    metrics["ubRMSE"].append(ubrmse)


df_metrics = pd.DataFrame(metrics)
vprint("high",f"Valid grid points: {len(df_metrics)}")
vprint("high",f"Number of NAN metrics values : {df_metrics.isnull().sum()}")

# In[37]:


# Cell 30: Spatial maps per metric
print("[cell 30] :  Spatial maps per metric")
metric_cfg = {
    "KGE": {
        "bounds": [-1, -0.8, -0.6, -0.5, 0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "cmap"  : mpl.cm.RdYlGn,
        "label" : "KGE"},
    "R²": {
        "bounds": [-1, -0.5, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "cmap"  : mpl.cm.YlGn,
        "label" : "R²"},
    "Correlation": {
        "bounds": [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "cmap"  : mpl.cm.YlGn,
        "label" : "Pearson r"},
    "Bias (%)": {
        "bounds": [-60, -30, -20, -10, -5, -2, 0, 2, 5, 10, 20, 30, 60],
        "cmap"  : mpl.cm.RdBu_r,
        "label" : "Bias (%)"},
    "Bias (mm/day)": {
        "bounds": [-5, -4, -3, -2, -1.5, -1.0, -0.5, -0.2, -0.05, 0, 0.05, 0.2, 0.5, 1.0, 1.5, 2, 3, 4, 5],
        "cmap"  : mpl.cm.RdBu_r,
        "label" : "Bias (mm/day)"},
    "RMSE": {
        "bounds": [0, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
        "cmap"  : mpl.cm.YlOrRd,
        "label" : "RMSE (mm/day)"},
    "MAE": {
        "bounds": [0, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0],
        "cmap"  : mpl.cm.YlOrRd,
        "label" : "MAE (mm/day)"},
    "MAE_Q95": {
        "bounds": [0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0],
        "cmap"  : mpl.cm.YlOrRd,
        "label" : "MAE Q95 (mm/day)"},
    "NSE": {
        "bounds": [-1, -0.5, 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "cmap"  : mpl.cm.RdYlGn,
        "label" : "NSE"},
    "ubRMSE": {
        "bounds": [0, 0.2, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0],
        "cmap"  : mpl.cm.YlOrRd,
        "label" : "ubRMSE (mm/day)"},
    
}

# build coordinate arrays for pcolormesh
lat_vals = np.array(sorted(df_metrics["lat"].unique()))
lon_vals = np.array(sorted(df_metrics["lon"].unique()))
lon2d, lat2d = np.meshgrid(lon_vals, lat_vals)
minx, miny, maxx, maxy = morocco_gdf.total_bounds

fig, axes = plt.subplots(2, 5, figsize=(25, 10),
                         subplot_kw={"projection": ccrs.PlateCarree()}, dpi=150)
axes_flat = axes.flatten()

for ax, (metric, cfg) in zip(axes_flat, metric_cfg.items()):
    grid = np.full((len(lat_vals), len(lon_vals)), np.nan)
    for _, row in df_metrics.iterrows():
        li = np.searchsorted(lat_vals, row["lat"])
        lo = np.searchsorted(lon_vals, row["lon"])
        grid[li, lo] = row[metric]

    bounds  = cfg["bounds"]
    ncolors = len(bounds) - 1
    cmap    = mpl.colors.ListedColormap(cfg["cmap"](np.linspace(0, 1, ncolors)))
    norm    = BoundaryNorm(boundaries=bounds, ncolors=ncolors)

    ax.pcolormesh(lon2d, lat2d, grid, cmap=cmap, norm=norm, shading="auto")
    morocco_gdf.boundary.plot(ax=ax, edgecolor="black", linewidth=1.5)
    ax.set_extent([minx, maxx, miny, maxy])

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels   = False
    gl.right_labels = False
    gl.xlabel_style = {"size": 8}
    gl.ylabel_style = {"size": 8}

    vmin_val = np.nanmin(df_metrics[metric].values)
    vmax_val = np.nanmax(df_metrics[metric].values)
    med_val  = np.nanmedian(df_metrics[metric].values)
    ax.set_title(f"{Model_name} — {metric}\n"
                 f"min={vmin_val:.3f}  med={med_val:.3f}  max={vmax_val:.3f}", fontsize=9)

    cbar_ax = ax.inset_axes([1.05, 0, 0.05, 1])
    mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap, norm=norm,
                               boundaries=bounds, ticks=bounds,
                               orientation="vertical").set_label(cfg["label"], fontsize=8)

plt.suptitle(f"Spatial evaluation, {Model_name} point-wise ET downscaling (2009–2014)", y=1.02)
plt.subplots_adjust(left=0.03, right=0.92, wspace=0.3)
plt.savefig(os.path.join(path_out, "spatial_metrics.png"), dpi=150, bbox_inches="tight")
plt.show()

# In[38]:


# Cell 31: Boxplots
print("[cell 31] : Boxplots")
fig, axes = plt.subplots(2, 5, figsize=(25, 8))
axes_flat = axes.flatten()

for ax, metric in zip(axes_flat, ["KGE", "R²", "Correlation", "Bias (%)", "Bias (mm/day)",
                                    "RMSE", "MAE", "MAE_Q95", "NSE", "ubRMSE"]):
    data = df_metrics[df_metrics["Model"] == f"{Model_name}"][metric].dropna()

    vmin_val = data.min()
    vmax_val = data.max()
    med_val  = data.median()

    bp = ax.boxplot(data, patch_artist=True,
                    boxprops=dict(facecolor='steelblue', alpha=0.7),
                    medianprops=dict(color='green', linewidth=2))

    # Annotate median value as text
    ax.text(1, med_val, f' {med_val:.3f}',
            va='center', ha='left', fontsize=9,
            color='red', fontweight='bold')

    ax.set_title(
        f"{metric}\n"
        f"min={vmin_val:.3f}  med={med_val:.3f}  max={vmax_val:.3f}",
        fontsize=9
    )
    ax.set_xticklabels([f"{Model_name}"])
    ax.grid(alpha=0.3)

    if metric in ["KGE", "Bias (%)", "Bias (mm/day)"]:
        ax.axhline(0, color='gray', linestyle='--', lw=0.8)
    elif metric in ["R²", "Correlation", "NSE"]:
        ax.axhline(1, color='gray', linestyle='--', lw=0.8)
    elif metric in ["RMSE", "MAE", "MAE_Q95", "ubRMSE"]:
        ax.axhline(0, color='gray', linestyle='--', lw=0.8)

plt.suptitle(f"{Model_name} point-wise ET downscaling — metric distributions (2009–2014)")
plt.tight_layout()
plt.savefig(os.path.join(path_out, "boxplots_metrics.png"), dpi=150)
plt.show()

# In[39]:


# cell 32: Load per-region shapefiles
print("[cell 32] : Load per-region shapefiles")

region_names = ["east", "north", "north_east", "south"]
gdfs = {name: gpd.read_file(os.path.join(path_shp_regions, f"{name}.shp"))
        for name in region_names}

# Assign a cluster id to every valid (lat, lon) point
unique_coords = pd.DataFrame({"lat": valid_lats, "lon": valid_lons})
gdf_coords    = gpd.GeoDataFrame(
    unique_coords,
    geometry=gpd.points_from_xy(unique_coords.lon, unique_coords.lat)
)
gdf_coords["region"] = np.nan
names  = {0: "east", 1: "north", 2: "north_east", 3: "south"}
id_map = {v: k for k, v in names.items()}   # {"east":0, ...}

for rname, gdf in gdfs.items():
    region_geom = unary_union(gdf.geometry)
    mask = gdf_coords.geometry.within(region_geom)
    gdf_coords.loc[mask, "region"] = id_map[rname]

# Build cluster_lookup
cluster_lookup = (
    gdf_coords[["lat", "lon", "region"]]
    .dropna(subset=["region"])
    .copy()
)
cluster_lookup["region"] = cluster_lookup["region"].astype(int)

print(cluster_lookup["region"].value_counts().sort_index()
      .rename(index=names).to_string())
print(f"\nTotal points assigned: {len(cluster_lookup)} / {len(unique_coords)}")


# Assign unassigned points to nearest region
unassigned = gdf_coords[gdf_coords["region"].isna()].copy()
if len(unassigned) > 0:
    print(f"Assigning {len(unassigned)} unassigned points to nearest region...")
    for idx, row in unassigned.iterrows():
        pt = Point(row.geometry)
        min_dist, nearest = float('inf'), -1
        for rname, gdf in gdfs.items():
            dist = unary_union(gdf.geometry).distance(pt)
            if dist < min_dist:
                min_dist = dist
                nearest  = id_map[rname]
        gdf_coords.loc[idx, "region"] = nearest

# Rebuild cluster_lookup after fixing
cluster_lookup = (
    gdf_coords[["lat", "lon", "region"]]
    .dropna(subset=["region"])
    .copy()
)


cluster_lookup["region"] = cluster_lookup["region"].astype(int)
vprint("meduim",f"{cluster_lookup["region"].value_counts().sort_index().rename(index=names).to_string()}")
vprint("meduim",f"Total points assigned: {len(cluster_lookup)} / {len(unique_coords)}")

# In[40]:


# cell 33: Build clean time series per region (cluster) for each model
print("[cell 33] : Build clean time series per region (cluster) for each model")
COLORS = {
    "LMDZ35": "#2c2c2c",
    "LSTM"  : "#3a86ff",
}
MODELS = ["LSTM"]
regions = list(names.values())

_preds = {
    "LMDZ35": obs_stacked,
    "LSTM"  : pred_stacked,
}
# Build cluster_ts
cluster_ts = {}
for cid, cname in names.items():
    sub  = cluster_lookup[cluster_lookup["region"] == cid]
    pts  = list(zip(sub["lat"], sub["lon"]))
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
            np.column_stack(cols) if cols else np.empty((730, 0))
        )
    n = cluster_ts[cname]["LMDZ35"].shape[1]
    print(f"  {cname:12s} → {n} clean points")
    vprint("meduim",f"{cname:12s} → {n} clean points")

# In[41]:


# cell 34: Time series: regional mean over full test period
print("[cell 34] : Time series: regional mean over full test period")

def plot_timeseries_regional_mean(TEST_YEAR, cluster_ts, save_dir, show=True):
    time_idx = pd.date_range(
        start=f'{TEST_YEAR}-01-01',
        end=f'{TEST_YEAR}-12-31',
        freq='D'
    )
    n_days = len(time_idx)

    fig, axes = plt.subplots(4, 1, figsize=(16, 12), sharex=True)
    fig.suptitle(
        f"Regional Mean Evaporation, Test Period {TEST_YEAR}\n"
        "(ERA5 variability vs model predictions)",
        fontsize=13
    )

    for ax, region in zip(axes, regions):

        era5_mean = np.nanmean(cluster_ts[region]['LMDZ35'], axis=1)[:n_days]

        ax.fill_between(time_idx, era5_mean, alpha=0.15, color=COLORS['LMDZ35'])
        ax.plot(time_idx, era5_mean, color=COLORS['LMDZ35'], lw=2, label='LMDZ35')

        for m in MODELS:
            pred_mean = np.nanmean(cluster_ts[region][m], axis=1)[:n_days]
            ax.plot(time_idx, pred_mean, color=COLORS[m], lw=1.4,
                    ls='--', label=m, alpha=0.9)

        mu = era5_mean.mean()
        ax.axhline(mu, color='gray', lw=1, ls=':', label=f'LMDZ35 mean ({mu:.2f})')

        ax.set_ylabel('mm/day')
        ax.set_title(region.replace('_', ' ').title(), fontsize=10, loc='left')
        ax.legend(fontsize=8, ncol=6, loc='upper right')

        ax.xaxis.set_major_locator(plt.matplotlib.dates.MonthLocator())
        ax.xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter('%b'))

    # year label on bottom axis
    mid = pd.Timestamp(f'{TEST_YEAR}-07-01')
    axes[-1].text(mid, -0.4, str(TEST_YEAR), ha='center', fontsize=9,
                  color='gray', transform=axes[-1].get_xaxis_transform())

    plt.tight_layout()
    plt.savefig(f"{save_dir}/timeseries_regional_mean_{TEST_YEAR}.png", dpi=150, bbox_inches='tight')
    if show:
        plt.show()
    else:
        plt.close()   # ← important: free memory without displaying
        vprint("high",f"  Saved (not displayed) to {save_dir}/timeseries_regional_mean_{TEST_YEAR}.png")


for year in TEST_YEARS:
    print()
    vprint("high",f"\nPlotting regional mean time series for test period {year}...")
    plot_timeseries_regional_mean(year, cluster_ts, save_dir=path_out, show=(year == TEST_YEARS[0])) 

# In[42]:


# cell 35: Stats table and  Negative values bar chart
print("[cell 35] : Stats table and  Negative values bar chart")

rows = []
for name, da in _preds.items():
    v = da.values.ravel()
    rows.append({
        'Model'      : name,
        'Min'        : np.nanmin(v).round(3),
        'Max'        : np.nanmax(v).round(3),
        'Mean'       : np.nanmean(v).round(3),
        'P99'        : np.nanpercentile(v, 99).round(3),
        'P99.9'      : np.nanpercentile(v, 99.9).round(3),
        'Neg (%)' : round(100 * np.sum(v < 0) / np.sum(~np.isnan(v)), 3),
    })

df_stats = pd.DataFrame(rows).set_index('Model')
vprint("high",f"{df_stats.to_string()}")

models_with_neg = df_stats[df_stats['Neg (%)'] > 0].index.tolist()

fig, ax = plt.subplots(figsize=(7, 4))
ax.set_title("% Negative Predictions per Model", fontsize=11)

bars = ax.bar(
    df_stats.index,
    df_stats['Neg (%)'],
    color=[COLORS[m] for m in df_stats.index],
    alpha=0.5
)
# label each bar
for bar, val in zip(bars, df_stats['Neg (%)']):
    if val > 0:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}%', ha='center', va='bottom', fontsize=9)

ax.axhline(0, color='black', lw=1)
ax.set_ylabel('% of predictions < 0')
ax.set_ylim(bottom=0)

plt.tight_layout()
plt.savefig(f"{path_out}/negative_values_pct.png", dpi=150, bbox_inches='tight')
plt.show()

# In[43]:


# cell 36: location of negative predictions (only models with negatives)
print("[cell 36] : location of negative predictions (only models with negatives)")

if models_with_neg:
    fig, axes = plt.subplots(1, len(models_with_neg),
                             figsize=(5 * len(models_with_neg), 4))
    if len(models_with_neg) == 1:
        axes = [axes]

    fig.suptitle("Negative LSTM Predictions (% of Time Steps < 0 per Grid Point)")

    da_map = {'LSTM': pred_stacked}

    for ax, m in zip(axes, models_with_neg):
        da   = da_map[m]                          # shape (730, points)
        vals = da.values                           # (730, n_points)

        # % of time steps that are negative per point
        neg_pct = 100 * np.sum(vals < 0, axis=0) / vals.shape[0]

        # recover lat/lon from the MultiIndex
        pts_idx = da.indexes['points']
        lats    = np.array([p[0] for p in pts_idx])
        lons    = np.array([p[1] for p in pts_idx])

        gdf_morocco.boundary.plot(ax=ax, color='black', lw=0.8)

        sc = ax.scatter(lons, lats, c=neg_pct, cmap='Reds',
                        s=18, vmin=0, vmax=neg_pct.max() or 1)
        plt.colorbar(sc, ax=ax, label='% negative')
        ax.set_xlabel('Longitude'); ax.set_ylabel('Latitude')
        ax.set_aspect('equal')

    plt.tight_layout()
    plt.savefig(f"{path_out}/negative_locations_map.png", dpi=150, bbox_inches='tight')
    plt.show()
else:
    vprint("low",f"No models produce negative values.")

# In[44]:


# cell 37: Plot seasonal ET distributions (boxplots) per region
print("[cell 37] : Plot seasonal ET distributions (boxplots) per region")

time_idx = pd.DatetimeIndex(time_vals_eval) 
SEASONS = {
    'Winter': [12, 1, 2],
    'Spring': [3, 4, 5],
    'Summer': [6, 7, 8],
    'Autumn': [9, 10, 11],
}

all_names = ['LMDZ35'] + MODELS
n_models  = len(all_names)

fig, axes = plt.subplots(1, 4, figsize=(18, 5), sharey=True)
fig.suptitle(
    "Seasonal Evaporation Distribution by Region",
    fontsize=12
)

for ax, region in zip(axes, regions):
    season_list  = list(SEASONS.keys())
    n_seasons    = len(season_list)

    # spacing: groups of models per season, gap between seasons
    group_width  = n_models * 0.9
    gap          = 1.2
    positions    = []
    tick_centers = []

    for s_idx, season in enumerate(season_list):
        months = SEASONS[season]
        mask   = time_idx.month.isin(months)
        center = s_idx * (group_width + gap)
        tick_centers.append(center + group_width / 2 - 0.45)

        for m_idx, name in enumerate(all_names):
            daily = np.nanmean(cluster_ts[region][name], axis=1)
            data  = daily[mask]
            data  = data[~np.isnan(data)]

            pos = center + m_idx * 0.9
            bp  = ax.boxplot(
                data,
                positions=[pos],
                widths=0.7,
                patch_artist=True,
                showfliers=False,
                medianprops=dict(color='black', lw=1.5),
                boxprops=dict(facecolor=COLORS[name], alpha=0.75),
                whiskerprops=dict(color=COLORS[name]),
                capprops=dict(color=COLORS[name]),
            )

    ax.set_xticks(tick_centers)
    ax.set_xticklabels(season_list, fontsize=10)
    ax.set_title(region.replace('_', ' ').title(), fontsize=11)
    if ax == axes[0]:
        ax.set_ylabel('Evap (mm/day)')

# legend
handles = [plt.matplotlib.patches.Patch(color=COLORS[n], label=n) for n in all_names]
axes[-1].legend(handles=handles, fontsize=8, loc='upper right')

plt.tight_layout()
plt.savefig(f"{path_out}/seasonal_distributions.png", dpi=150, bbox_inches='tight')
plt.show()

# In[45]:


print("df_metrics columns   :", df_metrics.columns.tolist())


# In[46]:


# cell 38: Attach cluster labels to df_metrics
print("[cell 38] : Attach cluster labels to df_metrics")

df_metrics = df_metrics.merge(
    cluster_lookup[["lat", "lon", "region"]],
    on=["lat", "lon"],
    how="left"
)

# Points not covered by any region shapefile will be NaN — drop them
df_metrics = df_metrics.dropna(subset=["region"])
df_metrics["region"] = df_metrics["region"].astype(int)

print(f"df_metrics after cluster join: {len(df_metrics)} rows")
print()
vprint("high",f"df_metrics after cluster join: {len(df_metrics)} rows")
vprint("high",f"{df_metrics["region"].value_counts().sort_index().rename(index=names)}")
vprint("high", f"{df_metrics.head()}")

# In[47]:


# cell 37 : Compute and save median evaluation metrics per region
print("[cell 37] : Compute and save median stats")

def save_median_stats_by_region(df, metrics, names, save_dir):
    """
    Compute median of each metric per model per region and save to CSV.
    names: {0: 'east', 1: 'north', 2: 'north_east', 3: 'south'}
    """
    # ── use whatever models are actually in df, not a hardcoded list ──
    models  = df["Model"].unique().tolist()
    regions = list(names.values())

    if not models:
        raise ValueError("df has no rows / no 'Model' column values — check upstream cells.")

    rows = []
    for metric in metrics:
        for model in models:
            row = {"Metric": metric, "Model": model}
            for region in regions:
                cid  = [k for k, v in names.items() if v == region][0]
                vals = df.loc[
                    (df["Model"] == model) & (df["region"] == cid), metric
                ].dropna()
                row[region] = vals.median() if not vals.empty else np.nan
            rows.append(row)

    stats_df = pd.DataFrame(rows).set_index(["Metric", "Model"])

    os.makedirs(f"{save_dir}/Metrics", exist_ok=True)
    fname = f"{save_dir}/Metrics/median_by_region.csv"
    stats_df.to_csv(fname, float_format="%.4f")
    vprint("high", f"Saved to {fname}")
    vprint("high", f"{stats_df.round(4).to_string()}")  # also fixed: was printing the string literal

    return stats_df

metrics_list = ["KGE", "Bias (%)", "Correlation", "R²", "Bias (mm/day)", "RMSE", "MAE", "MAE_Q95", "NSE", "ubRMSE"]
stats = save_median_stats_by_region(df_metrics, metrics_list, names, save_dir=path_out)

# In[48]:


# Cell 40: Plot median evaluation metrics per region (bar charts)
print("[cell 40] : Plot median evaluation metrics per region (bar charts)")

REGION_COLORS = {
    "east"      : "#e07b39",
    "north"     : "#3a86ff",
    "north_east": "#06d6a0",
    "south"     : "#ef233c",
}

metrics_to_plot = ["KGE", "NSE", "R²", "Correlation", "Bias (%)", "Bias (mm/day)",
                   "RMSE", "MAE", "MAE_Q95", "ubRMSE"]

fig, axes = plt.subplots(2, 5, figsize=(28, 10))
axes_flat = axes.flatten()
fig.suptitle(f"{Model_name}, Median metrics by region (2009–2014)", fontsize=12)
region_list     = list(names.values())  # ["east","north","north_east","south"]
for ax, metric in zip(axes_flat, metrics_to_plot):
    vals = []
    colors = []
    for region in region_list:
        cid  = [k for k, v in names.items() if v == region][0]
        data = df_metrics.loc[df_metrics["region"] == cid, metric].dropna()
        vals.append(data.median() if not data.empty else np.nan)
        colors.append(REGION_COLORS[region])

    bars = ax.bar(region_list, vals, color=colors, alpha=0.85, edgecolor="black", linewidth=0.5)

    # value label on each bar
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
        ax.axhline(1, color="green", lw=1, ls="--", alpha=0.5, label="perfect=1")
        ax.axhline(0, color="gray",  lw=0.8, ls=":")
    if metric in ["Bias (%)", "Bias (mm/day)"]:
        ax.axhline(0, color="gray",  lw=1, ls="--", label="no bias")
    if metric in ["RMSE", "MAE", "MAE_Q95", "ubRMSE"]:
        ax.axhline(0, color="gray",  lw=0.8, ls=":")
    

    ax.set_title(metric, fontsize=10)
    ax.set_ylabel(metric)
    ax.set_xticklabels([r.replace("_", "\n") for r in region_list], fontsize=8)
    ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(path_out, "metrics_by_region.png"), dpi=150, bbox_inches="tight")
plt.show()

# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



