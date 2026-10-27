import os
import random
import numpy as np
import torch
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union


# ==========================================================
# Verbose printing
# ==========================================================
_VERBOSE = True

def set_verbose(flag: bool):
    global _VERBOSE
    _VERBOSE = flag

def vprint(*args, **kwargs):
    if _VERBOSE:
        print(*args, **kwargs)


# ==========================================================
# Reproducibility
# ==========================================================

def set_seed(seed, deterministic=False):
    """Seed python, numpy and torch (CPU + all GPUs)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False


# ==========================================================
# Spatial masking
# ==========================================================

def mask_dataset(ds, name):
    """
    Apply a fixed lon/lat spatial mask to a dataset.
    Handles both sel-based and where-based masking.
    """
    lon_range = slice(-18, 0)
    lat_range = slice(20, 36)

    if name in ['mrso', 'mrsos']:
        return ds.sel(lon=lon_range, lat=lat_range)

    var_lon, var_lat = None, None
    for var_name in list(ds.variables):
        if var_name.startswith('lon') or var_name.startswith('longitude'):
            var_lon = var_name
        elif var_name.startswith('lat') or var_name.startswith('latitude'):
            var_lat = var_name

    mask = (
        (ds[var_lon] >= lon_range.start) & (ds[var_lon] <= lon_range.stop) &
        (ds[var_lat] >= lat_range.start) & (ds[var_lat] <= lat_range.stop)
    )
    return ds.where(mask, drop=True)


# ==========================================================
# Morocco land mask
# ==========================================================

def build_morocco_mask(lats, lons, shp_path):
    """
    Build a boolean (lat, lon) mask — True = inside Morocco.

    Args:
        lats, lons : 1-D coordinate arrays
        shp_path   : path to the Morocco shapefile directory
    """
    shp_files = [f for f in os.listdir(shp_path) if f.endswith(".shp")]
    if not shp_files:
        raise FileNotFoundError(f"No .shp file found in {shp_path}")

    gdf     = gpd.read_file(os.path.join(shp_path, shp_files[0]))
    morocco = unary_union(gdf.geometry)

    mask = np.zeros((len(lats), len(lons)), dtype=bool)
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            mask[i, j] = morocco.contains(Point(lon, lat))

    return mask


# ==========================================================
# Dataset inspector
# ==========================================================
def inspect(datasets):
    """
    Print key info for each dataset in a dict:
    shape, dimensions, time steps, grid resolution, NaN count, value range.
    """
    print("\nDATASET INSPECTION\n" + "=" * 50)
    for name, da in datasets.items():
        print(f"\n{name}")
        print(f"  Var   : {da.name}")
        print(f"  Shape : {da.shape} | Dims: {da.dims}")

        t = ("time_counter" if "time_counter" in da.dims
             else "time" if "time" in da.dims else None)
        print(f"  Time  : {len(da[t]) if t else '—'} steps")

        if "lat" in da.dims and "lon" in da.dims:
            lat, lon = da.lat.values, da.lon.values
            print(f"  Grid  : {len(lat)}×{len(lon)}")
            print(f"  Lat   : {lat.min():.2f}→{lat.max():.2f}  "
                  f"Lon: {lon.min():.2f}→{lon.max():.2f}")

        v   = da.values
        nan = np.isnan(v)
        print(f"  NaN   : {nan.sum()} ({nan.mean()*100:.2f}%)")
        v   = v[~nan]
        if v.size:
            print(f"  Range : [{v.min():.3f}, {v.max():.3f}]  Mean: {v.mean():.3f}")
        print("-" * 50)


# ==========================================================
# Model save / load
# ==========================================================

def save_model(cfg, model, train_losses=None, val_losses=None):
    """Save model checkpoint under model_save_dir/<model_type>/."""
    save_dir = os.path.join(cfg.model_save_dir, cfg.model_type)
    os.makedirs(save_dir, exist_ok=True)

    model_path = os.path.join(save_dir, f"{cfg.model_type}_downscaling.pth")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "train_losses":     train_losses,
            "val_losses":       val_losses,
            "config":           vars(cfg),
        },
        model_path,
    )
    vprint(f"Model saved at: {model_path}")
    return model_path


def load_model(cfg, model):
    """Load trained model checkpoint."""
    save_dir   = os.path.join(cfg.model_save_dir, cfg.model_type)
    model_path = os.path.join(save_dir, f"{cfg.model_type}_downscaling.pth")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")

    checkpoint = torch.load(model_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"], strict=False)

    vprint(f"Model loaded from: {model_path}")
    return model, checkpoint.get("train_losses"), checkpoint.get("val_losses")


# ==========================================================
# Early stopping
# ==========================================================

class EarlyStopper:
    def __init__(self, patience=20, min_delta=0.1):
        """
        min_delta is a percentage of the best val loss: an epoch counts as an
        improvement if val_loss < best - (min_delta / 100) * |best|.
        Independent of the loss type and of the target units.
        """
        self.patience     = patience
        self.min_delta    = min_delta / 100
        self.counter      = 0
        self.min_val_loss = float('inf')

    def early_stop(self, val_loss):
        if self.min_val_loss == float('inf'):
            threshold = float('inf')        # first epoch always counts as an improvement
        else:
            # abs(): Bernoulli-Gamma loss can be negative
            threshold = self.min_val_loss - self.min_delta * abs(self.min_val_loss)

        if val_loss < threshold:
            self.min_val_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ==========================================================
# Spatial sequence dataset
# ==========================================================

class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, X_np, Y_np, seq_len):
        self.X   = X_np
        self.Y   = Y_np
        self.seq = seq_len

    def __len__(self):
        return len(self.X) - self.seq + 1

    def __getitem__(self, i):
        # window ends at the target day (inclusive): same-day predictors are used
        x = torch.FloatTensor(self.X[i : i + self.seq])
        y = torch.FloatTensor(self.Y[i + self.seq - 1]).unsqueeze(0)
        return x, y
