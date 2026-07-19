import os
import numpy as np
import torch
import xarray as xr
import geopandas as gpd
from shapely.geometry import Point
from shapely.ops import unary_union
import torch.nn as nn


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
# Loss functions
# ==========================================================

class BernoulliGammaLossPointwise(nn.Module):
    def __init__(self):
        super(BernoulliGammaLossPointwise, self).__init__()

    def forward(self, true, pred):
        # true: (B,)   pred: (B, 3)
        # Prevent negative values which cause NaN in torch.log(true + epsilon)
        true = torch.clamp(true, min=0.0)
        
        occurrence      = pred[:, 0]
        shape_parameter = torch.exp(pred[:, 1])
        scale_parameter = torch.exp(pred[:, 2])
        bool_rain       = (true > 0).float()
        epsilon         = 1e-6

        loss = -(
            (1 - bool_rain) * torch.log(1 - occurrence + epsilon) +
            bool_rain * (
                torch.log(occurrence + epsilon) +
                (shape_parameter - 1) * torch.log(true + epsilon) -
                shape_parameter * torch.log(scale_parameter + epsilon) -
                torch.lgamma(shape_parameter + epsilon) -
                true / (scale_parameter + epsilon)
            )
        )
        return loss.mean()


# ==========================================================
# Early stopping
# ==========================================================

class EarlyStopper:
    def __init__(self, patience=20, min_delta=0.001):
        self.patience     = patience
        self.min_delta    = min_delta
        self.counter      = 0
        self.min_val_loss = float('inf')

    def early_stop(self, val_loss):
        if val_loss < self.min_val_loss - self.min_delta:
            self.min_val_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience


# ==========================================================
# Spatial Loss & Dataset
# ==========================================================

class BernoulliGammaLossSpatial(nn.Module):
    def __init__(self, mask=None):
        """
        mask : bool tensor (LAT, LON) — True inside Morocco.
               If provided, loss is averaged only over valid pixels.
        """
        super(BernoulliGammaLossSpatial, self).__init__()
        if mask is not None:
            self.register_buffer('mask', mask.float())
        else:
            self.mask = None

    def forward(self, true, pred):
        true = true.squeeze(1)                           # (B, H, W)
        
        # Replace NaNs in true with 0 and prevent negative values which cause NaN in torch.log
        true = torch.nan_to_num(true, nan=0.0).clamp(min=0.0)

        if self.mask is not None:
            # Zero out predictions and target outside the mask to prevent inf/nan from blowing up
            pred = pred * self.mask.unsqueeze(0).unsqueeze(0)
            true = true * self.mask.unsqueeze(0)

        occurrence      = pred[:, 0, :, :]
        # Clamp to prevent torch.exp() overflow
        shape_parameter = torch.exp(torch.clamp(pred[:, 1, :, :], max=20.0, min=-20.0))
        scale_parameter = torch.exp(torch.clamp(pred[:, 2, :, :], max=20.0, min=-20.0))
        bool_rain       = (true > 0).float()
        epsilon         = 1e-6

        loss_map = -(                                    # (B, H, W)
            (1 - bool_rain) * torch.log(1 - occurrence + epsilon) +
            bool_rain * (
                torch.log(occurrence + epsilon) +
                (shape_parameter - 1) * torch.log(true + epsilon) -
                shape_parameter * torch.log(scale_parameter + epsilon) -
                torch.lgamma(shape_parameter + epsilon) -
                true / (scale_parameter + epsilon)
            )
        )

        if self.mask is not None:
            return (loss_map * self.mask).sum() / (self.mask.sum() * loss_map.shape[0])
        return loss_map.mean()


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, X_np, Y_np, seq_len):
        self.X   = X_np
        self.Y   = Y_np
        self.seq = seq_len

    def __len__(self):
        return len(self.X) - self.seq

    def __getitem__(self, i):
        x = torch.FloatTensor(self.X[i : i + self.seq])
        y = torch.FloatTensor(self.Y[i + self.seq]).unsqueeze(0)
        return x, y
