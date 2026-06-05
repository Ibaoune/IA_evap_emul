import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from utils import vprint, build_morocco_mask


def split_by_years(X_all, Y_all, times, cfg):
    """
    Split arrays into train / val / test by year.

    Returns:
        X_train, X_val, X_test : np.ndarray
        Y_train, Y_val, Y_test : np.ndarray
        time_train, time_val, time_test : pd.DatetimeIndex
    """
    years = times.year

    train_mask = np.isin(years, cfg.train_years)
    val_mask   = np.isin(years, cfg.val_years)
    test_mask  = np.isin(years, cfg.test_years)

    vprint(f"Train : {train_mask.sum()} samples  "
           f"({cfg.train_years[0]}–{cfg.train_years[-1]})")
    vprint(f"Val   : {val_mask.sum()} samples  "
           f"({cfg.val_years[0]}–{cfg.val_years[-1]})")
    vprint(f"Test  : {test_mask.sum()} samples  "
           f"({cfg.test_years[0]}–{cfg.test_years[-1]})")

    return (
        X_all[train_mask], X_all[val_mask], X_all[test_mask],
        Y_all[train_mask], Y_all[val_mask], Y_all[test_mask],
        times[train_mask], times[val_mask], times[test_mask],
    )


def preprocess_dataPointwise(cfg, X_all, Y_all, times, lats, lons, channel_names):
    T, nlat, nlon, nfeat = X_all.shape

    morocco_mask = build_morocco_mask(lats, lons, cfg.shp_path)  # (nlat, nlon) bool

    years = np.array([t.year for t in times])
    train_mask = np.isin(years, cfg.train_years)
    val_mask   = np.isin(years, cfg.val_years)
    test_mask  = np.isin(years, cfg.test_years)

    X_train, Y_train, T_train = X_all[train_mask], Y_all[train_mask], times[train_mask]
    X_val,   Y_val,   T_val   = X_all[val_mask],   Y_all[val_mask],   times[val_mask]
    X_test,  Y_test,  T_test  = X_all[test_mask],  Y_all[test_mask],  times[test_mask]

    vprint(f"Fine-grid splits -- train: {train_mask.sum()}  val: {val_mask.sum()}  test: {test_mask.sum()} time steps")
    return (X_train, Y_train, T_train,
            X_val,   Y_val,   T_val,
            X_test,  Y_test,  T_test,
            channel_names, nlat, nlon, lats, lons, times, morocco_mask)