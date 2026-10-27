"""
Entry point for training.

Usage: run in terminal from project root, e.g.
    python train.py --config configs/XGB/conf0.yaml
    python train.py --config configs/MLP/conf1.yaml
    python train.py --config configs/LSTM/conf2.yaml
    python train.py --config configs/MLR-RF/conf0.yaml
"""


import os
import sys
import pickle
import shutil
import argparse

# ── project root on the path ─────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.core.config               import load_config
from src.data.data_loading         import load_datasets_fine
from src.data.preprocessing        import preprocess_data
from src.core.training             import PointWiseTrain, SpatialTrain
from src.core.utils                import vprint, set_verbose, set_seed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True,
                        help="Path to model config yaml, e.g. configs/XGB/conf0.yaml")
    parser.add_argument("--dry-run", action="store_true",
                        help="Run a quick dry run with 1 epoch and 1 year of data")
    args = parser.parse_args()

    cfg = load_config(config_path=args.config, train_mode=True)
    set_seed(cfg.seed)
    
    if args.dry_run:
        print("--- DRY RUN MODE ENABLED ---")
        cfg.epochs = 1
        cfg.train_years = [1979]
        cfg.val_years = [2006]
        cfg.test_years = [2009]
        if hasattr(cfg, 'xgb_n_estimators'): cfg.xgb_n_estimators = 2
        if hasattr(cfg, 'mlr_rf_n_estimators'): cfg.mlr_rf_n_estimators = 2
    
    set_verbose(cfg.verbose)

    vprint(f"=== Starting training  |  model={cfg.model_type.upper()}  conf={cfg.conf_name} ===")
    vprint(f"Results will be saved in: {cfg.path_out}")

    # ------------------------------------------------------------------
    # Copy the config yaml into the results folder
    # ------------------------------------------------------------------
    shutil.copy(args.config, os.path.join(cfg.path_out, f"{cfg.conf_name}.yaml"))

    # ------------------------------------------------------------------
    # Load & preprocess
    # ------------------------------------------------------------------
    X_all, Y_all, times, lats, lons, channel_names = load_datasets_fine(cfg)

    (
        X_train, Y_train, T_train,
        X_val,   Y_val,   T_val,
        X_test,  Y_test,  T_test,
        channel_names, nlat, nlon, lats, lons, times, morocco_mask,
    ) = preprocess_data(cfg, X_all, Y_all, times, lats, lons, channel_names)

    # ------------------------------------------------------------------
    # Train
    # ------------------------------------------------------------------
    spatial_models = ["autoencoder", "convlstm", "rcan"]
    
    if cfg.model_type.lower() in spatial_models:
        model_preds, all_train_losses, all_val_losses = SpatialTrain(
            cfg,
            X_train, Y_train, T_train,
            X_val,   Y_val,   T_val,
            X_test,  Y_test,  T_test,
            channel_names, nlat, nlon, lats, lons, morocco_mask,
        )
    else:
        model_preds, all_train_losses, all_val_losses = PointWiseTrain(
            cfg,
            X_train, Y_train, T_train,
            X_val,   Y_val,   T_val,
            X_test,  Y_test,  T_test,
            channel_names, nlat, nlon, lats, lons, morocco_mask,
        )

    # ------------------------------------------------------------------
    # Save predictions pickle inside results
    # ------------------------------------------------------------------
    preds_path = os.path.join(cfg.path_out, f"{cfg.model_type}_preds.pkl")
    with open(preds_path, "wb") as f:
        pickle.dump({
            "model_preds":      model_preds,
            "all_train_losses": all_train_losses,
            "all_val_losses":   all_val_losses,
            "nlat":  nlat,  "nlon":  nlon,
            "lats":  lats,  "lons":  lons,
            "times_fine": times,
        }, f)
    vprint(f"Predictions saved: {preds_path}")

    vprint("=== Training completed ===")


if __name__ == "__main__":
    main()
