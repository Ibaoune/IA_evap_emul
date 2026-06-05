import os
import pickle

from config import load_config
from data_loading import load_datasets_fine
from preprocessing import preprocess_dataPointwise
from training import PointWiseTrain
from utils import vprint, set_verbose, save_model

cfg = load_config(train_mode=True, path="config.yaml")
set_verbose(cfg.verbose)


def main():
    cfg = load_config(train_mode=True)
    vprint(f"=== Starting training ({cfg.model_type.upper()}) ===")


    if cfg.model_type in ("lstm", "mlp", "xgb", "mlr_rf"):
        (X_all, Y_all, times, lats, lons, channel_names) = load_datasets_fine(cfg)

        (
            X_train, Y_train, T_train,
            X_val,   Y_val,   T_val,
            X_test,  Y_test,  T_test,
            channel_names, nlat, nlon, lats, lons, times, morocco_mask
        ) = preprocess_dataPointwise(cfg, X_all, Y_all, times, lats, lons, channel_names)


        # for LSTM, MLP, XGB and MLR-RF training
        model_preds, all_train_losses, all_val_losses = PointWiseTrain(
            cfg, X_train, Y_train, T_train,
                X_val,   Y_val,   T_val,
                X_test,  Y_test,  T_test,
            channel_names, nlat, nlon, lats, lons, morocco_mask
        )

        out_dir = os.path.join(cfg.model_save_dir, cfg.model_type)
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"{cfg.model_type}_preds.pkl"), "wb") as f:
            pickle.dump({
                "model_preds":      model_preds,
                "all_train_losses": all_train_losses,
                "all_val_losses":   all_val_losses,
                "nlat": nlat, "nlon": nlon,
                "lats": lats, "lons": lons,
                "times_fine": times,
            }, f)
        vprint(f"{cfg.model_type.upper()} predictions saved.")

    else:
        raise NotImplementedError(f"model_type '{cfg.model_type}' not supported.")

    vprint("=== Training completed ===")


if __name__ == "__main__":
    main()