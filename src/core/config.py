import yaml
import torch
import os


def _to_int(x):
    return int(x) if x is not None else None

def _to_float(x):
    return float(x) if x is not None else None

def _to_bool(x):
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        return x.lower() in ["true", "1", "yes", "y"]
    return bool(x)


class Config:
    def __init__(self, cfg_dict, config_path, train_mode=True):
        self.train_mode  = train_mode
        self.config_path = config_path   # kept for reference / copying into results

        # conf_name is derived from the yaml filename, e.g. "conf0"
        self.conf_name = os.path.splitext(os.path.basename(config_path))[0]

        # ----------------------
        # Device
        # ----------------------
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # ----------------------
        # General
        # ----------------------
        self.verbose    = _to_bool(cfg_dict["general"].get("verbose", True))
        self.model_type = str(cfg_dict["general"].get("model_type", "lstm"))

        # ----------------------
        # Training
        # ----------------------
        tr = cfg_dict["training"]
        self.learning_rate       = _to_float(tr.get("learning_rate"))
        self.epochs              = _to_int(tr.get("epochs"))
        self.batch_size          = _to_int(tr.get("batch_size"))
        self.loss_type           = str(tr.get("loss_type"))
        self.early_stopping_max  = _to_int(tr.get("early_stopping_max"))
        self.scheduler_patience  = _to_int(tr.get("scheduler_patience"))
        self.weight_decay        = _to_float(tr.get("weight_decay"))

        # ----------------------
        # LSTM parameters
        # ----------------------
        lstm = cfg_dict.get("lstm", {})
        self.hidden_dim = _to_int(lstm.get("hidden_dim"))
        self.num_layers = _to_int(lstm.get("num_layers"))
        self.dropout    = _to_float(lstm.get("dropout"))
        self.seq_len    = _to_int(lstm.get("seq_len"))

        # ----------------------
        # MLP parameters
        # ----------------------
        mlp = cfg_dict.get("mlp", {})
        self.mlp_batch_size   = _to_int(mlp.get("batch_size"))
        self.mlp_weight_decay = _to_float(mlp.get("weight_decay"))
        self.hidden_layers    = mlp.get("hidden_layers")
        self.mlp_dropout      = _to_float(mlp.get("dropout"))

        # ----------------------
        # XGBoost parameters
        # ----------------------
        xgb = cfg_dict.get("xgb", {})
        self.xgb_n_estimators     = _to_int(xgb.get("n_estimators", 500))
        self.xgb_max_depth        = _to_int(xgb.get("max_depth", 6))
        self.xgb_learning_rate    = _to_float(xgb.get("learning_rate", 0.05))
        self.xgb_subsample        = _to_float(xgb.get("subsample", 0.8))
        self.xgb_colsample_bytree = _to_float(xgb.get("colsample_bytree", 0.8))
        self.xgb_n_jobs           = _to_int(xgb.get("n_jobs", -1))

        # ----------------------
        # MLR-RF parameters
        # ----------------------
        mlr_rf = cfg_dict.get("mlr_rf", {})
        self.mlr_rf_n_estimators = _to_int(mlr_rf.get("n_estimators", 100))
        self.mlr_rf_max_depth    = _to_int(mlr_rf.get("max_depth", 8))
        self.mlr_rf_n_jobs       = _to_int(mlr_rf.get("n_jobs", -1))

        # ----------------------
        # Region
        # ----------------------
        reg = cfg_dict["region"]
        self.lon_min = _to_float(reg.get("lon_min"))
        self.lon_max = _to_float(reg.get("lon_max"))
        self.lat_min = _to_float(reg.get("lat_min"))
        self.lat_max = _to_float(reg.get("lat_max"))

        # ----------------------
        # Dates
        # ----------------------
        self.train_start = _to_int(cfg_dict["dates"]["train"]["start"])
        self.train_end   = _to_int(cfg_dict["dates"]["train"]["end"])
        self.val_start   = _to_int(cfg_dict["dates"]["val"]["start"])
        self.val_end     = _to_int(cfg_dict["dates"]["val"]["end"])
        self.test_start  = _to_int(cfg_dict["dates"]["test"]["start"])
        self.test_end    = _to_int(cfg_dict["dates"]["test"]["end"])

        self.train_years = list(range(self.train_start, self.train_end + 1))
        self.val_years   = list(range(self.val_start,   self.val_end   + 1))
        self.test_years  = list(range(self.test_start,  self.test_end  + 1))

        # ----------------------
        # Paths
        # ----------------------
        paths = cfg_dict["paths"]
        self.data_LMDZ35_path   = str(paths["data_LMDZ35_path"])
        self.data_LMDZ250_path  = str(paths["data_LMDZ250_path"])
        self.altitude_path      = str(paths["altitude_path"])
        self.soil_moisture_path = str(paths["soil_moisture_path"])
        self.soiltile_path      = str(paths["soiltile_path"])
        self.shp_path           = str(paths["shp_path"])
        self.shp_regions_path   = str(paths["shp_regions_path"])
        self.tsol_path          = str(paths["tsol_path"])
        self.model_save_dir     = str(paths["model_save_dir"])

        # Results go in results/<MODEL_TYPE>/<conf_name>/
        base_results = str(paths["results_dir"])
        self.path_out = os.path.join(base_results, self.model_type, self.conf_name)

        os.makedirs(self.model_save_dir, exist_ok=True)
        os.makedirs(self.path_out,       exist_ok=True)


def load_config(config_path, train_mode=True):
    """
    Load config by merging configs/common.yaml with the given model config yaml.

    Args:
        config_path : path to the model-specific yaml, e.g. configs/XGB/conf0.yaml
        train_mode  : bool
    """
    # Always load common first
    common_path = os.path.join(os.path.dirname(os.path.dirname(config_path)), "common.yaml")
    with open(common_path, "r") as f:
        cfg_dict = yaml.safe_load(f)

    # Merge model-specific yaml (model keys override common)
    with open(config_path, "r") as f:
        model_cfg = yaml.safe_load(f)

    # Deep merge: model yaml wins at the key level inside each section
    for section, values in model_cfg.items():
        if section in cfg_dict and isinstance(cfg_dict[section], dict):
            cfg_dict[section].update(values)
        else:
            cfg_dict[section] = values

    return Config(cfg_dict, config_path=config_path, train_mode=train_mode)
