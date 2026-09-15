import xgboost as xgb


def build_xgb(cfg):
    """Build and return an XGBRegressor from config."""
    return xgb.XGBRegressor(
        n_estimators          = cfg.xgb_n_estimators,
        max_depth             = cfg.xgb_max_depth,
        learning_rate         = cfg.xgb_learning_rate,
        subsample             = cfg.xgb_subsample,
        colsample_bytree      = cfg.xgb_colsample_bytree,
        n_jobs                = cfg.xgb_n_jobs,
        early_stopping_rounds = 20,
        eval_metric           = "rmse",
        verbosity             = 0,
        random_state          = 42,
    )
