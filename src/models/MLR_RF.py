from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor


def build_mlr_rf(cfg):
    """Build and return (LinearRegression, RandomForestRegressor) from config."""
    lr = LinearRegression()
    rf = RandomForestRegressor(
        n_estimators = cfg.mlr_rf_n_estimators,
        max_depth    = cfg.mlr_rf_max_depth,
        n_jobs       = cfg.mlr_rf_n_jobs,
        random_state = 42,
    )
    return lr, rf
