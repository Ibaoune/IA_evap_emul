import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
from lstm_arch  import PointLSTM, make_sequences
from utils  import BernoulliGammaLossPointwise, EarlyStopper
from mlp_arch import PointMLP
from utils import vprint
import time
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb



# ==========================================================
# LSTM / MLP / XGB / MLR-RF training (point-wise)
# ==========================================================
def PointWiseTrain(cfg,
                   X_train, Y_train, T_train,
                   X_val,   Y_val,   T_val,
                   X_test,  Y_test,  T_test,
                   feature_names, nlat, nlon, lats, lons, morocco_mask):
    """
    X_*  : (T, nlat, nlon, C)  float32
    Y_*  : (T, nlat, nlon)     float32
    T_*  : DatetimeIndex of length T
    morocco_mask : (nlat, nlon) bool
    """
    vprint(f"Starting point-wise {cfg.model_type.upper()} training...")

    criterion        = BernoulliGammaLossPointwise()
    model_preds      = []
    all_train_losses = []
    all_val_losses   = []
    start            = time.time()

    valid_points = [(i, j) for i in range(nlat) for j in range(nlon) if morocco_mask[i, j]]
    n_points     = len(valid_points)

    for idx, (li, lj) in enumerate(valid_points):

        # ── extract per-point slices ──────────────────────────────────
        xtr = X_train[:, li, lj, :].astype(np.float32)
        ytr = Y_train[:, li, lj].astype(np.float32)
        xva = X_val[:,   li, lj, :].astype(np.float32)
        yva = Y_val[:,   li, lj].astype(np.float32)
        xte = X_test[:,  li, lj, :].astype(np.float32)
        yte = Y_test[:,  li, lj].astype(np.float32)

        # ── per-point standardization ─────────────────────────────────
        pt_scaler = StandardScaler()
        xtr = pt_scaler.fit_transform(xtr)
        xva = pt_scaler.transform(xva)
        xte = pt_scaler.transform(xte)

        # ── model-specific setup & training ──────────────────────────
        if cfg.model_type == "lstm":
            Xtr_np, ytr_np = make_sequences(xtr, ytr, seq_len=cfg.seq_len)
            Xva_np, yva_np = make_sequences(xva, yva, seq_len=cfg.seq_len)
            context        = np.concatenate([xva[-cfg.seq_len:], xte], axis=0)
            ycontext       = np.concatenate([yva[-cfg.seq_len:], yte], axis=0)
            Xte_np, _      = make_sequences(context, ycontext, seq_len=cfg.seq_len)

            Xtr_t = torch.tensor(Xtr_np, dtype=torch.float32)
            ytr_t = torch.tensor(ytr_np, dtype=torch.float32)
            Xva_t = torch.tensor(Xva_np, dtype=torch.float32)
            yva_t = torch.tensor(yva_np, dtype=torch.float32)
            Xte_t = torch.tensor(Xte_np, dtype=torch.float32)

            m = PointLSTM(input_dim=len(feature_names), hidden_dim=cfg.hidden_dim,
                          num_layers=cfg.num_layers, dropout=cfg.dropout).to(cfg.device)

        elif cfg.model_type == "mlp":
            Xtr_t = torch.tensor(xtr, dtype=torch.float32)
            ytr_t = torch.tensor(ytr, dtype=torch.float32)
            Xva_t = torch.tensor(xva, dtype=torch.float32)
            yva_t = torch.tensor(yva, dtype=torch.float32)
            Xte_t = torch.tensor(xte, dtype=torch.float32)

            m = PointMLP(input_dim=len(feature_names), hidden_dims=cfg.hidden_layers,
                         dropout=cfg.mlp_dropout).to(cfg.device)

        elif cfg.model_type == "xgb":
            import xgboost as xgb_lib
            m = xgb_lib.XGBRegressor(
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
            m.fit(xtr, ytr, eval_set=[(xva, yva)], verbose=False)

            pred    = m.predict(xte).clip(min=0)
            tr_pred = m.predict(xtr).clip(min=0)

        elif cfg.model_type == "mlr_rf":
            lr = LinearRegression()
            lr.fit(xtr, ytr)
            lr_pred_tr = lr.predict(xtr)

            rf = RandomForestRegressor(
                n_estimators = cfg.mlr_rf_n_estimators,
                max_depth    = cfg.mlr_rf_max_depth,
                n_jobs       = cfg.mlr_rf_n_jobs,
                random_state = 42,
            )
            rf.fit(xtr, ytr - lr_pred_tr)

            pred    = (lr.predict(xte) + rf.predict(xte)).clip(min=0)
            tr_pred = (lr_pred_tr      + rf.predict(xtr)).clip(min=0)

        # ── PyTorch training loop (lstm / mlp only) ───────────────────
        if cfg.model_type in ("lstm", "mlp"):
            wd = cfg.mlp_weight_decay if cfg.model_type == "mlp" else cfg.weight_decay
            bs = cfg.mlp_batch_size   if cfg.model_type == "mlp" else cfg.batch_size

            opt = optim.Adam(m.parameters(), lr=cfg.learning_rate, weight_decay=wd)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="min", factor=0.5, patience=cfg.scheduler_patience, min_lr=1e-6)
            loader = DataLoader(TensorDataset(Xtr_t, ytr_t),
                                batch_size=bs, shuffle=True, drop_last=True)
            early_stopper = EarlyStopper(patience=cfg.early_stopping_max, min_delta=0.001)
            best_val, best_w = float('inf'), None

            train_losses_pt, val_losses_pt = [], []
            for epoch in range(cfg.epochs):
                m.train()
                epoch_train = 0.0
                for xb, yb in loader:
                    xb, yb = xb.to(cfg.device), yb.to(cfg.device)
                    opt.zero_grad()
                    loss = criterion(yb, m(xb))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                    opt.step()
                    epoch_train += loss.item()
                train_losses_pt.append(epoch_train / len(loader))

                m.eval()
                with torch.no_grad():
                    vl = criterion(yva_t.to(cfg.device), m(Xva_t.to(cfg.device))).item()
                val_losses_pt.append(vl)
                scheduler.step(vl)

                if vl < best_val:
                    best_val = vl
                    best_w   = {k: v.clone() for k, v in m.state_dict().items()}
                if early_stopper.early_stop(vl):
                    break

            if best_w is not None:
                m.load_state_dict(best_w)
            m.eval()
            with torch.no_grad():
                raw_te = m(Xte_t.to(cfg.device)).cpu().numpy()
                raw_tr = m(Xtr_t.to(cfg.device)).cpu().numpy()

            pred    = (raw_te[:, 0] * np.exp(raw_te[:, 1]) * np.exp(raw_te[:, 2])).clip(min=0)
            tr_pred = (raw_tr[:, 0] * np.exp(raw_tr[:, 1]) * np.exp(raw_tr[:, 2])).clip(min=0)

            all_train_losses.append(train_losses_pt)
            all_val_losses.append(val_losses_pt)
            best_val_log = f"best_val={best_val:.4f} | "
        else:
            # xgb / mlr_rf have no epoch losses
            best_val_log = ""

        # ── build te_df & store result ────────────────────────────────
        offset = cfg.seq_len if cfg.model_type == "lstm" else 0
        te_df  = pd.DataFrame({
            "time":    T_test[offset:],
            "target":  yte[offset:],
            "lat_idx": li,
            "lon_idx": lj,
        })

        model_preds.append({
            "lat":      lats[li], "lon":     lons[lj],
            "lat_idx":  li,       "lon_idx": lj,
            "te_df":    te_df,
            "pred":     pred,
            "tr_pred":  tr_pred,
            "tr_target": ytr,
        })

        if (idx + 1) % 100 == 0 or (idx + 1) == n_points:
            elapsed = (time.time() - start) / 60
            eta     = (elapsed / (idx + 1)) * (n_points - idx - 1)
            vprint(f"  [{idx+1}/{n_points}] lat={lats[li]:.2f} lon={lons[lj]:.2f} "
                   f"| {best_val_log}{elapsed:.1f}min ETA {eta:.1f}min")

    vprint(f"{cfg.model_type.upper()} point-wise training done.")
    return model_preds, all_train_losses, all_val_losses