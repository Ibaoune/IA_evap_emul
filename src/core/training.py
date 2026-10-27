"""
training.py
===========
Point-wise training loop for LSTM / MLP / XGB / MLR-RF.
Logic is identical to the original — only imports are updated.
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler
import time

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.models.LSTM import PointLSTM, make_sequences
from src.models.MLP  import PointMLP
from src.models.XGB  import build_xgb
from src.models.MLR_RF import build_mlr_rf
from src.models.AutoEncoder import AutoEncoder
from src.models.ConvLSTM import ConvLSTM
from src.models.RCAN import RCAN
from src.core.utils import SequenceDataset, EarlyStopper, vprint
from src.core.losses import build_loss


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

    criterion        = build_loss(cfg.loss_type).to(cfg.device)
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
                          num_layers=cfg.num_layers, dropout=cfg.dropout,
                          out_channels=criterion.n_outputs).to(cfg.device)

        elif cfg.model_type == "mlp":
            Xtr_t = torch.tensor(xtr, dtype=torch.float32)
            ytr_t = torch.tensor(ytr, dtype=torch.float32)
            Xva_t = torch.tensor(xva, dtype=torch.float32)
            yva_t = torch.tensor(yva, dtype=torch.float32)
            Xte_t = torch.tensor(xte, dtype=torch.float32)

            m = PointMLP(input_dim=len(feature_names), hidden_dims=cfg.hidden_layers,
                         dropout=cfg.mlp_dropout,
                         out_channels=criterion.n_outputs).to(cfg.device)

        elif cfg.model_type == "xgb":
            m = build_xgb(cfg)
            m.fit(xtr, ytr, eval_set=[(xva, yva)], verbose=False)

            pred    = m.predict(xte).clip(min=0)
            tr_pred = m.predict(xtr).clip(min=0)

        elif cfg.model_type == "mlr_rf":
            lr, rf = build_mlr_rf(cfg)
            lr.fit(xtr, ytr)
            lr_pred_tr = lr.predict(xtr)
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
            early_stopper = EarlyStopper(patience=cfg.early_stopping_max, min_delta=cfg.min_delta)
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
                pred    = criterion.predict(m(Xte_t.to(cfg.device))).cpu().numpy()
                tr_pred = criterion.predict(m(Xtr_t.to(cfg.device))).cpu().numpy()

            all_train_losses.append(train_losses_pt)
            all_val_losses.append(val_losses_pt)
            best_val_log = f"best_val={best_val:.4f} | "
        else:
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


# ==========================================================
# Spatial training loop (AutoEncoder, ConvLSTM, RCAN)
# ==========================================================

def SpatialTrain(cfg,
                 X_train, Y_train, T_train,
                 X_val,   Y_val,   T_val,
                 X_test,  Y_test,  T_test,
                 feature_names, nlat, nlon, lats, lons, morocco_mask):
    """
    X_*  : (T, nlat, nlon, C)  float32
    Y_*  : (T, nlat, nlon)     float32
    """
    vprint(f"Starting spatial {cfg.model_type.upper()} training...")

    # Transpose to (T, C, nlat, nlon)
    X_tr = X_train.transpose(0, 3, 1, 2).copy()
    X_va = X_val.transpose(0, 3, 1, 2).copy()
    X_te = X_test.transpose(0, 3, 1, 2).copy()

    EPS = 1e-8
    # ── per-pixel temporal normalization (fit on train only) ──
    # Skip one-hot or mask channels if we want, but for now we normalize everything
    # except masks which are usually 0/1. Let's just normalize channels strictly.
    for c in range(X_tr.shape[1]):
        name = feature_names[c]
        if name in ["land_mask", "soiltile_bare", "soiltile_short", "soiltile_tall", "month_sin", "month_cos"]:
            continue
        mean_map = X_tr[:, c, :, :].mean(axis=0)
        std_map  = X_tr[:, c, :, :].std(axis=0)
        std_map  = np.where(std_map < EPS, 1.0, std_map)
        X_tr[:, c, :, :] = (X_tr[:, c, :, :] - mean_map) / std_map
        X_va[:, c, :, :] = (X_va[:, c, :, :] - mean_map) / std_map
        X_te[:, c, :, :] = (X_te[:, c, :, :] - mean_map) / std_map

    INPUT_SHAPE = (X_tr.shape[1], nlat, nlon)
    OUTPUT_SHAPE = (1, nlat, nlon)

    mask_tensor = torch.BoolTensor(morocco_mask).to(cfg.device)
    criterion   = build_loss(cfg.loss_type, mask=mask_tensor).to(cfg.device)
    n_out       = criterion.n_outputs

    # ── model-specific setup ──
    if cfg.model_type == "autoencoder":
        model = AutoEncoder(INPUT_SHAPE, OUTPUT_SHAPE, latent_dim=cfg.ae_latent_dim, out_channels=n_out)
        is_seq = False
    elif cfg.model_type == "convlstm":
        model = ConvLSTM(in_channels=INPUT_SHAPE[0], hidden_channels=cfg.clstm_hidden_channels, out_channels=n_out)
        is_seq = True
        seq_len = cfg.seq_len if hasattr(cfg, 'seq_len') and cfg.seq_len is not None else 7
    elif cfg.model_type == "rcan":
        model = RCAN(INPUT_SHAPE, OUTPUT_SHAPE, n_feat=cfg.rcan_n_feat, n_groups=cfg.rcan_n_groups, n_rcab=cfg.rcan_n_rcab, reduction=cfg.rcan_reduction, out_channels=n_out)
        is_seq = False
    else:
        raise ValueError(f"Unknown spatial model type: {cfg.model_type}")

    model = model.to(cfg.device)

    # ── datasets & loaders ──
    if is_seq:
        train_ds = SequenceDataset(X_tr, Y_train, seq_len)
        val_ds   = SequenceDataset(X_va, Y_val, seq_len)
        test_ds  = SequenceDataset(X_te, Y_test, seq_len)
        te_offset = seq_len
    else:
        # Standard TensorDataset
        train_ds = TensorDataset(torch.FloatTensor(X_tr), torch.FloatTensor(Y_train).unsqueeze(1))
        val_ds   = TensorDataset(torch.FloatTensor(X_va), torch.FloatTensor(Y_val).unsqueeze(1))
        test_ds  = TensorDataset(torch.FloatTensor(X_te), torch.FloatTensor(Y_test).unsqueeze(1))
        te_offset = 0

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader  = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    optimizer = optim.Adam(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=cfg.scheduler_patience, min_lr=1e-6)
    early_stopper = EarlyStopper(patience=cfg.early_stopping_max, min_delta=cfg.min_delta)

    best_val_loss = float('inf')
    best_w = None

    all_train_losses, all_val_losses = [], []
    start_time = time.time()

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running_train = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(cfg.device), yb.to(cfg.device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(yb, pred)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_train += loss.item()
        
        train_loss = running_train / len(train_loader)
        all_train_losses.append(train_loss)

        model.eval()
        running_val = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(cfg.device), yb.to(cfg.device)
                running_val += criterion(yb, model(xb)).item()
        val_loss = running_val / max(len(val_loader), 1)
        all_val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_w = {k: v.clone() for k, v in model.state_dict().items()}

        vprint(f"Epoch {epoch:3d}/{cfg.epochs} | train={train_loss:.4f} | val={val_loss:.4f} | lr={optimizer.param_groups[0]['lr']:.6f}")

        if early_stopper.early_stop(val_loss):
            vprint(f"Early stop at epoch {epoch}")
            break

    if best_w is not None:
        model.load_state_dict(best_w)

    vprint(f"Training completed in {(time.time() - start_time)/60:.1f} minutes")

    # ── Inference on whole domain ──
    model.eval()
    all_preds_te = []
    with torch.no_grad():
        for xb, _ in test_loader:
            all_preds_te.append(criterion.predict(model(xb.to(cfg.device))).cpu().numpy())
    pred_te = np.concatenate(all_preds_te, axis=0)  # (T_te, nlat, nlon)

    # ── Package into model_preds same format as PointWiseTrain ──
    # Note: PointWiseTrain returns a list of dicts per valid point.
    # To maintain compatibility with train.py, we can just reshape pred_te into the same format!
    model_preds = []
    valid_points = [(i, j) for i in range(nlat) for j in range(nlon) if morocco_mask[i, j]]

    for (li, lj) in valid_points:
        te_df = pd.DataFrame({
            "time":    T_test[te_offset:],
            "target":  Y_test[te_offset:, li, lj],
            "lat_idx": li,
            "lon_idx": lj,
        })
        model_preds.append({
            "lat":      lats[li], "lon":     lons[lj],
            "lat_idx":  li,       "lon_idx": lj,
            "te_df":    te_df,
            "pred":     pred_te[:, li, lj],
            "tr_pred":  np.zeros(len(Y_train)), # spatial doesn't easily return train preds in bulk
            "tr_target": Y_train[:, li, lj],
        })

    vprint(f"{cfg.model_type.upper()} spatial training done.")
    return model_preds, all_train_losses, all_val_losses
