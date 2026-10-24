import numpy as np
import torch
import torch.nn as nn


class PointLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_layers=2, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.fc = nn.Linear(hidden_dim, 3)   # 3 outputs: occurrence, log-shape, log-scale

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])         # (B, 3)
        # sigmoid for occurrence, raw for shape/scale (exponentiated in loss)
        return torch.cat([torch.sigmoid(out[:, :1]), out[:, 1:]], dim=1)


def make_sequences(X_vals, y_vals, seq_len=7):
    """
    Build sliding-window sequences from a sorted per-point array.

    Returns:
        X_seq : np.ndarray (N, seq_len, n_features)
        y_seq : np.ndarray (N,)
    """
    X_vals = X_vals.astype(np.float32)
    y_vals = y_vals.astype(np.float32)
    X_seq, y_seq = [], []
    for i in range(seq_len, len(X_vals)):
        X_seq.append(X_vals[i - seq_len: i])
        y_seq.append(y_vals[i])
    return np.array(X_seq), np.array(y_seq)