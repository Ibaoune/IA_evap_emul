import torch
import torch.nn as nn


class PointMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 64], dropout=0.2, out_channels=3):
        super().__init__()
        layers, in_d = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            in_d = h
        layers.append(nn.Linear(in_d, out_channels))   # raw outputs, interpreted by the loss
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)                   # (B, out_channels)