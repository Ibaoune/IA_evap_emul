import torch
import torch.nn as nn



class PointMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 64], dropout=0.2):
        super().__init__()
        layers, in_d = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(in_d, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            in_d = h
        layers.append(nn.Linear(in_d, 3))   # 3 outputs: occurrence, log-shape, log-scale
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        out = self.net(x)                                              # (B, 3)
        return torch.cat([torch.sigmoid(out[:, :1]), out[:, 1:]], dim=1)
