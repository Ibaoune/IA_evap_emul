import torch
import torch.nn as nn

class ConvLSTMCell(nn.Module):
    def __init__(self, in_channels, hidden_channels, kernel_size=5):
        super().__init__()
        self.hidden_channels = hidden_channels
        pad = kernel_size // 2
        # One conv computes all 4 gates at once
        self.gates = nn.Conv2d(
            in_channels + hidden_channels,
            4 * hidden_channels,
            kernel_size, padding=pad
        )

    def forward(self, x, h, c):
        # x: (B, in_C, H, W)   h/c: (B, hid_C, H, W)
        i, f, g, o = self.gates(torch.cat([x, h], dim=1)).chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next

    def init_hidden(self, B, H, W, device):
        z = torch.zeros(B, self.hidden_channels, H, W, device=device)
        return z, z.clone()

class ConvLSTM(nn.Module):
    """
    Paper: 4-layer stacked ConvLSTM
      Layers 1–3 : 64 filters, 5×5 kernel
      Layer  4   : 1×1 conv (output head)
    Modification: output head → `out_channels` raw channels, set by the loss.
    Input : (B, T, C, H, W)
    Output: (B, out_channels, H, W)
    """
    def __init__(self, in_channels, hidden_channels=32, kernel_size=5, out_channels=3):
        super().__init__()
        h = hidden_channels
        self.cell1 = ConvLSTMCell(in_channels, h, kernel_size)  # layer 1
        self.cell2 = ConvLSTMCell(h,           h, kernel_size)  # layer 2
        self.cell3 = ConvLSTMCell(h,           h, kernel_size)  # layer 3

        # Layer 4 — 1×1 conv output head
        self.drop = nn.Dropout2d(0.2)
        self.tail = nn.Conv2d(h, out_channels, 1)

    def forward(self, x):
        B, T, C, H, W = x.shape

        h1, c1 = self.cell1.init_hidden(B, H, W, x.device)
        h2, c2 = self.cell2.init_hidden(B, H, W, x.device)
        h3, c3 = self.cell3.init_hidden(B, H, W, x.device)

        for t in range(T):
            h1, c1 = self.cell1(x[:, t], h1, c1)
            h2, c2 = self.cell2(self.drop(h1), h2, c2)   # ← dropout here
            h3, c3 = self.cell3(self.drop(h2), h3, c3)   # ← and here

        # Predictions from last hidden state
        return self.tail(h3)                          # (B, out_channels, H, W)
