import torch
import torch.nn as nn

class ChannelAttention(nn.Module):
    def __init__(self, n_feat, reduction=16):
        super().__init__()
        self.body = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),          # global avg pool → (B, C, 1, 1)
            nn.Conv2d(n_feat, n_feat // reduction, 1),  # Conv 1×1, 4
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat // reduction, n_feat, 1),  # Conv 1×1, 64
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.body(x)              # element-wise product

class RCAB(nn.Module):
    def __init__(self, n_feat=64, reduction=16, dropout=0.1):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout), 
            nn.Conv2d(n_feat, n_feat, 3, padding=1),
            ChannelAttention(n_feat, reduction)
        )
    def forward(self, x):
        return x + self.body(x)              # RCAB skip connection

class ResidualGroup(nn.Module):
    def __init__(self, n_feat=64, n_rcab=20, reduction=16):
        super().__init__()
        self.body = nn.Sequential(
            *[RCAB(n_feat, reduction) for _ in range(n_rcab)],
            nn.Conv2d(n_feat, n_feat, 3, padding=1)
        )
    def forward(self, x):
        return x + self.body(x)              # RG skip connection

class RCAN(nn.Module):
    def __init__(self, input_shape, output_shape,
                 n_feat=64, n_groups=10, n_rcab=20, reduction=16, out_channels=3):
        """
        input_shape  : (C, lat, lon)
        output_shape : (1, lat, lon)
        out_channels : number of raw output channels (set by the loss)
        """
        super().__init__()
        self.output_shape = output_shape

        # Head
        self.head = nn.Conv2d(input_shape[0], n_feat, 3, padding=1)

        # 10 Residual Groups + long skip connection
        self.body = nn.Sequential(*[
            ResidualGroup(n_feat, n_rcab, reduction)
            for _ in range(n_groups)
        ])

        self.body_end = nn.Conv2d(n_feat, n_feat, 3, padding=1)

        # Upscale to fine grid (non-integer scale → bilinear + conv)
        self.upsample = nn.Sequential(
            nn.Upsample(size=(output_shape[1], output_shape[2]),
                        mode='bilinear', align_corners=False),
            nn.Conv2d(n_feat, n_feat, 3, padding=1),
            nn.ReLU(inplace=True)
        )

        # Tail
        self.tail = nn.Conv2d(n_feat, out_channels, 3, padding=1)

    def forward(self, x):
        head = self.head(x)
        body = self.body_end(self.body(head))
        out  = head + body
        out  = self.upsample(out)
        return self.tail(out)                    # (B, out_channels, lat_f, lon_f)
