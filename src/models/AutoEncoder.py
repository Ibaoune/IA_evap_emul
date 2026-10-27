import torch
import torch.nn as nn
import torch.nn.functional as F

class AutoEncoder(nn.Module):
    def __init__(self, input_shape, output_shape, latent_dim, out_channels=3):
        """
        input_shape : tuple (C, LAT, LON)
        output_shape: tuple (1, LAT, LON)
        latent_dim  : int
        out_channels: number of raw output channels (set by the loss)
        """
        super(AutoEncoder, self).__init__()
        self.input_shape  = input_shape
        self.output_shape = output_shape
        self.latent_dim   = latent_dim
        
        # Encoder
        self.encoder_conv1 = nn.Conv2d(input_shape[0], 50, kernel_size=3, padding=1)
        self.encoder_conv2 = nn.Conv2d(50, 25, kernel_size=3, padding=1)
        self.encoder_conv3 = nn.Conv2d(25,  1, kernel_size=3, padding=1)
        flat_size = 1 * input_shape[1] * input_shape[2]
        self.encoder_fc   = nn.Linear(flat_size, latent_dim)
        
        # Decoder
        self.decoder_fc    = nn.Linear(latent_dim, 1 * output_shape[1] * output_shape[2])
        self.decoder_conv0 = nn.ConvTranspose2d(1,  1,  kernel_size=3, padding=1)
        self.decoder_conv1 = nn.ConvTranspose2d(1,  25, kernel_size=3, padding=1)
        self.decoder_conv2 = nn.ConvTranspose2d(25, 50, kernel_size=3, padding=1)
        self.decoder_out   = nn.ConvTranspose2d(50, out_channels, kernel_size=3, padding=1)

    def encode(self, x):
        x = F.relu(self.encoder_conv1(x))
        x = F.relu(self.encoder_conv2(x))
        x = F.relu(self.encoder_conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.encoder_fc(x))
        return x

    def decode(self, z):
        z = F.relu(self.decoder_fc(z))
        z = z.view(z.size(0), 1, self.output_shape[1], self.output_shape[2])
        z = F.relu(self.decoder_conv0(z))
        z = F.relu(self.decoder_conv1(z))
        z = F.relu(self.decoder_conv2(z))
        return self.decoder_out(z)           # (B, out_channels, LAT, LON)

    def forward(self, x):
        return self.decode(self.encode(x))
