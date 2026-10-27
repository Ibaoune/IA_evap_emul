"""
losses.py
=========
Loss functions for the neural models, selected by name via `training.loss_type`.

Each loss owns the full contract between the network and the target:
  - n_outputs : number of raw output channels the network must produce
  - forward   : training objective on raw outputs
  - predict   : conversion of raw outputs into an ET estimate

Models therefore emit raw (unactivated) outputs and stay loss-agnostic.
Tensors are either point-wise (B, n_outputs) or spatial (B, n_outputs, H, W).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================================
# Base class
# ==========================================================

class BaseLoss(nn.Module):
    n_outputs = 1

    def __init__(self, mask=None):
        """
        mask : bool tensor (H, W) — True inside Morocco (spatial models only).
               If provided, the loss is averaged only over valid pixels.
        """
        super().__init__()
        self.register_buffer("mask", mask.float() if mask is not None else None)

    @staticmethod
    def _prep_true(true):
        # (B, 1, H, W) -> (B, H, W); NaNs and negatives would break log terms
        if true.dim() == 4:
            true = true.squeeze(1)
        return torch.nan_to_num(true, nan=0.0).clamp(min=0.0)

    def _reduce(self, loss_map):
        # loss_map: (B,) point-wise or (B, H, W) spatial
        if self.mask is not None and loss_map.dim() == 3:
            return (loss_map * self.mask).sum() / (self.mask.sum() * loss_map.shape[0])
        return loss_map.mean()

    def predict(self, raw):
        """raw: (B, n_outputs[, H, W]) -> ET (B[, H, W])"""
        raise NotImplementedError


# ==========================================================
# Bernoulli-Gamma (occurrence logit, log-shape, log-scale)
# ==========================================================
class BernoulliGammaLoss(BaseLoss):
    n_outputs = 3

    def __init__(self, mask=None, eps=1e-6):
        super().__init__(mask)
        self.eps = eps

    def forward(self, true, raw):
        true = self._prep_true(true)
        if self.mask is not None and true.dim() == 3:
            # Neutralise pixels outside the mask so they cannot produce inf/nan
            raw  = raw * self.mask
            true = true * self.mask

        logit = raw[:, 0]
        # Clamp to prevent torch.exp() overflow
        shape = torch.exp(raw[:, 1].clamp(-20.0, 20.0))
        scale = torch.exp(raw[:, 2].clamp(-20.0, 20.0))
        wet   = (true > 0).float()

        # logsigmoid(x) = log(p), logsigmoid(-x) = log(1 - p): stabler than log(sigmoid + eps)
        loss_map = -(
            (1 - wet) * F.logsigmoid(-logit) +
            wet * (
                F.logsigmoid(logit) +
                (shape - 1) * torch.log(true + self.eps) -
                shape * torch.log(scale) -
                torch.lgamma(shape) -
                true / scale
            )
        )
        return self._reduce(loss_map)

    def predict(self, raw):
        # E[y] = p * shape * scale
        occ = torch.sigmoid(raw[:, 0])
        return (occ * torch.exp(raw[:, 1]) * torch.exp(raw[:, 2])).clamp(min=0)


# ==========================================================
# Deterministic regression losses (single output channel)
# ==========================================================

class MSELoss(BaseLoss):
    def forward(self, true, raw):
        return self._reduce((raw[:, 0] - self._prep_true(true)) ** 2)

    def predict(self, raw):
        return raw[:, 0].clamp(min=0)


class MAELoss(BaseLoss):
    def forward(self, true, raw):
        return self._reduce((raw[:, 0] - self._prep_true(true)).abs())

    def predict(self, raw):
        return raw[:, 0].clamp(min=0)


class HuberLoss(BaseLoss):
    def __init__(self, mask=None, delta=1.0):
        super().__init__(mask)
        self.delta = delta

    def forward(self, true, raw):
        return self._reduce(F.huber_loss(raw[:, 0], self._prep_true(true),
                                         reduction="none", delta=self.delta))

    def predict(self, raw):
        return raw[:, 0].clamp(min=0)


# ==========================================================
# Registry
# ==========================================================

LOSSES = {
    "bernoulli_gamma": BernoulliGammaLoss,
    "mse":             MSELoss,
    "mae":             MAELoss,
    "huber":           HuberLoss,
}


def build_loss(name, mask=None):
    """Instantiate the loss registered under `name` (case-insensitive)."""
    key = name.lower()
    if key not in LOSSES:
        raise ValueError(f"Unknown loss_type '{name}'. Available: {sorted(LOSSES)}")
    return LOSSES[key](mask=mask)
