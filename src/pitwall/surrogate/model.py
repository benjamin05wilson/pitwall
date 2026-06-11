"""PyTorch surrogate network.

A small MLP (256-128-64) that emulates the Monte-Carlo simulator. It predicts the
focal car's expected finishing position with a **heteroscedastic uncertainty
head** (mean + log-variance, trained by Gaussian NLL), plus podium/points
probabilities. Input normalisation is stored as buffers so the saved model is
fully self-contained — load it and call ``predict`` with raw features.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .features import feature_dim


class SurrogateNet(nn.Module):
    def __init__(self, in_dim: int | None = None, hidden=(256, 128, 64), dropout=0.15):
        super().__init__()
        in_dim = in_dim or feature_dim()
        layers, d = [], in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.head_pos_mean = nn.Linear(d, 1)
        self.head_pos_logvar = nn.Linear(d, 1)
        self.head_std = nn.Linear(d, 1)
        self.head_probs = nn.Linear(d, 2)  # podium, points (logits)
        # Normalisation buffers (filled at train time).
        self.register_buffer("x_mean", torch.zeros(in_dim))
        self.register_buffer("x_std", torch.ones(in_dim))

    def set_norm(self, x_mean: np.ndarray, x_std: np.ndarray) -> None:
        self.x_mean = torch.tensor(x_mean, dtype=torch.float32)
        self.x_std = torch.tensor(np.where(x_std < 1e-6, 1.0, x_std), dtype=torch.float32)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.trunk((x - self.x_mean) / self.x_std)
        return {
            "pos_mean": self.head_pos_mean(z).squeeze(-1),
            "pos_logvar": self.head_pos_logvar(z).squeeze(-1).clamp(-6, 6),
            "std": torch.nn.functional.softplus(self.head_std(z).squeeze(-1)),
            "prob_logits": self.head_probs(z),
        }

    @torch.no_grad()
    def predict(self, x: np.ndarray) -> dict[str, np.ndarray]:
        """Raw-feature inference. Returns mean position, its 1-sigma uncertainty,
        predicted spread, and podium/points probabilities."""
        self.eval()
        single = x.ndim == 1
        xt = torch.tensor(np.atleast_2d(x), dtype=torch.float32)
        out = self.forward(xt)
        probs = torch.sigmoid(out["prob_logits"]).numpy()
        res = {
            "mean_pos": out["pos_mean"].numpy(),
            "pos_uncertainty": np.exp(0.5 * out["pos_logvar"].numpy()),
            "std_pos": out["std"].numpy(),
            "p_podium": probs[:, 0],
            "p_points": probs[:, 1],
        }
        return {k: (v[0] if single else v) for k, v in res.items()}


def gaussian_nll(mean: torch.Tensor, logvar: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    inv = torch.exp(-logvar)
    return (0.5 * (logvar + inv * (target - mean) ** 2)).mean()
