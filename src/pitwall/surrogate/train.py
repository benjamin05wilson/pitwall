"""Train + evaluate the surrogate, and load it for inference."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from .model import SurrogateNet, gaussian_nll

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "data" / "surrogate" / "surrogate.pt"


def train(
    X: np.ndarray, Y: np.ndarray, *, epochs: int = 120, batch_size: int = 256,
    lr: float = 5e-4, val_frac: float = 0.15, seed: int = 0, verbose: bool = True,
) -> tuple[SurrogateNet, dict]:
    """Y columns: [mean_pos, std_pos, p_podium, p_points]."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    n = len(X)
    idx = rng.permutation(n)
    n_val = int(n * val_frac)
    vi, ti = idx[:n_val], idx[n_val:]

    net = SurrogateNet(in_dim=X.shape[1])
    net.set_norm(X[ti].mean(0), X[ti].std(0))

    Xt = torch.tensor(X, dtype=torch.float32)
    Yt = torch.tensor(Y, dtype=torch.float32)
    train_dl = DataLoader(TensorDataset(Xt[ti], Yt[ti]), batch_size=batch_size, shuffle=True)
    opt = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    bce = nn.BCEWithLogitsLoss()

    best_val, best_state = float("inf"), None
    for ep in range(epochs):
        net.train()
        for xb, yb in train_dl:
            opt.zero_grad()
            out = net(xb)
            loss = (
                gaussian_nll(out["pos_mean"], out["pos_logvar"], yb[:, 0])
                + 0.5 * nn.functional.mse_loss(out["std"], yb[:, 1])
                + bce(out["prob_logits"][:, 0], yb[:, 2])
                + bce(out["prob_logits"][:, 1], yb[:, 3])
            )
            loss.backward()
            opt.step()
        sched.step()
        net.eval()
        with torch.no_grad():
            out = net(Xt[vi])
            vpos = nn.functional.l1_loss(out["pos_mean"], Yt[vi, 0]).item()
        if vpos < best_val:
            best_val, best_state = vpos, {k: v.clone() for k, v in net.state_dict().items()}
        if verbose and (ep % 20 == 0 or ep == epochs - 1):
            print(f"  epoch {ep:3d}  val MAE(pos)={vpos:.3f}")

    if best_state:
        net.load_state_dict(best_state)
    metrics = evaluate_surrogate(net, Xt[vi].numpy(), Y[vi])
    metrics["val_mae_pos"] = best_val
    return net, metrics


def evaluate_surrogate(net: SurrogateNet, X: np.ndarray, Y: np.ndarray) -> dict:
    pred = net.predict(X)
    mae_pos = float(np.mean(np.abs(pred["mean_pos"] - Y[:, 0])))
    rmse_pos = float(np.sqrt(np.mean((pred["mean_pos"] - Y[:, 0]) ** 2)))
    # Calibration of the uncertainty: fraction of truths within +/-1 sigma.
    within = np.mean(np.abs(pred["mean_pos"] - Y[:, 0]) <= pred["pos_uncertainty"])
    mae_pod = float(np.mean(np.abs(pred["p_podium"] - Y[:, 2])))
    ss_res = np.sum((pred["mean_pos"] - Y[:, 0]) ** 2)
    ss_tot = np.sum((Y[:, 0] - Y[:, 0].mean()) ** 2)
    return {
        "mae_pos": mae_pos, "rmse_pos": rmse_pos, "r2_pos": float(1 - ss_res / ss_tot),
        "uncertainty_coverage_1sigma": float(within), "mae_p_podium": mae_pod,
        "n_val": len(Y),
    }


def save(net: SurrogateNet, path: Path | str = DEFAULT_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": net.state_dict(), "in_dim": net.x_mean.numel()}, path)


def load(path: Path | str = DEFAULT_PATH) -> SurrogateNet:
    ckpt = torch.load(path, map_location="cpu")
    net = SurrogateNet(in_dim=ckpt["in_dim"])
    net.load_state_dict(ckpt["state_dict"])
    net.eval()
    return net
