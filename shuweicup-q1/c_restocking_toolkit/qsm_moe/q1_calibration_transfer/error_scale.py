"""Fit horizon-dependent error scales to frozen QSM centers on the ten fit environments."""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from transfer_config import OUT, WORK
from qsm_data import reproducible_torch
from settings import array

SEEDS = (91103, 91117, 91129)
OFFSETS = np.arange(60, -1, -3)
HORIZONS = np.arange(1, 31, dtype=np.float32)


def time_pieces(cadence):
    age = cadence[:, None] * HORIZONS
    return np.stack([np.clip(age, 0, .10), np.clip(age-.10, 0, .20),
                     np.clip(age-.30, 0, .30), np.maximum(age-.60, 0)], axis=-1).astype(np.float32)


class ErrorScale(nn.Module):
    """Historical motion and the current short-window kinematic features predict four positive rates."""
    def __init__(self, width):
        super().__init__()
        self.encode = nn.Sequential(nn.Linear(width, 40), nn.LayerNorm(40), nn.SiLU())
        self.gru = nn.GRU(40, 48, num_layers=2, dropout=.1, batch_first=True)
        self.recent = nn.Sequential(nn.Linear(width, 32), nn.LayerNorm(32), nn.SiLU())
        self.rates = nn.Sequential(nn.Linear(128, 56), nn.LayerNorm(56), nn.SiLU(), nn.Linear(56, 4))

    def forward(self, series, current, pieces):
        history, _ = self.gru(self.encode(series))
        encoded = torch.cat([history[:, -1], history.mean(1), self.recent(current)], dim=1)
        rate = .004 + .35 * F.softplus(self.rates(encoded))
        return .002 + (pieces * rate[:, None]).sum(-1)


class Features:
    def __init__(self, x, starts, cadence, device):
        self.device = device
        self.x = torch.tensor(x, device=device, dtype=torch.float32)
        self.starts = torch.tensor(starts, device=device, dtype=torch.long)
        self.offsets = torch.tensor(OFFSETS, device=device)
        self.pieces = torch.tensor(time_pieces(cadence), device=device)

    def inputs(self, ids):
        rows = torch.as_tensor(ids, dtype=torch.long, device=self.device)
        history = torch.maximum(rows[:, None]-self.offsets, self.starts[rows, None])
        return self.x[history], self.x[rows], self.pieces[rows]


@torch.inference_mode()
def predict_models(models, features, ids):
    batch_scores = []
    for batch in np.array_split(ids, max(1, int(np.ceil(len(ids)/512)))):
        arguments = features.inputs(batch)
        batch_scores.append(torch.stack([net(*arguments) for net in models]).mean(0).cpu().numpy())
    return np.concatenate(batch_scores)


def fit_scales(d):
    reproducible_torch()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = Features(d.x, array("start"), d.cad, device)
    center = np.load(WORK/"corrected_center.npy", mmap_mode="r")
    fit = np.flatnonzero(d.fit & d.mask[:, -1])
    errors = torch.tensor(np.linalg.norm(center[fit]-d.y[fit], axis=2), device=device)
    usable = torch.tensor(d.mask[fit], device=device)
    _, groups, counts = np.unique(d.env[fit], return_inverse=True, return_counts=True)
    # Equalize environments halfway to reduce dependence on long recordings.
    probability = 1 / np.sqrt(counts[groups])
    probability /= probability.sum()
    models = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = ErrorScale(d.x.shape[1]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.0014, weight_decay=.003)
        schedule = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 16, eta_min=.00018)
        for _ in range(16):
            for batch in np.array_split(rng.choice(len(fit), min(6144, len(fit)),
                                                   replace=False, p=probability), 24):
                predicted = model(*data.inputs(fit[batch]))
                idx = torch.tensor(batch, device=device)
                loss = (errors[idx]/predicted + predicted.log())[usable[idx]].mean()
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 3.)
                optimizer.step()
            schedule.step()
        torch.save(model.state_dict(), OUT/f"error_scale_{seed}.pt")
        models.append(model.eval())
    origin = d.meta.origin.to_numpy()
    issued = np.flatnonzero((origin >= 10) & ((origin-10)%30 == 0))
    scales = predict_models(models, data, issued)
    if not (np.isfinite(scales).all() and (np.diff(scales, axis=1) >= 0).all()):
        raise ValueError("Error scale must be finite and nondecreasing with time")
    np.savez_compressed(OUT/"issued_scales.npz", row=issued, gru=scales)
    return dict(fit_rows=len(fit), fit_envs=len(counts), seeds=list(SEEDS),
                issue_only_features="past 61 frames with stride 3; current QSM short-window kinematics and trailing cadence",
                loss="masked 3D distance to fixed QSM center divided by scale plus log scale")
