"""A Bayesian KalmanNet adaptation: sampled neural gains, total covariance and delayed online calibration."""
import json

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from motion_data import MotionData, SCALE, KIT, OUT, baseline, point_tables

SEED, EPOCHS, SAMPLES = 4107, 24, 16


class GainFilter(nn.Module):
    def __init__(self, dims):
        super().__init__()
        self.gain_features = nn.Sequential(nn.Linear(14, 32), nn.Tanh())
        self.gains = nn.Linear(32, 6)
        self.context = nn.Sequential(nn.Linear(dims + 6, 80), nn.SiLU(), nn.Linear(80, 48), nn.SiLU())
        self.forecast = nn.Linear(48, 4)
        self.cholesky = nn.Linear(48, 6)
        nn.init.zeros_(self.gains.weight)
        nn.init.zeros_(self.gains.bias)
        nn.init.zeros_(self.forecast.bias)

    def forward(self, x, h, dt, flags):
        n = len(h)
        # One posterior dropout draw per sequence, shared across all filter updates.
        draw = torch.empty(n, 32, device=x.device).bernoulli_(0.85) / 0.85
        pos, vel = h[:, 0], torch.zeros_like(h[:, 0])
        correction = torch.zeros_like(vel)
        for i in range(1, h.shape[1]):
            step = dt[:, i - 1:i].clamp(1e-4, 0.15)
            prior = pos + vel * step
            inov = h[:, i] - prior
            feat = torch.cat((inov, h[:, i] - h[:, i-1], correction, vel * step, flags[:, i:i+1], step * 30), 1)
            kpos, kvel = self.gains(self.gain_features(feat) * draw).chunk(2, dim=1)
            correction = kpos.sigmoid() * inov
            pos = prior + correction
            vel = vel + 0.7 * kvel.sigmoid() * inov / step
        features = self.context(torch.cat((x, pos, vel * .05), 1))
        forecast = self.forecast(features)
        tau = 0.04 + 0.65 * forecast[:, 3:4].sigmoid()
        mean = pos + vel * tau * (-torch.expm1(-1.0/tau)) + forecast[:, :3]
        raw = self.cholesky(features)
        diag = F.softplus(raw[:, :3]) + 0.035
        L = torch.diag_embed(diag)
        L[:, 1, 0], L[:, 2, 0], L[:, 2, 1] = raw[:, 3], raw[:, 4], raw[:, 5]
        return mean, L


def covariance_tables(methods, cuts):
    pooled, env = [], []
    for name, raw in methods.items():
        data = raw.copy()
        data["state"] = np.digitize(data.speed, cuts)
        valid = data[(data.split == "validation") & data.qualified]
        norm = valid.err / valid.scale
        global_q = np.quantile(norm, .95, method="higher")
        q = {}
        for key, group in valid.groupby(["state", "flag"]):
            q[key] = np.quantile(group.err / group.scale, .95, method="higher") if len(group) >= 30 else global_q
        data["base_radius"] = data.scale * np.array([q.get(key, global_q) for key in zip(data.state, data.flag)])
        # Update only when a previously predicted target has arrived and its raw flag is good.
        # The retrospective bone-quality screen is exclusively an evaluation mask here.
        radius = np.empty(len(data))
        for _, g in data.groupby(["sample", "hand"], sort=False):
            o, t = g.origin.to_numpy(), g.target.to_numpy()
            e, feedback, base = g.err.to_numpy(), g.feedback_ok.to_numpy(), g.base_radius.to_numpy()
            theta, j = 0., 0
            used = np.empty(len(g))
            for i in range(len(g)):
                while j < i and t[j] <= o[i]:
                    if feedback[j]:
                        theta += .01 * (float(e[j] > used[j]) - .05)
                    j += 1
                used[i] = base[i] * np.exp(theta)
            radius[g.index] = used
        data = data.assign(radius_cm=radius * 100, covered=data.err <= radius)
        scored = data[data.qualified]
        for group_keys, sink in ((["split"], pooled), (["split", "environment_id"], env)):
            s = scored.groupby(group_keys).agg(n=("err", "size"), coverage=("covered", "mean"),
                                               mean_radius_cm=("radius_cm", "mean")).reset_index()
            sink.append(s.assign(model=name))
    return pd.concat(pooled, ignore_index=True), pd.concat(env, ignore_index=True)


def fit_filter(d, a, dev, seed, frozen_gain=False):
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    train = np.flatnonzero((d.a["split"] == 0) & d.a["mask"][:, -1])
    model = GainFilter(a["x"].shape[1]).to(dev)
    if frozen_gain:
        # At their zero initialization the position and velocity gains are 0.5 and 0.35.
        model.gains.requires_grad_(False)
        model.gain_features.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=.0015, weight_decay=.00025)
    for epoch in range(EPOCHS):
        losses = []
        for ids in np.array_split(rng.choice(train, 10000, replace=False), 25):
            x = torch.as_tensor(ids, device=dev)
            mu, chol = model(a["x"][x], a["hist"][x], a["dt"][x], a["hist_flags"][x])
            diff = a["y"][x, -1] - mu
            white = torch.linalg.solve_triangular(chol, diff.unsqueeze(2), upper=False).squeeze(2)
            nll = .5 * white.square().sum(1) + chol.diagonal(dim1=1, dim2=2).log().sum(1)
            loss = (nll + 1.25 * diff.norm(dim=1)).mean()
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 4.0)
            opt.step()
            losses.append(loss.item())
        if (epoch + 1) % 6 == 0:
            print("gain epoch", epoch + 1, "loss", np.mean(losses), flush=True)
    return model.eval()


def evaluate_filter(d, a, dev, model, seed):
    mu_all = np.zeros((len(d.a["x"]), 3))
    scale_all = np.ones(len(mu_all))
    torch.manual_seed(seed + 1)
    with torch.no_grad():
        held = np.flatnonzero(d.a["split"] != 0)
        for ids in np.array_split(held, int(np.ceil(len(held)/1024))):
            ix = torch.as_tensor(ids, device=dev)
            draws = [model(a["x"][ix], a["hist"][ix], a["dt"][ix], a["hist_flags"][ix]) for _ in range(SAMPLES)]
            means = torch.stack([x[0] for x in draws])
            mean = means.mean(0)
            centered = means - mean
            chol = torch.stack([x[1] for x in draws])
            cov = (chol @ chol.transpose(-1, -2) + centered.unsqueeze(-1) * centered.unsqueeze(-2)).mean(0)
            assert torch.linalg.cholesky_ex(cov).info.max().item() == 0
            mu_all[ids] = mean.cpu()
            scale_all[ids] = SCALE * cov.diagonal(dim1=1, dim2=2).sum(-1).sqrt().cpu().numpy()
    return d.error_rows(mu_all, scale_all)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(3)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    d = MotionData()
    a = {key: torch.as_tensor(d.a[key], device=dev) for key in ("x", "hist", "dt", "hist_flags", "y")}
    model = fit_filter(d, a, dev, SEED)
    torch.save({"state": model.state_dict(), "seed": SEED, "epochs": EPOCHS}, OUT / "bayesian_gain.pt")
    base = baseline(d)
    new = evaluate_filter(d, a, dev, model, SEED)
    point, env = point_tables({"boosting": base, "bayesian_gain": new})
    point.to_csv(OUT / "gain_point.csv", index=False)
    env.to_csv(OUT / "gain_point_by_environment.csv", index=False)
    cuts = np.quantile(np.concatenate([r["speed5"][np.isfinite(r["speed5"])] for r in d.tracks if r["split"] == "train"]), [1/3, 2/3])
    bands, env_bands = covariance_tables({"boosting_same_calibration": base, "bayesian_gain": new}, cuts)
    old = pd.read_csv(KIT / "results/q1_real/final7_aci_pooled.csv").query("method=='boost_cells' and gamma==0.01 and horizon==30").iloc[0]
    bands.loc[len(bands)] = ["test", old.n, old.coverage, old.mean_radius_mm / 10, "boosting_published"]
    bands.to_csv(OUT / "prediction_intervals.csv", index=False)
    env_bands.to_csv(OUT / "intervals_by_environment.csv", index=False)
    ref = point.query("model=='boosting' and split=='test'").iloc[0]
    assert ref["size"] == old.n and abs(ref.macro_cm - 19.03137) < .001
    print(point.to_string(index=False), bands.to_string(index=False), sep="\n", flush=True)


if __name__ == "__main__":
    main()
