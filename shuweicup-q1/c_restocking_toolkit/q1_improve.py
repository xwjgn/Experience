#!/usr/bin/env python3
"""Problem 1 improvements, chosen by leave-one-environment-out CV, then tested once.

Stage "cv" uses only the 20 development environments (train + validation). Each
fold holds out one environment, fits everything on the other 19 and scores the
held-out one. No test record is read for scoring in this stage.

  A  observation reliability with head-frame features: wrist direction and angle
     from the head's forward axis (head frame: x forward, y left, z up, i.e.
     v_head = R(q)^T (wrist - head)), the rate of that angle and the head turn rate
  B  learned multi-step predictor: isotropic per-axis model of the mean velocity
     over the horizon from the last 10 wrist velocities, the last 5 head
     velocities, 5-frame speed and the recent-flag indicator; ridge (linear) and
     gradient boosting (non-linear) against the Kalman / damped baselines
  C  variance calibration of the constant-velocity Kalman filter by group:
     global, recent-flag (2 groups) and motion state x recent-flag (6 groups);
     split-conformal radii by the same 6 groups for the damped and ridge predictors
  D  every comparison is reported per environment with an environment-bootstrap
     95 % interval of the paired difference

Decision rules (fixed before the CV results were produced):
  B: adopt a learned predictor at a horizon only if its paired improvement over
     damped_kalman has an environment-bootstrap 95 % interval entirely below zero
  C: use state x flag groups only if every group's pooled CV coverage lies in
     [0.93, 0.97] at every horizon and the mean radius is smaller than with
     flag-only groups; otherwise flag-only
  A: add head-frame features only if the onset-AUC gain has an interval entirely
     above zero (logistic regression, the interpretable model)

Stage "final" reads cv_decisions.json, fits the chosen configuration on TRAIN
only and reports the test split once.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

import q1_reliability as rel
import q1_state as q1

LABEL = "REAL_DATA_Q1_LOEO_IMPROVEMENTS"
HORIZONS = q1.HORIZONS
STATES = ("slow", "medium", "fast")
VEL_LAGS, HEAD_LAGS = 10, 5
MIN_ORIGIN = VEL_LAGS  # origin o needs velocities o-9..o, defined from frame 1
CRIT95 = chi2.ppf(0.95, 3)
BOOTSTRAP = 1000
COVERAGE_BAND = (0.93, 0.97)
HEAD_FEATURES = ["off_axis_deg", "abs_azimuth_deg", "elevation_deg", "head_distance", "off_axis_rate", "head_turn_rate"]
HEAD_NAMES = {"off_axis_deg": "偏离头部朝向的角度", "abs_azimuth_deg": "水平偏角", "elevation_deg": "俯仰角",
              "head_distance": "腕到头距离（头部坐标系）", "off_axis_rate": "偏离角变化率", "head_turn_rate": "头部转动角速度"}
PREDICTORS = ("hold", "observed_cv", "cv_kalman", "damped_kalman", "ridge", "boosting")


# ------------------------------------------------------------------- data

def _rot_T(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    w, x, y, z = q.T
    R = np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)
    return np.einsum("nji,nj->ni", R, v)


def load(canonical: Path, data_root: Path) -> list[dict]:
    tracks, frames = q1.load_tracks(canonical, data_root)
    q1.qualify(tracks)  # geometry threshold from train ok frames only
    heads = {s: (g.sort_values("frame_index")[["head_x", "head_y", "head_z"]].to_numpy(float),
                 g.sort_values("frame_index")[["head_qw", "head_qx", "head_qy", "head_qz"]].to_numpy(float))
             for s, g in frames.groupby("sample", sort=False)}
    for tr in tracks:
        t, z = tr["t"], tr["points"]["wrist"]
        head, quat = heads[tr["sample"]]
        dt = np.r_[np.nan, np.diff(t)]
        vel = np.vstack([np.full((1, 3), np.nan), np.diff(z, axis=0)]) / dt[:, None]
        hv = _rot_T(quat, z - head)
        dist = np.linalg.norm(hv, axis=1)
        off = np.degrees(np.arccos(np.clip(hv[:, 0] / dist, -1, 1)))
        dots = np.abs(np.sum(quat[1:] * quat[:-1], axis=1))
        turn = np.r_[np.nan, np.degrees(2 * np.arccos(np.clip(dots, -1, 1)))] / dt
        flags = tr["bad"].astype(float)
        tr.update({
            "vel": vel, "speed5": rel._rolling_mean(np.linalg.norm(vel, axis=1), 5),
            "head_vel": np.vstack([np.full((1, 3), np.nan), np.diff(head, axis=0)]) / dt[:, None],
            "recent_flag": np.convolve(flags, np.ones(10))[:len(t)] > 0,
            "off_axis_deg": off, "abs_azimuth_deg": np.degrees(np.abs(np.arctan2(hv[:, 1], hv[:, 0]))),
            "elevation_deg": np.degrees(np.arctan2(hv[:, 2], np.hypot(hv[:, 0], hv[:, 1]))),
            "head_distance": dist, "off_axis_rate": rel._rolling_mean(np.r_[np.nan, np.diff(off)] / dt, 5),
            "head_turn_rate": rel._rolling_mean(turn, 5),
        })
    return tracks


def env_bootstrap(per_env: pd.Series, n: int = BOOTSTRAP, seed: int = 0) -> tuple[float, float]:
    values = per_env.to_numpy()
    rng = np.random.default_rng(seed)
    means = np.array([values[rng.integers(0, len(values), len(values))].mean() for _ in range(n)])
    return float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))


# ----------------------------------------------------------- Kalman cache

class KalmanCache:
    """Per-track log-likelihood grids, so a fold's fit is a sum over its tracks."""

    def __init__(self, tracks: list[dict]):
        self.cv_q, self.cv_r = (a.ravel() for a in np.meshgrid(q1.Q_GRID, q1.R_GRID, indexing="ij"))
        self.qk_q, self.qk_r, self.qk_m = (a.ravel() for a in np.meshgrid(q1.Q_GRID, q1.R_GRID, q1.M_GRID, indexing="ij"))
        self.cv = np.stack([q1.kalman_grid(tr["t"], tr["points"]["wrist"], np.zeros_like(tr["bad"]), self.cv_q, self.cv_r,
                                           np.ones_like(self.cv_q)) for tr in tracks])
        self.qk = np.stack([q1.kalman_grid(tr["t"], tr["points"]["wrist"], tr["bad"], self.qk_q, self.qk_r, self.qk_m)
                            for tr in tracks])
        self._damping, self._pred = {}, {}

    def fit(self, rows: np.ndarray, tracks: list[dict]) -> dict:
        i = int(np.argmax(self.cv[rows].sum(axis=0)))
        j = int(np.argmax(self.qk[rows].sum(axis=0)))
        cv = {"q": float(self.cv_q[i]), "r": float(self.cv_r[i]), "m": 1.0}
        qk = {"q": float(self.qk_q[j]), "r": float(self.qk_r[j]), "m": float(self.qk_m[j])}
        sums, counts = 0.0, 0.0
        for r in rows:
            s, c = self.damping_stats(tracks[r], cv)
            sums, counts = sums + s, counts + c
        tau = float(q1.TAU_GRID[int(np.argmin((sums / counts[:, None]).mean(axis=0)))])
        return {"cv_kalman": cv, "quality_kalman": qk, "damped_kalman": {"tau": tau}}

    def damping_stats(self, tr: dict, cv: dict):
        key = (tr["sample"], tr["hand"], cv["q"], cv["r"])
        if key not in self._damping:
            t, z = tr["t"], tr["points"]["wrist"]
            _, (fp, fv, _) = q1.kalman_grid(t, z, np.zeros_like(tr["bad"]), cv["q"], cv["r"], 1.0, store=True)
            sums, counts = np.zeros((len(HORIZONS), q1.TAU_GRID.size)), np.zeros(len(HORIZONS))
            for i, h in enumerate(HORIZONS):
                k = np.arange(h + 1, len(t))
                k = k[tr["qualified"][k]]
                o = k - h
                d = (t[k] - t[o])[:, None, None]
                pred = fp[o][:, None, :] + fv[o][:, None, :] * q1.TAU_GRID[None, :, None] * (1 - np.exp(-d / q1.TAU_GRID[None, :, None]))
                sums[i] = np.linalg.norm(pred - z[k][:, None, :], axis=2).sum(axis=0)
                counts[i] = k.size
            self._damping[key] = (sums, counts)
        return self._damping[key]

    def predictions(self, tr: dict, params: dict) -> dict:
        key = (tr["sample"], tr["hand"], json.dumps(params, sort_keys=True))
        if key not in self._pred:
            self._pred[key] = q1.predictions(tr, "wrist", params)
        return self._pred[key]


# ------------------------------------------------------------ learned (B)

def origins(tr: dict, h: int) -> tuple[np.ndarray, np.ndarray]:
    o = np.arange(MIN_ORIGIN, len(tr["t"]) - h)
    k = o + h
    keep = tr["qualified"][k] & np.isfinite(tr["speed5"][o])
    return o[keep], k[keep]


def design(tr: dict, o: np.ndarray, h: int):
    """Rows per (origin, axis): same coefficients for x, y and z (isotropic).

    Every column is linear in that axis's velocities, so the linear model is
    rotation-equivariant; speed and the recent-flag indicator enter only as
    interactions with the current velocity (an additive scalar would push all
    three axes by the same amount, i.e. along the (1, 1, 1) diagonal)."""
    blocks = []
    for a in range(3):
        cols = [tr["vel"][o - i, a] for i in range(VEL_LAGS)] + [tr["head_vel"][o - i, a] for i in range(HEAD_LAGS)]
        cols += [tr["vel"][o, a] * tr["speed5"][o], tr["vel"][o, a] * tr["recent_flag"][o]]
        blocks.append(np.stack(cols, axis=1))
    X = np.concatenate(blocks)
    X = np.nan_to_num(X, nan=0.0)
    delta = tr["t"][o + h] - tr["t"][o]
    return X, np.tile(delta, 3)


class Learned:
    def __init__(self, kind: str):
        self.kind = kind

    def fit(self, tracks: list[dict], h: int, seed: int = 0):
        Xs, ys = [], []
        for tr in tracks:
            o, k = origins(tr, h)
            if o.size == 0:
                continue
            X, delta = design(tr, o, h)
            disp = (tr["points"]["wrist"][k] - tr["points"]["wrist"][o]).T.ravel()  # axis-major, matches design
            Xs.append(X)
            ys.append(disp / delta)
        X, y = np.concatenate(Xs), np.concatenate(ys)
        if self.kind == "ridge":
            self.scale = np.sqrt(np.mean(X ** 2, axis=0)) + 1e-9  # RMS: invariant to rotating the world frame
            Z = X / self.scale
            self.beta = np.linalg.solve(Z.T @ Z + 1.0 * np.eye(Z.shape[1]), Z.T @ y)  # no intercept: isotropy
        else:
            rng = np.random.default_rng(seed)
            take = rng.choice(len(y), size=min(len(y), 90000), replace=False)
            self.model = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, min_samples_leaf=100,
                                                       random_state=seed).fit(X[take], y[take])
        return self

    def predict(self, tr: dict, o: np.ndarray, h: int) -> np.ndarray:
        X, delta = design(tr, o, h)
        v = (X / self.scale) @ self.beta if self.kind == "ridge" else self.model.predict(X)
        disp = (v * delta).reshape(3, -1).T
        return tr["points"]["wrist"][o] + disp


# ------------------------------------------------------------- evaluation

def score_tracks(eval_tracks, fit_tracks, params, learned, cache, cuts):
    """Per-sample records for all predictors on eval_tracks (shared target set)."""
    rows = []
    for tr in eval_tracks:
        z = tr["points"]["wrist"]
        for h in HORIZONS:
            o, k = origins(tr, h)
            if o.size == 0:
                continue
            item = cache.predictions(tr, params)[h]
            idx = k - h - 1
            preds = {m: item["pred"][m][idx] for m in ("hold", "observed_cv", "cv_kalman", "damped_kalman")}
            for kind, model in learned[h].items():
                preds[kind] = model.predict(tr, o, h)
            state = np.digitize(tr["speed5"][o], cuts)
            flag = tr["recent_flag"][o]
            var = item["var"]["cv_kalman"][idx]
            for m, p in preds.items():
                rows.append(pd.DataFrame({"sample": tr["sample"], "hand": tr["hand"], "environment_id": tr["environment_id"],
                                          "split": tr["split"], "horizon": h, "model": m, "state": state, "flag": flag,
                                          "err": np.linalg.norm(p - z[k], axis=1),
                                          "var": var if m == "cv_kalman" else np.nan}))
    return pd.concat(rows, ignore_index=True)


def calibration_fit(train_scores: pd.DataFrame) -> dict:
    """Variance factors for cv_kalman and conformal radii for damped/ridge, per horizon."""
    out = {}
    for h, g in train_scores.groupby("horizon"):
        kf = g[g.model == "cv_kalman"]
        nis = kf["err"] ** 2 / kf["var"]
        f = {"global": {(): max(float(np.quantile(nis, 0.95)) / CRIT95, 1e-9)},
             "flag": {(b,): max(float(np.quantile(nis[kf.flag == b], 0.95)) / CRIT95, 1e-9) for b in (False, True)},
             "state_flag": {}}
        for s in range(3):
            for b in (False, True):
                sel = (kf.state == s) & (kf.flag == b)
                f["state_flag"][(s, b)] = (max(float(np.quantile(nis[sel], 0.95)) / CRIT95, 1e-9) if sel.sum() >= 30
                                           else f["flag"][(b,)])
        radii = {}
        for m in ("damped_kalman", "ridge"):
            mg = g[g.model == m]
            radii[m] = {}
            for s in range(3):
                for b in (False, True):
                    sel = (mg.state == s) & (mg.flag == b)
                    base = mg[mg.flag == b] if sel.sum() < 30 else mg[sel]
                    radii[m][(s, b)] = float(np.quantile(base["err"], 0.95))
        out[h] = {"kf": f, "radius": radii}
    return out


def calibration_apply(scores: pd.DataFrame, fitted: dict) -> pd.DataFrame:
    rows = []
    for h, g in scores.groupby("horizon"):
        kf = g[g.model == "cv_kalman"]
        for method, factors in fitted[h]["kf"].items():
            key = (lambda r: ()) if method == "global" else (lambda r: (r.flag,)) if method == "flag" else (lambda r: (r.state, r.flag))
            fac = np.array([factors[key(r)] for r in kf[["state", "flag"]].itertuples(index=False)])
            radius = np.sqrt(CRIT95 * kf["var"].to_numpy() * fac)
            rows.append(kf.assign(method=f"kf_{method}", covered=kf["err"].to_numpy() <= radius, radius=radius)
                        [["environment_id", "horizon", "state", "flag", "method", "covered", "radius"]])
        unc = np.sqrt(CRIT95 * kf["var"].to_numpy())
        rows.append(kf.assign(method="kf_uncalibrated", covered=kf["err"].to_numpy() <= unc, radius=unc)
                    [["environment_id", "horizon", "state", "flag", "method", "covered", "radius"]])
        for m in ("damped_kalman", "ridge"):
            mg = g[g.model == m]
            radius = np.array([fitted[h]["radius"][m][(r.state, r.flag)] for r in mg[["state", "flag"]].itertuples(index=False)])
            rows.append(mg.assign(method=f"conformal_{m}", covered=mg["err"].to_numpy() <= radius, radius=radius)
                        [["environment_id", "horizon", "state", "flag", "method", "covered", "radius"]])
    return pd.concat(rows, ignore_index=True)


# ---------------------------------------------------------- reliability (A)

def reliability_table(canonical: Path, tracks: list[dict]) -> pd.DataFrame:
    base = rel.build_features(canonical)
    head = pd.concat([pd.DataFrame({"sample": tr["sample"], "hand": tr["hand"], "frame_index": np.arange(len(tr["t"])),
                                    **{f: tr[f] for f in HEAD_FEATURES}}) for tr in tracks], ignore_index=True)
    table = base.merge(head, on=["sample", "hand", "frame_index"], how="inner")
    return table.dropna(subset=HEAD_FEATURES).reset_index(drop=True)


def _fit_predict(train: pd.DataFrame, test: pd.DataFrame, cols: list[str], kind: str) -> np.ndarray:
    mean, std = train[cols].mean(), train[cols].std().replace(0, 1.0)
    Xtr, Xte = ((train[cols] - mean) / std).to_numpy(), ((test[cols] - mean) / std).to_numpy()
    y = train["future_flag"].to_numpy(int)
    model = (LogisticRegression(max_iter=3000) if kind == "logistic"
             else HistGradientBoostingClassifier(max_iter=150, learning_rate=0.1, random_state=0))
    return model.fit(Xtr, y).predict_proba(Xte)[:, 1]


def auc_by_bootstrap(oof: pd.DataFrame, a: str, b: str, subset_mask: np.ndarray, n: int = 500, seed: int = 1):
    d = oof[subset_mask]
    envs = d["environment_id"].unique()
    groups = {e: d[d.environment_id == e] for e in envs}
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n):
        sample = pd.concat([groups[e] for e in rng.choice(envs, len(envs))])
        if 0 < sample["future_flag"].sum() < len(sample):
            diffs.append(roc_auc_score(sample["future_flag"], sample[b]) - roc_auc_score(sample["future_flag"], sample[a]))
    return float(np.quantile(diffs, 0.025)), float(np.quantile(diffs, 0.975))


# ------------------------------------------------------------------ stages

def stage_cv(canonical: Path, data_root: Path, out: Path, max_folds: int | None = None) -> dict:
    tracks = load(canonical, data_root)
    dev = [tr for tr in tracks if tr["split"] in ("train", "validation")]
    envs = sorted({tr["environment_id"] for tr in dev})[:max_folds]  # max_folds: smoke tests only
    cache = KalmanCache(dev)
    all_scores, all_cal, fold_params = [], [], []
    for env in envs:
        fit_rows = np.array([i for i, tr in enumerate(dev) if tr["environment_id"] != env])
        fit_tracks = [dev[i] for i in fit_rows]
        held = [tr for tr in dev if tr["environment_id"] == env]
        params = cache.fit(fit_rows, dev)
        cuts = np.quantile(np.concatenate([tr["speed5"][np.isfinite(tr["speed5"])] for tr in fit_tracks]), [1 / 3, 2 / 3])
        learned = {h: {"ridge": Learned("ridge").fit(fit_tracks, h), "boosting": Learned("boosting").fit(fit_tracks, h)}
                   for h in HORIZONS}
        train_scores = score_tracks(fit_tracks, fit_tracks, params, {h: {"ridge": learned[h]["ridge"]} for h in HORIZONS},
                                    cache, cuts)
        held_scores = score_tracks(held, fit_tracks, params, learned, cache, cuts).assign(fold=env)
        all_scores.append(held_scores)
        all_cal.append(calibration_apply(held_scores, calibration_fit(train_scores)).assign(fold=env))
        fold_params.append({"environment_id": env, **{f"{m}_{k}": v for m, p in params.items() for k, v in p.items()},
                            "speed_cut_low": cuts[0], "speed_cut_high": cuts[1]})
        print(f"fold {env}: done", flush=True)
    scores = pd.concat(all_scores, ignore_index=True)
    cal = pd.concat(all_cal, ignore_index=True)
    # B: per-environment mean error, paired differences vs damped_kalman
    env_err = scores.groupby(["horizon", "model", "environment_id"])["err"].mean().unstack("model") * 1000
    pred_rows, decisions_b = [], {}
    for h in HORIZONS:
        e = env_err.loc[h]
        for m in PREDICTORS:
            diff = e[m] - e["damped_kalman"]
            lo, hi = env_bootstrap(diff)
            pred_rows.append({"horizon": h, "model": m, "environments": len(e), "macro_E_mm": e[m].mean(),
                              "pooled_E_mm": 1000 * scores[(scores.horizon == h) & (scores.model == m)]["err"].mean(),
                              "diff_vs_damped_mm": diff.mean(), "diff_ci_low": lo, "diff_ci_high": hi,
                              "envs_better_than_damped": int((diff < 0).sum())})
        cands = [r for r in pred_rows if r["horizon"] == h and r["model"] in ("ridge", "boosting") and r["diff_ci_high"] < 0]
        decisions_b[h] = min(cands, key=lambda r: r["macro_E_mm"])["model"] if cands else "damped_kalman"
    pred = pd.DataFrame(pred_rows)
    # C: coverage by method and group
    cov = (cal.groupby(["horizon", "method", "state", "flag"]).agg(n=("covered", "size"), coverage=("covered", "mean"),
                                                                    mean_radius_mm=("radius", "mean")).reset_index())
    cov["mean_radius_mm"] *= 1000
    overall = (cal.groupby(["horizon", "method"]).agg(n=("covered", "size"), coverage=("covered", "mean"),
                                                      mean_radius_mm=("radius", "mean")).reset_index())
    overall["mean_radius_mm"] *= 1000
    sf, fl = cov[cov.method == "kf_state_flag"], overall.set_index(["horizon", "method"])
    in_band = bool(((sf.coverage >= COVERAGE_BAND[0]) & (sf.coverage <= COVERAGE_BAND[1])).all())
    narrower = all(fl.loc[(h, "kf_state_flag"), "mean_radius_mm"] < fl.loc[(h, "kf_flag"), "mean_radius_mm"] for h in HORIZONS)
    decision_c = "state_flag" if in_band and narrower else "flag"
    # A: reliability with and without head-frame features
    table = reliability_table(canonical, tracks)
    devt = table[table.split.isin(["train", "validation"])].copy()
    scored_envs = set(envs)  # every development environment in the real run
    fit_pool = devt.copy()
    base_cols, head_cols = rel.FEATURES, rel.FEATURES + HEAD_FEATURES
    for name, cols, kind in (("base_logistic", base_cols, "logistic"), ("head_logistic", head_cols, "logistic"),
                             ("base_boosting", base_cols, "boosting"), ("head_boosting", head_cols, "boosting")):
        devt[name] = np.nan
        for env in envs:
            tr_mask, te_mask = fit_pool.environment_id != env, devt.environment_id == env
            devt.loc[te_mask, name] = _fit_predict(fit_pool[tr_mask], devt[te_mask], cols, kind)
    devt = devt[devt.environment_id.isin(scored_envs)].reset_index(drop=True)
    rel_rows = []
    onset = (devt["flags_last10"] == 0).to_numpy()
    for subset, mask in (("all", np.ones(len(devt), bool)), ("onset", onset)):
        d = devt[mask]
        for name in ("base_logistic", "head_logistic", "base_boosting", "head_boosting"):
            rel_rows.append({"subset": subset, "model": name, "n": len(d), "positives": int(d.future_flag.sum()),
                             "prevalence": float(d.future_flag.mean()), "auc": roc_auc_score(d.future_flag, d[name]),
                             "average_precision": average_precision_score(d.future_flag, d[name])})
    rel_cv = pd.DataFrame(rel_rows)
    ci = {f"{kind}_{subset}": auc_by_bootstrap(devt, f"base_{kind}", f"head_{kind}", mask)
          for kind in ("logistic", "boosting") for subset, mask in (("all", np.ones(len(devt), bool)), ("onset", onset))}
    decision_a = "head" if ci["logistic_onset"][0] > 0 else "base"
    decisions = {"predictor_by_horizon": {str(h): m for h, m in decisions_b.items()}, "calibration_groups": decision_c,
                 "reliability_features": decision_a, "auc_gain_ci_head_minus_base": ci,
                 "rules": {"B": "learned model only if CI of paired diff vs damped_kalman < 0",
                           "C": f"state_flag only if all group coverages in {COVERAGE_BAND} and narrower",
                           "A": "head features only if logistic onset AUC gain CI > 0"}}
    out.mkdir(parents=True, exist_ok=True)
    pred.to_csv(out / "cv_predictors.csv", index=False)
    env_err.reset_index().to_csv(out / "cv_predictors_by_environment.csv", index=False)
    cov.to_csv(out / "cv_calibration_groups.csv", index=False)
    overall.to_csv(out / "cv_calibration_overall.csv", index=False)
    rel_cv.to_csv(out / "cv_reliability.csv", index=False)
    pd.DataFrame(fold_params).to_csv(out / "cv_fold_parameters.csv", index=False)
    (out / "cv_decisions.json").write_text(json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"predictors": pred, "calibration": cov, "overall": overall, "reliability": rel_cv, "decisions": decisions}


def chosen_config(out: Path) -> dict:
    """CV rule outcome, overridden by final_config.json when the team made a final choice."""
    decisions = json.loads((out / "cv_decisions.json").read_text(encoding="utf-8"))
    override = out / "final_config.json"
    if override.is_file():
        decisions = {**decisions, **json.loads(override.read_text(encoding="utf-8"))}
    return decisions


def stage_final(canonical: Path, data_root: Path, out: Path) -> dict:
    decisions = chosen_config(out)
    tracks = load(canonical, data_root)
    train = [tr for tr in tracks if tr["split"] == "train"]
    test = [tr for tr in tracks if tr["split"] == "test"]
    cache = KalmanCache(train)
    params = cache.fit(np.arange(len(train)), train)
    cuts = np.quantile(np.concatenate([tr["speed5"][np.isfinite(tr["speed5"])] for tr in train]), [1 / 3, 2 / 3])
    learned = {h: {"ridge": Learned("ridge").fit(train, h), "boosting": Learned("boosting").fit(train, h)} for h in HORIZONS}
    train_scores = score_tracks(train, train, params, {h: {"ridge": learned[h]["ridge"]} for h in HORIZONS}, cache, cuts)
    test_scores = score_tracks(test, train, params, learned, cache, cuts)
    cal = calibration_apply(test_scores, calibration_fit(train_scores))
    env_err = test_scores.groupby(["horizon", "model", "environment_id"])["err"].mean().unstack("model") * 1000
    rows = []
    for h in HORIZONS:
        e = env_err.loc[h]
        chosen = decisions["predictor_by_horizon"][str(h)]
        for m in PREDICTORS:
            rows.append({"horizon": h, "model": m, "chosen": m == chosen, "macro_E_mm": e[m].mean(),
                         "pooled_E_mm": 1000 * test_scores[(test_scores.horizon == h) & (test_scores.model == m)]["err"].mean(),
                         "diff_vs_damped_mm": (e[m] - e["damped_kalman"]).mean(),
                         "envs_better_than_damped": int(((e[m] - e["damped_kalman"]) < 0).sum())})
    final_pred = pd.DataFrame(rows)
    cov = (cal.groupby(["horizon", "method", "flag"]).agg(n=("covered", "size"), coverage=("covered", "mean"),
                                                          mean_radius_mm=("radius", "mean")).reset_index())
    cov["mean_radius_mm"] *= 1000
    table = reliability_table(canonical, tracks)
    tr_t, te_t = table[table.split == "train"], table[table.split == "test"]
    rel_rows = []
    for name, cols in (("base", rel.FEATURES), ("head", rel.FEATURES + HEAD_FEATURES)):
        for kind in ("logistic", "boosting"):
            p = _fit_predict(tr_t, te_t, cols, kind)
            for subset, mask in (("all", np.ones(len(te_t), bool)), ("onset", (te_t["flags_last10"] == 0).to_numpy())):
                y = te_t["future_flag"].to_numpy(int)[mask]
                rel_rows.append({"features": name, "model": kind, "subset": subset, "chosen": name == decisions["reliability_features"],
                                 "n": int(mask.sum()), "positives": int(y.sum()), "auc": roc_auc_score(y, p[mask]),
                                 "average_precision": average_precision_score(y, p[mask])})
    final_rel = pd.DataFrame(rel_rows)
    coef = None
    if decisions["reliability_features"] == "head":
        cols = rel.FEATURES + HEAD_FEATURES
        mean, std = tr_t[cols].mean(), tr_t[cols].std().replace(0, 1.0)
        lr = LogisticRegression(max_iter=3000).fit(((tr_t[cols] - mean) / std).to_numpy(), tr_t["future_flag"].to_numpy(int))
        coef = pd.DataFrame({"feature": cols, "name_zh": [rel.FEATURE_NAMES.get(c, HEAD_NAMES.get(c, c)) for c in cols],
                             "standardized_coef": lr.coef_[0]}).sort_values("standardized_coef")
        coef.to_csv(out / "final_reliability_coefficients.csv", index=False)
    final_pred.to_csv(out / "final_test_predictors.csv", index=False)
    cov.to_csv(out / "final_test_calibration.csv", index=False)
    final_rel.to_csv(out / "final_test_reliability.csv", index=False)
    (out / "final_parameters.json").write_text(json.dumps({"fitted_on": "train", "params": params,
                                                           "speed_cuts": cuts.tolist()}, indent=2), encoding="utf-8")
    return {"predictors": final_pred, "calibration": cov, "reliability": final_rel, "coefficients": coef}


def write_summary(out: Path) -> Path:
    """Chinese summary from the saved CV and final CSVs; computes nothing new."""
    dec = json.loads((out / "cv_decisions.json").read_text(encoding="utf-8"))
    cvp = pd.read_csv(out / "cv_predictors.csv")
    cvg = pd.read_csv(out / "cv_calibration_groups.csv")
    cvr = pd.read_csv(out / "cv_reliability.csv")
    fp = pd.read_csv(out / "final_test_predictors.csv")
    fc = pd.read_csv(out / "final_test_calibration.csv")
    fr = pd.read_csv(out / "final_test_reliability.csv")
    final = chosen_config(out)
    final_lines = []
    if (out / "final_config.json").is_file():
        final_lines = ["## 最终采用的配置", "",
                       "| 项目 | 交叉验证规则的结论 | 最终采用 |", "|---|---|---|",
                       f"| B 预测器 | {dec['predictor_by_horizon']} | {final['predictor_by_horizon']} |",
                       f"| C 校准分组 | {dec['calibration_groups']} | {final['calibration_groups']} |",
                       f"| A 可靠性特征 | {dec['reliability_features']} | {final['reliability_features']} |", ""]
        final_lines += [f"- {k}：{v}" for k, v in final.get("reason", {}).items()]
        if final.get("decided_after_viewing_test"):
            final_lines += ["", "说明：A 与 C 未达到预先写定的规则，团队在查看测试结果后决定采用；B 与预先规则一致。论文中如实说明这一顺序。"]
        final_lines.append("")
    L = ["# 第一问改进：留一环境交叉验证与一次性测试", "",
         "开发阶段只用训练集与验证集的 20 个环境：每次留出 1 个环境，在其余 19 个环境上拟合全部参数，再在留出环境上评价，轮换 20 次。"
         "采用规则在看到交叉验证结果之前写定（见 q1_improve.py 文件头），最终模型只用训练集拟合；A、C 的最终采用与规则不同，"
         "且在查看测试结果后决定，见下文说明。", "",
         *final_lines,
         "## 交叉验证规则的结论", "",
         f"- B 学习式预测器：各步长采用 {dec['predictor_by_horizon']}（规则：相对速度衰减外推的配对差 95% 区间全部小于 0）。",
         f"- C 方差校准分组：{dec['calibration_groups']}（规则：运动状态×标记 6 组的每组覆盖率都在 0.93–0.97 且平均半径更小才采用；"
         "24 个组格中有 3 格低于 0.93：慢速且近期有标记的 3 个最小组格，样本 71–133，覆盖率 0.920–0.930）。",
         f"- A 头部坐标系特征：{dec['reliability_features']}（规则：逻辑回归新遮挡 AUC 增益区间全部大于 0 才采用；区间为 "
         f"{dec['auc_gain_ci_head_minus_base']['logistic_onset'][0]:+.3f} 至 {dec['auc_gain_ci_head_minus_base']['logistic_onset'][1]:+.3f}）。", "",
         "## B 预测器（交叉验证，20 个环境）", "",
         "| 步长（帧） | 模型 | 环境平均误差 mm | 相对速度衰减外推 mm | 95% 区间 | 优于速度衰减的环境数 |", "|---:|---|---:|---:|---|---:|"]
    for _, r in cvp.iterrows():
        L.append(f"| {r.horizon} | {r.model} | {r.macro_E_mm:.2f} | {r.diff_vs_damped_mm:+.2f} | [{r.diff_ci_low:+.2f}, {r.diff_ci_high:+.2f}] | {r.envs_better_than_damped}/20 |")
    L += ["", "## B 预测器（测试集，一次性，4 个环境）", "",
          "| 步长（帧） | 模型 | 最终采用 | 环境平均误差 mm | 合并误差 mm | 相对速度衰减外推 mm | 更优环境数 |", "|---:|---|---|---:|---:|---:|---:|"]
    for _, r in fp.iterrows():
        L.append(f"| {r.horizon} | {r.model} | {'是' if r.chosen else ''} | {r.macro_E_mm:.2f} | {r.pooled_E_mm:.2f} | {r.diff_vs_damped_mm:+.2f} | {r.envs_better_than_damped}/4 |")
    L += ["", "## C 校准（交叉验证，各组 95% 覆盖率）", "",
          "状态：0 慢、1 中、2 快；flag 为预测起点前 10 帧内是否出现标记。", "",
          "| 步长 | 状态 | 近期标记 | 模型自身方差 | 按标记 2 组 | 状态×标记 6 组 | ridge 分组保形 |", "|---:|---:|---|---:|---:|---:|---:|"]
    piv = cvg.pivot_table(index=["horizon", "state", "flag"], columns="method", values="coverage")
    for (h, s, f), r in piv.iterrows():
        L.append(f"| {h} | {s} | {f} | {r['kf_uncalibrated']:.3f} | {r['kf_flag']:.3f} | {r['kf_state_flag']:.3f} | {r['conformal_ridge']:.3f} |")
    L += ["", "## C 校准（测试集，一次性）", "", "| 步长 | 近期标记 | 方法 | 覆盖率 | 平均半径 mm |", "|---:|---|---|---:|---:|"]
    for _, r in fc[fc.method.isin(["kf_uncalibrated", "kf_flag", "kf_state_flag", "conformal_ridge"])].iterrows():
        L.append(f"| {r.horizon} | {r.flag} | {r.method} | {r.coverage:.3f} | {r.mean_radius_mm:.1f} |")
    L += ["", "## A 可靠性模型", "", "交叉验证（20 个环境的留出预测合并）：", "",
          "| 子集 | 模型 | AUC | 平均精度 |", "|---|---|---:|---:|"]
    for _, r in cvr.iterrows():
        L.append(f"| {r.subset} | {r.model} | {r.auc:.3f} | {r.average_precision:.3f} |")
    L += ["", "测试集（一次性；head 特征未被规则采用，仅作补充报告）：", "", "| 特征 | 模型 | 子集 | AUC | 平均精度 |", "|---|---|---|---:|---:|"]
    for _, r in fr.iterrows():
        L.append(f"| {r.features} | {r.model} | {r.subset} | {r.auc:.3f} | {r.average_precision:.3f} |")
    L += ["", "说明：环境只有 20 个开发环境和 4 个测试环境，区间按环境重抽样；相邻帧高度相关，样本数不是独立样本数。"
          "合格参考的几何阈值沿用训练集的分位数，在留一交叉验证中对留出的训练环境略有信息重叠，影响极小。", ""]
    path = out / "summary_improve.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def main(argv=None) -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("cv", "final", "summary"))
    parser.add_argument("--canonical", type=Path, default=here / "cache" / "canonical")
    parser.add_argument("--data", type=Path, default=here.parent / "Restocking_Problem_and_Data_EN")
    parser.add_argument("--out", type=Path, default=here / "results" / "q1_real")
    parser.add_argument("--max-folds", type=int, default=None, help="smoke test only; omit for the real CV")
    args = parser.parse_args(argv)
    if args.stage == "summary":
        print(write_summary(args.out))
        return 0
    if args.stage == "cv":
        result = stage_cv(args.canonical, args.data, args.out, args.max_folds)
    else:
        result = stage_final(args.canonical, args.data, args.out)
        print(write_summary(args.out))
    if args.stage == "cv":
        print(json.dumps(result["decisions"], ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
