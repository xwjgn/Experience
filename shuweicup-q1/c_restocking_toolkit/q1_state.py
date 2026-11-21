#!/usr/bin/env python3
"""Problem 1 on the real trajectories: quality audit, qualified references and
causal state predictors (needs numpy, pandas, scipy; reads trajectory_adapter output).

Representative point: the wrist keypoint j00 (primary) and, as a sensitivity
check, the palm centre = mean of j00, j05, j09, j13, j17 with fixed weights.
Left and right hands are separate tracks; a definition never changes with
visibility.

Qualified reference frame: reported hand quality "ok" AND hand-geometry check
passed (max relative bone-length deviation from the record median <= the 99.5 %
quantile of ok TRAIN frames). The threshold is frozen from train and applied to
validation and test. Flagged frames stay in the input stream (they are recorded
observations) but are never scored as references.

Predictors, all causal (prediction of frame k uses frames <= k-h only):
  hold            last observation
  observed_cv     velocity from the last two observations
  cv_kalman       constant-velocity Kalman filter, continuous white-acceleration
                  density q (exact variable-dt discretisation), isotropic R = r I
  quality_kalman  same, R = r I for ok frames and m r I for flagged frames
Kalman parameters maximise the one-step predictive log-likelihood on TRAIN
records over a log grid; validation selects between the two Kalman models by
one-step negative log-likelihood; test is reported once, by environment.

Metric: E = mean Euclidean distance between prediction and the qualified
recorded reference (problem statement section 4), at horizons of 1, 3, 10 and 30
frames, plus 90 %/95 % coverage of the Kalman predictive ball for the reference.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2

LABEL = "REAL_DATA_Q1_STATE_ESTIMATION"
HANDS = ("left", "right")
POINTS = {"wrist": (0,), "palm_center": (0, 5, 9, 13, 17)}
HORIZONS = (1, 3, 10, 30)
MODELS = ("hold", "observed_cv", "cv_kalman", "quality_kalman", "damped_kalman")
# damped_kalman: cv_kalman filtering, but the extrapolated velocity decays with time
# constant tau: p + v * tau * (1 - exp(-delta / tau)). One-step likelihood cannot
# identify tau, so it is chosen on TRAIN by the mean error over all horizons.
TAU_GRID = np.array([0.05, 0.1, 0.15, 0.2, 0.3, 0.45, 0.7, 1.0, 1.5, 2.5, 5.0, 10.0])
BONES = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (0, 9), (9, 10), (10, 11), (11, 12),
         (0, 13), (13, 14), (14, 15), (15, 16), (0, 17), (17, 18), (18, 19), (19, 20)]
GEOMETRY_QUANTILE = 0.995
BOUNDARY_S = 0.5
INITIAL_VELOCITY_SD = 0.5  # m/s, prior on the first frame of each track
Q_GRID = np.geomspace(0.03, 300.0, 17)    # m^2/s^3
R_GRID = np.geomspace(1e-7, 1e-3, 17)     # m^2 (sd 0.3 mm .. 3 cm)
M_GRID = np.array([1.0, 3.0, 10.0, 30.0, 100.0, 1e3, 1e4, 1e6])  # flagged-frame noise multiplier


# ----------------------------------------------------------------------------- data

def load_tracks(canonical: Path, data_root: Path) -> tuple[list[dict], pd.DataFrame]:
    frames = pd.read_parquet(canonical / "frames.parquet")
    segments = pd.read_csv(canonical / "segments.csv")
    seg_index = pd.read_csv(data_root / "Restocking_Tasks_and_Constraints" / "source_segment_index.csv")
    tasks = seg_index.groupby("episode_uuid")["task"].agg(lambda s: ";".join(sorted(set(s))))
    tracks = []
    for sample, g in frames.groupby("sample", sort=False):
        g = g.sort_values("frame_index")
        t = g["t_s"].to_numpy(np.float64)
        n = len(t)
        stage = np.full(n, "outside", dtype=object)
        for _, s in segments[segments["sample"] == sample].iterrows():
            a, b = int(s["first_frame"]), int(s["end_frame"])
            stage[a:b] = np.where(stage[a:b] == "outside", "interior", stage[a:b])
        for _, s in segments[segments["sample"] == sample].iterrows():
            for edge in (float(s["start_s"]), float(s["end_s"])):
                stage[np.abs(t - edge) <= BOUNDARY_S] = "boundary"
        meta = g.iloc[0]
        for hand in HANDS:
            joints = np.stack([g[[f"{hand}_j{j:02d}_{a}" for a in "xyz"]].to_numpy(np.float64) for j in range(21)], axis=1)
            lengths = np.stack([np.linalg.norm(joints[:, a] - joints[:, b], axis=1) for a, b in BONES], axis=1)
            deviation = np.max(np.abs(lengths / np.median(lengths, axis=0) - 1.0), axis=1)
            quality = g[f"{hand}_quality"].to_numpy(object)
            tracks.append({
                "sample": sample, "hand": hand, "dataset": meta["dataset"], "split": meta["split"],
                "environment_id": meta["environment_id"], "episode_uuid": meta["episode_uuid"],
                "task": tasks.get(meta["episode_uuid"], ""), "t": t, "quality": quality,
                "bad": quality != "ok", "bone_deviation": deviation, "stage": stage,
                "points": {name: joints[:, list(idx)].mean(axis=1) for name, idx in POINTS.items()},
            })
    return tracks, frames


def _runs(mask: np.ndarray) -> list[int]:
    runs, current = [], 0
    for value in mask:
        if value:
            current += 1
        elif current:
            runs.append(current)
            current = 0
    if current:
        runs.append(current)
    return runs


def quality_audit(tracks: list[dict]) -> dict:
    rows = []
    for tr in tracks:
        runs = _runs(tr["bad"])
        speed = np.linalg.norm(np.diff(tr["points"]["wrist"], axis=0), axis=1) / np.diff(tr["t"])
        rows.append({
            "sample": tr["sample"], "hand": tr["hand"], "dataset": tr["dataset"], "split": tr["split"],
            "environment_id": tr["environment_id"], "task": tr["task"], "frames": len(tr["t"]),
            "duration_s": float(tr["t"][-1] - tr["t"][0]), "median_dt_s": float(np.median(np.diff(tr["t"]))),
            "max_dt_s": float(np.max(np.diff(tr["t"]))),
            "occlusion": int(np.sum(tr["quality"] == "occlusion")), "near_edge": int(np.sum(tr["quality"] == "near edge")),
            "flagged_rate": float(np.mean(tr["bad"])), "flagged_runs": len(runs),
            "longest_flagged_run_frames": max(runs) if runs else 0,
            "median_wrist_speed_mps": float(np.median(speed)),
        })
    per_record = pd.DataFrame(rows)

    def grouped(keys):
        out = per_record.groupby(keys).agg(tracks=("frames", "size"), frames=("frames", "sum"),
                                           occlusion=("occlusion", "sum"), near_edge=("near_edge", "sum"),
                                           flagged_runs=("flagged_runs", "sum"),
                                           longest_run=("longest_flagged_run_frames", "max")).reset_index()
        out["flagged_rate"] = (out["occlusion"] + out["near_edge"]) / out["frames"]
        return out

    by_stage = []
    for tr in tracks:
        for stage in ("interior", "boundary", "outside"):
            m = tr["stage"] == stage
            by_stage.append({"dataset": tr["dataset"], "stage": stage, "frames": int(m.sum()),
                             "flagged": int(tr["bad"][m].sum())})
    stage = pd.DataFrame(by_stage).groupby(["dataset", "stage"]).sum().reset_index()
    stage["flagged_rate"] = stage["flagged"] / stage["frames"].clip(lower=1)
    geometry = []
    for cls in ("ok", "occlusion", "near edge"):
        dev = np.concatenate([tr["bone_deviation"][tr["quality"] == cls] for tr in tracks])
        spd = np.concatenate([(np.linalg.norm(np.diff(tr["points"]["wrist"], axis=0), axis=1) / np.diff(tr["t"]))[tr["quality"][1:] == cls]
                              for tr in tracks])
        geometry.append({"quality": cls, "frames": int(dev.size),
                         "bone_deviation_median": float(np.median(dev)), "bone_deviation_q99": float(np.quantile(dev, 0.99)),
                         "wrist_speed_median_mps": float(np.median(spd)), "wrist_speed_q99_mps": float(np.quantile(spd, 0.99))})
    return {"per_record": per_record, "by_source_split": grouped(["dataset", "split"]),
            "by_environment": grouped(["split", "environment_id"]), "by_task": grouped(["task"]),
            "by_stage": stage, "geometry": pd.DataFrame(geometry)}


def qualify(tracks: list[dict]) -> dict:
    ok_train = np.concatenate([tr["bone_deviation"][~tr["bad"]] for tr in tracks if tr["split"] == "train"])
    threshold = float(np.quantile(ok_train, GEOMETRY_QUANTILE))
    for tr in tracks:
        tr["qualified"] = (~tr["bad"]) & (tr["bone_deviation"] <= threshold)
    rows = []
    for split in ("train", "validation", "test"):
        for ds in ("EgoDemo", "EgoPro", "EgoStandard", "all"):
            sel = [tr for tr in tracks if tr["split"] == split and (ds == "all" or tr["dataset"] == ds)]
            frames = sum(len(tr["t"]) for tr in sel)
            if frames:
                rows.append({"split": split, "dataset": ds, "hand_frames": frames,
                             "flagged": int(sum(tr["bad"].sum() for tr in sel)),
                             "geometry_rejected_ok": int(sum((~tr["bad"] & ~tr["qualified"]).sum() for tr in sel)),
                             "qualified": int(sum(tr["qualified"].sum() for tr in sel)),
                             "retention": sum(tr["qualified"].sum() for tr in sel) / frames})
    return {"geometry_threshold": threshold, "geometry_quantile": GEOMETRY_QUANTILE,
            "fitted_on": "ok frames of train records", "retention": pd.DataFrame(rows)}


# --------------------------------------------------------------------------- Kalman

def kalman_grid(t, z, bad, q, r, m, store=False):
    """Constant-velocity Kalman filter, isotropic, vectorised over parameter vectors.

    Returns the one-step predictive log-likelihood of frames 1..n-1 per parameter
    set and, if store, the filtered state after each frame (single parameter set).
    """
    q, r, m = (np.atleast_1d(np.asarray(x, dtype=np.float64)) for x in (q, r, m))
    G = q.shape[0]
    noise0 = r * np.where(bad[0], m, 1.0)
    p = np.repeat(z[:1], G, axis=0)
    v = np.zeros((G, 3))
    P11, P12, P22 = noise0.copy(), np.zeros(G), np.full(G, INITIAL_VELOCITY_SD ** 2)
    loglik = np.zeros(G)
    n = len(t)
    if store:
        out_p, out_v, out_P = np.empty((n, 3)), np.empty((n, 3)), np.empty((n, 3))
        out_p[0], out_v[0], out_P[0] = p[0], v[0], (P11[0], P12[0], P22[0])
    log2pi = math.log(2 * math.pi)
    for k in range(1, n):
        dt = t[k] - t[k - 1]
        pp = p + v * dt
        a11 = P11 + 2 * dt * P12 + dt * dt * P22 + q * dt ** 3 / 3
        a12 = P12 + dt * P22 + q * dt * dt / 2
        a22 = P22 + q * dt
        S = a11 + (r * m if bad[k] else r)
        y = z[k] - pp
        loglik -= 0.5 * ((y * y).sum(axis=1) / S + 3 * (np.log(S) + log2pi))
        K1, K2 = a11 / S, a12 / S
        p = pp + K1[:, None] * y
        v = v + K2[:, None] * y
        P11, P12, P22 = (1 - K1) * a11, (1 - K1) * a12, a22 - K2 * a12
        if store:
            out_p[k], out_v[k], out_P[k] = p[0], v[0], (P11[0], P12[0], P22[0])
    if store:
        return loglik, (out_p, out_v, out_P)
    return loglik


def fit_kalman(tracks: list[dict], point: str, quality_aware: bool) -> dict:
    qq, rr = np.meshgrid(Q_GRID, R_GRID, indexing="ij")
    if quality_aware:
        qq, rr, mm = (a.ravel() for a in np.meshgrid(Q_GRID, R_GRID, M_GRID, indexing="ij"))
    else:
        qq, rr, mm = qq.ravel(), rr.ravel(), np.ones(qq.size)
    total, frames = np.zeros(qq.size), 0
    for tr in tracks:
        if tr["split"] != "train":
            continue
        bad = tr["bad"] if quality_aware else np.zeros_like(tr["bad"])
        total += kalman_grid(tr["t"], tr["points"][point], bad, qq, rr, mm)
        frames += len(tr["t"]) - 1
    best = int(np.argmax(total))
    params = {"q": float(qq[best]), "r": float(rr[best]), "m": float(mm[best])}
    edges = {"q_at_grid_edge": params["q"] in (Q_GRID[0], Q_GRID[-1]),
             "r_at_grid_edge": params["r"] in (R_GRID[0], R_GRID[-1]),
             "m_at_grid_edge": quality_aware and params["m"] in (M_GRID[0], M_GRID[-1])}
    return {**params, "measurement_sd_mm": 1000 * math.sqrt(params["r"]),
            "flagged_measurement_sd_mm": 1000 * math.sqrt(params["r"] * params["m"]),
            "train_mean_nll_per_frame": float(-total[best] / frames), "train_frames": frames, **edges}


# ------------------------------------------------------------------------ evaluate

def predictions(tr: dict, point: str, params: dict) -> dict:
    """Prediction for every target frame k and horizon h from information up to k-h."""
    t, z = tr["t"], tr["points"][point]
    out = {}
    states = {}
    for name in ("cv_kalman", "quality_kalman"):
        p = params[name]
        bad = tr["bad"] if name == "quality_kalman" else np.zeros_like(tr["bad"])
        loglik, (fp, fv, fP) = kalman_grid(t, z, bad, p["q"], p["r"], p["m"], store=True)
        states[name] = (fp, fv, fP, p)
    for h in HORIZONS:
        k = np.arange(h + 1, len(t))  # origin o = k - h >= 1 so observed_cv is defined
        if k.size == 0:
            continue
        o = k - h
        delta = (t[k] - t[o])[:, None]
        pred = {"hold": z[o],
                "observed_cv": z[o] + (z[o] - z[o - 1]) / (t[o] - t[o - 1])[:, None] * delta}
        var = {}
        for name, (fp, fv, fP, p) in states.items():
            d = delta[:, 0]
            pred[name] = fp[o] + fv[o] * delta
            var[name] = fP[o, 0] + 2 * d * fP[o, 1] + d * d * fP[o, 2] + p["q"] * d ** 3 / 3 + p["r"]
        if "damped_kalman" in params:
            fp, fv, _, _ = states["cv_kalman"]
            tau = params["damped_kalman"]["tau"]
            pred["damped_kalman"] = fp[o] + fv[o] * tau * (1.0 - np.exp(-delta / tau))
        out[h] = {"target": k, "origin": o, "pred": pred, "var": var}
    return out


def fit_damping(tracks: list[dict], point: str, cv_params: dict) -> dict:
    """Choose tau on TRAIN: mean qualified error over HORIZONS, cv_kalman filter states."""
    sums = np.zeros((len(HORIZONS), TAU_GRID.size))
    counts = np.zeros(len(HORIZONS))
    for tr in tracks:
        if tr["split"] != "train":
            continue
        t, z = tr["t"], tr["points"][point]
        _, (fp, fv, _) = kalman_grid(t, z, np.zeros_like(tr["bad"]), cv_params["q"], cv_params["r"], 1.0, store=True)
        for i, h in enumerate(HORIZONS):
            k = np.arange(h + 1, len(t))
            k = k[tr["qualified"][k]]
            o = k - h
            delta = (t[k] - t[o])[:, None, None]
            pred = fp[o][:, None, :] + fv[o][:, None, :] * TAU_GRID[None, :, None] * (1 - np.exp(-delta / TAU_GRID[None, :, None]))
            sums[i] += np.linalg.norm(pred - z[k][:, None, :], axis=2).sum(axis=0)
            counts[i] += k.size
    objective = (sums / counts[:, None]).mean(axis=0)
    best = int(np.argmin(objective))
    return {"tau": float(TAU_GRID[best]), "train_mean_error_mm_over_horizons": float(1000 * objective[best]),
            "tau_at_grid_edge": best in (0, TAU_GRID.size - 1)}


def evaluate(tracks: list[dict], point: str, params: dict) -> pd.DataFrame:
    rows = []
    c90, c95 = chi2.ppf(0.90, 3), chi2.ppf(0.95, 3)
    for tr in tracks:
        z = tr["points"][point]
        recent_bad = np.convolve(tr["bad"].astype(float), np.ones(10), mode="full")[:len(z)] > 0  # flagged in last 10 frames
        for h, item in predictions(tr, point, params).items():
            k = item["target"]
            keep = tr["qualified"][k]
            k_ = k[keep]
            origin_context = np.where(recent_bad[item["origin"][keep]], "flagged_recently", "clean_history")
            for model in MODELS:
                err = np.linalg.norm(item["pred"][model][keep] - z[k_], axis=1)
                frame = {"sample": tr["sample"], "hand": tr["hand"], "dataset": tr["dataset"], "split": tr["split"],
                         "environment_id": tr["environment_id"], "point": point, "model": model, "horizon": h}
                if model in item["var"]:
                    s = item["var"][model][keep]
                    nis = (err ** 2) / s
                    nll = 0.5 * (nis + 3 * np.log(2 * math.pi * s))
                    cov90, cov95, radius95 = nis <= c90, nis <= c95, np.sqrt(c95 * s)
                else:
                    nll = cov90 = cov95 = radius95 = np.full(err.shape, np.nan)
                for context in ("clean_history", "flagged_recently"):
                    sel = origin_context == context
                    if sel.any():
                        rows.append({**frame, "context": context, "n": int(sel.sum()), "sum_err": float(err[sel].sum()),
                                     "sum_sq_err": float((err[sel] ** 2).sum()), "errors": err[sel],
                                     "nis": nis[sel] if model in item["var"] else np.empty(0),
                                     "variance": s[sel] if model in item["var"] else np.empty(0),
                                     "sum_nll": float(np.nansum(nll[sel])) if model in item["var"] else np.nan,
                                     "cov90": float(np.nansum(cov90[sel])) if model in item["var"] else np.nan,
                                     "cov95": float(np.nansum(cov95[sel])) if model in item["var"] else np.nan,
                                     "sum_radius95": float(np.nansum(radius95[sel])) if model in item["var"] else np.nan})
    return pd.DataFrame(rows)


CALIBRATION_LEVEL = 0.95
CALIBRATION_METHODS = ("uncalibrated", "global_scale", "regime_scale")


def calibrate(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scale the Kalman predictive variance so the 95 % ball covers TRAIN residuals.

    global_scale: one factor per (point, model, horizon); regime_scale: separate
    factors for clean_history and flagged_recently (flag within the 10 frames
    before the prediction origin, known at prediction time). Factors are fitted
    on train only and applied unchanged to validation and test. Adjacent frames
    are dependent, so coverage is an empirical check, not a finite-sample guarantee.
    """
    kalman = raw[raw["model"].isin(["cv_kalman", "quality_kalman"])]
    crit = chi2.ppf(CALIBRATION_LEVEL, 3)
    factors, rows = [], []
    for (point, model, h), g in kalman.groupby(["point", "model", "horizon"]):
        train = g[g.split == "train"]
        all_nis = np.concatenate(train["nis"].to_list())
        scale = {"uncalibrated": {"clean_history": 1.0, "flagged_recently": 1.0}}
        global_factor = max(float(np.quantile(all_nis, CALIBRATION_LEVEL)) / crit, 1e-9)
        scale["global_scale"] = {c: global_factor for c in ("clean_history", "flagged_recently")}
        scale["regime_scale"] = {}
        for ctx in ("clean_history", "flagged_recently"):
            ctx_nis = np.concatenate(train[train.context == ctx]["nis"].to_list())
            scale["regime_scale"][ctx] = max(float(np.quantile(ctx_nis, CALIBRATION_LEVEL)) / crit, 1e-9)
            factors.append({"point": point, "model": model, "horizon": h, "context": ctx,
                            "train_n": int(ctx_nis.size), "global_factor": global_factor,
                            "regime_factor": scale["regime_scale"][ctx]})
        for split in ("train", "validation", "test"):
            for ctx in ("clean_history", "flagged_recently", "all"):
                sel = g[(g.split == split) & ((g.context == ctx) if ctx != "all" else True)]
                if sel.empty:
                    continue
                for method in CALIBRATION_METHODS:
                    nis_parts, radius_parts = [], []
                    for _, row in sel.iterrows():
                        f = scale[method][row["context"]]
                        nis_parts.append(row["nis"] / f)
                        radius_parts.append(np.sqrt(crit * row["variance"] * f))
                    nis_all, radius = np.concatenate(nis_parts), np.concatenate(radius_parts)
                    rows.append({"point": point, "model": model, "horizon": h, "split": split, "context": ctx,
                                 "method": method, "n": int(nis_all.size),
                                 "coverage95": float(np.mean(nis_all <= crit)),
                                 "mean_radius95_mm": 1000 * float(np.mean(radius))})
    return pd.DataFrame(factors), pd.DataFrame(rows)


def summarise(raw: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    def agg(g):
        errors = np.concatenate(g["errors"].to_list())
        n = int(g["n"].sum())
        kalman = bool(g["sum_nll"].notna().any())  # only Kalman rows carry a likelihood
        return pd.Series({
            "n": n, "tracks": len(set(zip(g["sample"], g["hand"]))),
            "E_mean_mm": 1000 * g["sum_err"].sum() / n, "median_mm": 1000 * float(np.median(errors)),
            "p90_mm": 1000 * float(np.quantile(errors, 0.9)), "rmse_mm": 1000 * math.sqrt(g["sum_sq_err"].sum() / n),
            "mean_nll": g["sum_nll"].sum() / n if kalman else np.nan,
            "coverage90": g["cov90"].sum() / n if kalman else np.nan,
            "coverage95": g["cov95"].sum() / n if kalman else np.nan,
            "mean_radius95_mm": 1000 * g["sum_radius95"].sum() / n if kalman else np.nan,
        })
    return raw.groupby(keys).apply(agg, include_groups=False).reset_index()


def run(canonical: Path, data_root: Path, out: Path) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    tracks, _ = load_tracks(canonical, data_root)
    audit = quality_audit(tracks)
    qual = qualify(tracks)
    fitted, metrics, env_metrics, context_metrics = {}, [], [], []
    raw_all = []
    for point in POINTS:
        params = {"cv_kalman": fit_kalman(tracks, point, False), "quality_kalman": fit_kalman(tracks, point, True)}
        params["damped_kalman"] = fit_damping(tracks, point, params["cv_kalman"])
        fitted[point] = params
        raw = evaluate(tracks, point, params)
        raw_all.append(raw)
        metrics.append(summarise(raw, ["point", "model", "horizon", "split"]).assign(dataset="all"))
        metrics.append(summarise(raw, ["point", "model", "horizon", "split", "dataset"]))
        env_metrics.append(summarise(raw, ["point", "model", "horizon", "split", "environment_id"]))
        context_metrics.append(summarise(raw, ["point", "model", "horizon", "split", "context"]))
    metrics = pd.concat(metrics, ignore_index=True)
    env_metrics = pd.concat(env_metrics, ignore_index=True)
    context_metrics = pd.concat(context_metrics, ignore_index=True)
    calibration_factors, calibration = calibrate(pd.concat(raw_all, ignore_index=True))
    # Validation chooses between the two Kalman models (one-step NLL), per point.
    selection = {}
    for point in POINTS:
        v = metrics[(metrics.point == point) & (metrics.split == "validation") & (metrics.dataset == "all")
                    & (metrics.horizon == 1) & metrics.model.isin(["cv_kalman", "quality_kalman"])]
        selection[point] = v.sort_values("mean_nll").iloc[0]["model"]
    # Paired environment-level differences on test (quality vs cv Kalman), wrist.
    env_pairs = []
    for point in POINTS:
        for h in HORIZONS:
            e = env_metrics[(env_metrics.point == point) & (env_metrics.horizon == h) & (env_metrics.split == "test")]
            piv = e.pivot(index="environment_id", columns="model", values="E_mean_mm")
            for env, row in piv.iterrows():
                env_pairs.append({"point": point, "horizon": h, "environment_id": env,
                                  **{m: row[m] for m in MODELS},
                                  "quality_minus_cv_kalman_mm": row["quality_kalman"] - row["cv_kalman"],
                                  "best_kalman_minus_hold_mm": min(row["quality_kalman"], row["cv_kalman"]) - row["hold"]})
    env_pairs = pd.DataFrame(env_pairs)
    # Write tables.
    for name, frame in {**{f"quality_{k}": v for k, v in audit.items()},
                        "qualification_retention": qual["retention"], "prediction_metrics": metrics,
                        "environment_metrics": env_metrics, "context_metrics": context_metrics,
                        "test_environment_pairs": env_pairs, "calibration_factors": calibration_factors,
                        "calibration_coverage": calibration}.items():
        frame.to_csv(out / f"{name}.csv", index=False, encoding="utf-8")
    summary = {
        "label": LABEL, "points": {k: list(v) for k, v in POINTS.items()}, "primary_point": "wrist",
        "horizons_frames": HORIZONS, "models": MODELS,
        "qualification": {k: v for k, v in qual.items() if k != "retention"},
        "fitted_parameters": fitted, "validation_selected_kalman": selection,
        "grids": {"q": [Q_GRID[0], Q_GRID[-1], len(Q_GRID)], "r": [R_GRID[0], R_GRID[-1], len(R_GRID)],
                  "m": M_GRID.tolist()},
        "information_rules": ["parameters and geometry threshold fitted on train only",
                              "validation used only to choose between Kalman models",
                              "prediction of frame k uses observations up to k-h; the scored reference is never an input",
                              "flagged frames are inputs but never scored references"],
        "interpretation": "Errors are against recorded coordinates (qualified references), not independent ground truth.",
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    (out / "summary.md").write_text(markdown(summary, audit, qual, metrics, env_pairs, context_metrics, calibration),
                                    encoding="utf-8")
    return {"summary": summary, "metrics": metrics, "audit": audit, "qualification": qual,
            "env_pairs": env_pairs, "context": context_metrics, "env_metrics": env_metrics, "tracks": tracks,
            "calibration": calibration, "calibration_factors": calibration_factors}


def _fmt(x, digits=2):
    return "—" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{digits}f}"


def markdown(summary, audit, qual, metrics, env_pairs, context, calibration=None) -> str:
    lines = ["# 第一问：真实轨迹状态估计结果", "",
             "数据：69 条记录 × 左右手，共 132,140 条手部帧轨迹。参数与几何阈值只用训练集拟合，验证集只用于在两种卡尔曼模型间选择，测试集只报告一次。",
             "误差相对于记录中的合格参考坐标，不是独立的物理真值。", "",
             "## 1 数据质量", "", "| 来源 | 划分 | 手部帧 | 遮挡 | 靠边 | 标记率 | 最长连续标记（帧） |", "|---|---|---:|---:|---:|---:|---:|"]
    for _, r in audit["by_source_split"].iterrows():
        lines.append(f"| {r.dataset} | {r.split} | {r.frames} | {r.occlusion} | {r.near_edge} | {r.flagged_rate:.2%} | {r.longest_run} |")
    lines += ["", "| 质量标记 | 帧数 | 骨长偏差中位数 | 骨长偏差 99% | 腕速中位数 m/s | 腕速 99% m/s |", "|---|---:|---:|---:|---:|---:|"]
    for _, r in audit["geometry"].iterrows():
        lines.append(f"| {r.quality} | {r.frames} | {r.bone_deviation_median:.3f} | {r.bone_deviation_q99:.3f} | {r.wrist_speed_median_mps:.3f} | {r.wrist_speed_q99_mps:.3f} |")
    lines += ["", "| 来源 | 阶段 | 帧数 | 标记率 |", "|---|---|---:|---:|"]
    for _, r in audit["by_stage"].iterrows():
        lines.append(f"| {r.dataset} | {r.stage} | {r.frames} | {r.flagged_rate:.2%} |")
    lines += ["", f"阶段定义：interior 为标注片段内部，boundary 为片段起止 ±{BOUNDARY_S} 秒，outside 为片段之外。", "",
              "## 2 合格参考", "",
              f"合格 = 质量标记为 ok，且最大相对骨长偏差 ≤ {qual['geometry_threshold']:.4f}（训练集 ok 帧的 {GEOMETRY_QUANTILE:.1%} 分位，冻结后用于验证与测试）。", "",
              "| 划分 | 来源 | 手部帧 | 被标记 | 几何剔除 | 合格 | 保留率 |", "|---|---|---:|---:|---:|---:|---:|"]
    for _, r in qual["retention"].iterrows():
        lines.append(f"| {r.split} | {r.dataset} | {r.hand_frames} | {r.flagged} | {r.geometry_rejected_ok} | {r.qualified} | {r.retention:.2%} |")
    lines += ["", "## 3 训练集拟合参数", "", "| 代表点 | 模型 | q (m²/s³) | 观测噪声 sd (mm) | 标记帧噪声 sd (mm) | 训练每帧 NLL | 网格边界 |", "|---|---|---:|---:|---:|---:|---|"]
    for point, models in summary["fitted_parameters"].items():
        for name, p in models.items():
            if name == "damped_kalman":
                continue
            edge = ", ".join(k.split("_")[0] for k in ("q_at_grid_edge", "r_at_grid_edge", "m_at_grid_edge") if p.get(k)) or "否"
            lines.append(f"| {point} | {name} | {p['q']:.3g} | {p['measurement_sd_mm']:.2f} | {p['flagged_measurement_sd_mm']:.2f} | {p['train_mean_nll_per_frame']:.3f} | {edge} |")
    damped = ", ".join(f"{point} τ = {m['damped_kalman']['tau']} s{'（网格边界）' if m['damped_kalman']['tau_at_grid_edge'] else ''}"
                       for point, m in summary["fitted_parameters"].items())
    lines += ["", f"damped_kalman 沿用 cv_kalman 的滤波状态，外推速度按 exp(−Δ/τ) 衰减；τ 在训练集上按各步长平均误差选取：{damped}。没有给出其预测区间。", "",
              f"验证集按一步预测 NLL 选出的卡尔曼模型：{summary['validation_selected_kalman']}。", "",
              "## 4 测试集预测误差（腕点，全部来源）", "",
              "| 预测步长（帧） | 模型 | E 均值 mm | 中位数 mm | 90% 分位 mm | 95% 覆盖率 | 95% 半径 mm |", "|---:|---|---:|---:|---:|---:|---:|"]
    t = metrics[(metrics.point == "wrist") & (metrics.split == "test") & (metrics.dataset == "all")]
    for h in HORIZONS:
        for model in MODELS:
            r = t[(t.horizon == h) & (t.model == model)].iloc[0]
            lines.append(f"| {h} | {model} | {r.E_mean_mm:.2f} | {r.median_mm:.2f} | {r.p90_mm:.2f} | {_fmt(r.coverage95, 3)} | {_fmt(r.mean_radius95_mm)} |")
    lines += ["", "## 5 测试集各环境（腕点）", "", "| 步长 | 环境 | hold | observed_cv | cv_kalman | quality_kalman | 质量KF − 标准KF |", "|---:|---|---:|---:|---:|---:|---:|"]
    for _, r in env_pairs[env_pairs.point == "wrist"].iterrows():
        lines.append(f"| {r.horizon} | {r.environment_id} | {r.hold:.2f} | {r.observed_cv:.2f} | {r.cv_kalman:.2f} | {r.quality_kalman:.2f} | {r.quality_minus_cv_kalman_mm:+.2f} |")
    lines += ["", "## 6 近期有标记帧时的误差（测试集，腕点）", "",
              "flagged_recently：预测起点之前 10 帧内出现过遮挡或靠边标记。", "",
              "| 步长 | 情形 | 样本数 | hold | observed_cv | cv_kalman | quality_kalman |", "|---:|---|---:|---:|---:|---:|---:|"]
    c = context[(context.point == "wrist") & (context.split == "test")]
    for h in HORIZONS:
        for ctx in ("clean_history", "flagged_recently"):
            sel = c[(c.horizon == h) & (c.context == ctx)].set_index("model")
            if len(sel):
                lines.append(f"| {h} | {ctx} | {int(sel.loc['hold', 'n'])} | " + " | ".join(f"{sel.loc[m, 'E_mean_mm']:.2f}" for m in MODELS) + " |")
    lines += ["", "## 7 代表点敏感性（测试集，全部来源，E 均值 mm）", "", "| 步长 | 模型 | 腕点 | 掌心 |", "|---:|---|---:|---:|"]
    for h in HORIZONS:
        for model in MODELS:
            vals = [metrics[(metrics.point == p) & (metrics.split == "test") & (metrics.dataset == "all") & (metrics.horizon == h) & (metrics.model == model)].iloc[0].E_mean_mm for p in POINTS]
            lines.append(f"| {h} | {model} | {vals[0]:.2f} | {vals[1]:.2f} |")
    if calibration is not None and len(calibration):
        lines += ["", "## 8 不确定性校准（常速度卡尔曼，腕点，95% 名义水平）", "",
                  "uncalibrated 为模型自身方差；global_scale 每个步长一个放大系数；regime_scale 按预测起点前 10 帧是否出现标记分两组。系数只在训练集拟合。", "",
                  "| 步长 | 划分 | 情形 | 方法 | 样本数 | 覆盖率 | 平均半径 mm |", "|---:|---|---|---|---:|---:|---:|"]
        c = calibration[(calibration.point == "wrist") & (calibration.model == "cv_kalman")
                        & calibration.split.isin(["validation", "test"])]
        for h in HORIZONS:
            for split in ("validation", "test"):
                for ctx in ("clean_history", "flagged_recently"):
                    for method in CALIBRATION_METHODS:
                        r = c[(c.horizon == h) & (c.split == split) & (c.context == ctx) & (c.method == method)]
                        if len(r):
                            r = r.iloc[0]
                            lines.append(f"| {h} | {split} | {ctx} | {method} | {r.n} | {r.coverage95:.3f} | {r.mean_radius95_mm:.2f} |")
    lines += ["", "全部分组结果见同目录 CSV；测试集只有 4 个环境，跨环境结论需谨慎。", ""]
    return "\n".join(lines)


def main(argv=None) -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--canonical", type=Path, default=here / "cache" / "canonical")
    parser.add_argument("--data", type=Path, default=here.parent / "Restocking_Problem_and_Data_EN")
    parser.add_argument("--out", type=Path, default=here / "results" / "q1_real")
    args = parser.parse_args(argv)
    result = run(args.canonical, args.data, args.out)
    print(f"Q1 done; selected Kalman by validation: {result['summary']['validation_selected_kalman']}")
    print(f"Summary: {(args.out / 'summary.md').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
