#!/usr/bin/env python3
"""Problem 1, second (literature-driven) improvement round: LOEO CV, then test once.

1  Camera-frustum visibility features (EgoPoser, ECCV 2024; OmniEgoCap preprint).
   All 21 joints are projected into the head-left image (pinhole, OpenCV axes,
   camera -> world pose from the adapter). Normalised margin to the image border
   m = min(u/W, 1-u/W, v/H, 1-v/H): negative outside the image, -1 behind the
   camera. Features: wrist margin, minimum (first joint to leave) and maximum over
   joints, fraction of joints in view, rate of the minimum margin (5-frame mean,
   1/s), its linear prediction 0.33 s ahead and the time to exit (0-2 s).
2  Calibration of the Kalman predictive variance by motion state x recent flag:
   (a) cells: one factor per cell (round-1 choice);
   (b) clustered: cells with fewer than N_MIN training scores are pooled with the
       same-flag cell whose 95 % score quantile is closest (after Ding et al.,
       NeurIPS 2023);
   (c) quantreg: linear 95 % quantile regression of the log score on the cell
       indicators and continuous covariates (inspired by Gibbs, Cherian & Candes,
       JRSSB 2025; their exact finite-sample guarantee is not claimed).
3  Uncertainty for the boosting predictor (USST, ICCV 2023; Ye & Kim, ECCV 2018):
   residual norms by 2-way cross-fitting inside the training environments;
   (a) cells: split-conformal radius per motion state x flag cell;
   (b) quantile: 95 % quantile boosting of the residual norm on covariates
       (including the frustum margin), scaled by a conformal factor from the other half.
4  Alarm metrics for the reliability model: threshold that maximises F1 on the
   pooled development out-of-fold predictions; precision, recall and F1.
5  Prediction error by occlusion severity at the prediction origin.

Decision rules (written before the CV was run):
  R1 adopt frustum features if the CV onset AUC improves for both logistic
     regression and gradient boosting (point estimates; intervals reported);
  R2 Kalman calibration: smallest maximum |coverage - 0.95| over the 24 cells;
     ties within 0.005 broken by the smaller mean radius;
  R3 boosting uncertainty: the same criterion between (a) and (b);
  R4 alarm threshold: F1-maximising on development out-of-fold predictions.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score

import q1_improve as qi
import q1_reliability as rel
from q1_improve import CRIT95, HORIZONS, KalmanCache, Learned, origins

LABEL = "REAL_DATA_Q1_LOEO_ROUND2"
FRUSTUM_FEATURES = ["wrist_margin", "min_margin", "max_margin", "frac_in_view", "margin_rate", "margin_pred",
                    "time_to_exit"]
FRUSTUM_NAMES = {"wrist_margin": "腕点到画面边缘余量", "min_margin": "最先出画关节的余量", "max_margin": "最内侧关节的余量",
                 "frac_in_view": "画面内关节比例", "margin_rate": "最小余量变化率", "margin_pred": "0.33 s 后的预测余量",
                 "time_to_exit": "预计离开画面时间"}
UNC_FEATURES = ["state", "flag", "since_last_flag", "run_len", "speed5", "min_margin", "margin_rate", "head_turn_rate",
                "off_axis_rate"]
N_MIN = 500
SEVERITY = ("clean", "recovering", "flag_run_1-2", "flag_run_3-5", "flag_run_6+")
KF_METHODS = ("cells", "clustered", "quantreg")
BOOST_METHODS = ("cells", "quantile")
SCORED_MODELS = ("hold", "cv_kalman", "damped_kalman", "boosting")


# ----------------------------------------------------------------- features

def _rot(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q.T
    return np.stack([
        np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
        np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
        np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def frustum_margins(joints: np.ndarray, cam_t: np.ndarray, cam_q: np.ndarray, intr) -> np.ndarray:
    """Normalised margin of every joint to the image border, shape (n, joints)."""
    fx, fy, cx, cy, W, H = intr
    pc = np.einsum("nji,nkj->nki", _rot(cam_q), joints - cam_t[:, None, :])  # R^T (p - t)
    depth = pc[..., 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = fx * pc[..., 0] / depth + cx
        v = fy * pc[..., 1] / depth + cy
        m = np.minimum.reduce([u / W, 1 - u / W, v / H, 1 - v / H])
    return np.nan_to_num(np.where(depth > 1e-6, m, -1.0), nan=-1.0)


def load2(canonical: Path, data_root: Path) -> list[dict]:
    tracks = qi.load(canonical, data_root)
    joint_cols = [f"{h}_j{j:02d}_{a}" for h in ("left", "right") for j in range(21) for a in "xyz"]
    cam_cols = ["cam_x", "cam_y", "cam_z", "cam_qw", "cam_qx", "cam_qy", "cam_qz",
                "cam_fx", "cam_fy", "cam_cx", "cam_cy", "cam_width", "cam_height"]
    frames = pd.read_parquet(canonical / "frames.parquet", columns=["sample", "frame_index"] + joint_cols + cam_cols)
    groups = {s: g.sort_values("frame_index") for s, g in frames.groupby("sample", sort=False)}
    for tr in tracks:
        g = groups[tr["sample"]]
        joints = np.stack([g[[f"{tr['hand']}_j{j:02d}_{a}" for a in "xyz"]].to_numpy(float) for j in range(21)], axis=1)
        intr = tuple(float(g[c].iloc[0]) for c in cam_cols[7:])
        m = frustum_margins(joints, g[cam_cols[:3]].to_numpy(float), g[cam_cols[3:7]].to_numpy(float), intr)
        m_min = m.min(axis=1)
        dt = np.r_[np.nan, np.diff(tr["t"])]
        rate = rel._rolling_mean(np.r_[np.nan, np.diff(m_min)] / dt, 5)
        tte = np.where(m_min <= 0, 0.0, np.where(rate < 0, np.clip(m_min / np.maximum(-rate, 1e-6), 0, 2.0), 2.0))
        run = np.zeros(len(m_min))
        since = np.zeros(len(m_min))
        r, s = 0, 30
        for i, f in enumerate(tr["bad"]):
            r, s = (r + 1, 0) if f else (0, min(s + 1, 30))
            run[i], since[i] = r, s
        q = tr["quality"]
        tr.update({"wrist_margin": m[:, 0], "min_margin": m_min, "max_margin": m.max(axis=1),
                   "frac_in_view": (m > 0).mean(axis=1), "margin_rate": rate, "margin_pred": m_min + rate * 0.33,
                   "time_to_exit": np.where(np.isnan(rate), np.nan, tte), "run_len": run, "since_last_flag": since,
                   "is_near_edge": q == "near edge", "is_occlusion": q == "occlusion"})
    return tracks


def severity(run: np.ndarray, since: np.ndarray) -> np.ndarray:
    return np.select([run >= 6, run >= 3, run >= 1, since <= 10],
                     ["flag_run_6+", "flag_run_3-5", "flag_run_1-2", "recovering"], "clean")


# ---------------------------------------------------------------- scoring

def score2(tracks, params, learned, cache, cuts) -> pd.DataFrame:
    parts = []
    for tr in tracks:
        z = tr["points"]["wrist"]
        for h in HORIZONS:
            o, k = origins(tr, h)
            if o.size == 0:
                continue
            item = cache.predictions(tr, params)[h]
            idx = k - h - 1
            preds = {m: item["pred"][m][idx] for m in ("hold", "cv_kalman", "damped_kalman")}
            if learned is not None:
                preds["boosting"] = learned[h].predict(tr, o, h)
            base = {"sample": tr["sample"], "hand": tr["hand"], "environment_id": tr["environment_id"], "horizon": h,
                    "origin": o, "target": k,  # frame indices; used by online (ACI) calibration
                    "state": np.digitize(tr["speed5"][o], cuts), "flag": tr["recent_flag"][o],
                    **{c: tr[c][o] for c in ("since_last_flag", "run_len", "speed5", "min_margin", "margin_rate",
                                             "head_turn_rate", "off_axis_rate")},
                    "var": item["var"]["cv_kalman"][idx]}
            for m, p in preds.items():
                parts.append(pd.DataFrame({**base, "model": m, "err": np.linalg.norm(p - z[k], axis=1)}))
    df = pd.concat(parts, ignore_index=True)
    df["sev"] = severity(df["run_len"].to_numpy(), df["since_last_flag"].to_numpy())
    return df


def _qr_design(d: pd.DataFrame) -> np.ndarray:
    s1, s2, f = (d["state"] == 1).astype(float), (d["state"] == 2).astype(float), d["flag"].astype(float)
    cols = [s1, s2, f, f * s1, f * s2, np.log1p(d["since_last_flag"]), d["speed5"],
            d["min_margin"].clip(-1, 1), d["margin_rate"].clip(-5, 5)]
    return np.nan_to_num(np.column_stack(cols), nan=0.0)


def kf_calibration_fit(train_kf: pd.DataFrame, seed: int = 0) -> dict:
    fitted = {}
    for h, g in train_kf.groupby("horizon"):
        nis = (g["err"] ** 2 / g["var"]).to_numpy()
        cell = list(zip(g["state"].to_numpy(), g["flag"].to_numpy()))
        cells = sorted(set(cell))
        by_cell = {c: nis[[x == c for x in cell]] for c in cells}
        q95 = lambda a: float(np.quantile(a, 0.95)) / CRIT95
        flag_pool = {b: nis[g["flag"].to_numpy() == b] for b in (False, True)}
        f_cells = {c: q95(v) if v.size >= 30 else q95(flag_pool[c[1]]) for c, v in by_cell.items()}
        clusters = {c: {c} for c in cells}
        while True:
            small = [c for c in clusters if sum(by_cell[x].size for x in clusters[c]) < N_MIN]
            if not small or len(clusters) == 1:
                break
            c = min(small, key=lambda x: sum(by_cell[y].size for y in clusters[x]))
            same = [d for d in clusters if d != c and d[1] == c[1]] or [d for d in clusters if d != c]
            pool = lambda key: np.concatenate([by_cell[x] for x in clusters[key]])
            target = min(same, key=lambda d: abs(np.quantile(pool(d), 0.95) - np.quantile(pool(c), 0.95)))
            clusters[target] |= clusters.pop(c)
        f_clustered = {}
        for key, members in clusters.items():
            factor = q95(np.concatenate([by_cell[x] for x in members]))
            for x in members:
                f_clustered[x] = factor
        rng = np.random.default_rng(seed)
        take = rng.choice(len(g), size=min(len(g), 30000), replace=False)
        X = _qr_design(g.iloc[take])
        qr = QuantileRegressor(quantile=0.95, alpha=0.0, solver="highs").fit(X, np.log(nis[take]))
        fitted[h] = {"cells": f_cells, "clustered": f_clustered, "quantreg": qr,
                     "clusters": [sorted(map(lambda x: (int(x[0]), bool(x[1])), m)) for m in clusters.values()]}
    return fitted


def kf_calibration_apply(kf: pd.DataFrame, fitted: dict) -> pd.DataFrame:
    rows = []
    for h, g in kf.groupby("horizon"):
        f = fitted[h]
        var = g["var"].to_numpy()
        keys = list(zip(g["state"].to_numpy(), g["flag"].to_numpy()))
        for method in KF_METHODS:
            if method == "quantreg":
                factor = np.exp(f["quantreg"].predict(_qr_design(g))) / CRIT95
            else:
                fallback = np.median(list(f[method].values()))
                factor = np.array([f[method].get(k, fallback) for k in keys])
            radius = np.sqrt(CRIT95 * var * factor)
            rows.append(g.assign(method=f"kf_{method}", radius=radius, covered=g["err"].to_numpy() <= radius))
    return pd.concat(rows, ignore_index=True)


def _halves(tracks):
    envs = sorted({tr["environment_id"] for tr in tracks})
    a = set(envs[::2])
    return [tr for tr in tracks if tr["environment_id"] in a], [tr for tr in tracks if tr["environment_id"] not in a]


def _residual_rows(tracks, model, h, cuts) -> pd.DataFrame:
    parts = []
    for tr in tracks:
        o, k = origins(tr, h)
        if o.size == 0:
            continue
        err = np.linalg.norm(model.predict(tr, o, h) - tr["points"]["wrist"][k], axis=1)
        parts.append(pd.DataFrame({"state": np.digitize(tr["speed5"][o], cuts), "flag": tr["recent_flag"][o],
                                   **{c: tr[c][o] for c in UNC_FEATURES if c not in ("state", "flag")}, "err": err}))
    return pd.concat(parts, ignore_index=True)


def boost_uncertainty_fit(fit_tracks, cuts, seed: int = 0) -> dict:
    a, b = _halves(fit_tracks)
    fitted = {}
    for h in HORIZONS:
        ma, mb = Learned("boosting").fit(a, h), Learned("boosting").fit(b, h)
        ra, rb = _residual_rows(a, mb, h, cuts), _residual_rows(b, ma, h, cuts)  # out-of-sample residuals
        both = pd.concat([ra, rb], ignore_index=True)
        flag_pool = {f: both[both.flag == f]["err"] for f in (False, True)}
        cells = {}
        for s in range(3):
            for f in (False, True):
                sel = both[(both.state == s) & (both.flag == f)]["err"]
                cells[(s, f)] = float(np.quantile(sel if sel.size >= 30 else flag_pool[f], 0.95))
        qmodel = HistGradientBoostingRegressor(loss="quantile", quantile=0.95, max_iter=200, learning_rate=0.05,
                                               min_samples_leaf=100, random_state=seed)
        qmodel.fit(ra[UNC_FEATURES].astype(float).to_numpy(), ra["err"].to_numpy())
        ratio = rb["err"].to_numpy() / np.maximum(qmodel.predict(rb[UNC_FEATURES].astype(float).to_numpy()), 1e-6)
        n = ratio.size
        level = min(1.0, math.ceil((n + 1) * 0.95) / n)
        fitted[h] = {"cells": cells, "quantile": qmodel, "scale": float(np.quantile(ratio, level))}
    return fitted


def boost_uncertainty_apply(bst: pd.DataFrame, fitted: dict) -> pd.DataFrame:
    rows = []
    for h, g in bst.groupby("horizon"):
        f = fitted[h]
        keys = list(zip(g["state"].to_numpy(), g["flag"].to_numpy()))
        r_cells = np.array([f["cells"][k] for k in keys])
        r_q = f["scale"] * f["quantile"].predict(g[UNC_FEATURES].astype(float).to_numpy())
        for method, radius in (("boost_cells", r_cells), ("boost_quantile", r_q)):
            rows.append(g.assign(method=method, radius=radius, covered=g["err"].to_numpy() <= radius))
    return pd.concat(rows, ignore_index=True)


# ------------------------------------------------------------ reliability

def reliability_table2(canonical: Path, tracks: list[dict]) -> pd.DataFrame:
    base = rel.build_features(canonical)
    parts = []
    for tr in tracks:
        n = len(tr["t"])
        cols = {"sample": tr["sample"], "hand": tr["hand"], "frame_index": np.arange(n)}
        cols.update({f: tr[f] for f in qi.HEAD_FEATURES + FRUSTUM_FEATURES})
        for kind in ("is_near_edge", "is_occlusion"):
            c = np.cumsum(np.r_[0.0, tr[kind].astype(float)])
            fut = np.full(n, np.nan)
            k = np.arange(n - rel.LOOKAHEAD)
            fut[k] = (c[k + 1 + rel.LOOKAHEAD] - c[k + 1]) > 0
            cols["future_" + kind[3:]] = fut
        parts.append(pd.DataFrame(cols))
    table = base.merge(pd.concat(parts, ignore_index=True), on=["sample", "hand", "frame_index"], how="inner")
    return table.dropna(subset=qi.HEAD_FEATURES + FRUSTUM_FEATURES).reset_index(drop=True)


FEATURE_SETS = {"head": rel.FEATURES + qi.HEAD_FEATURES, "frustum": rel.FEATURES + qi.HEAD_FEATURES + FRUSTUM_FEATURES}


def reliability_metrics(d: pd.DataFrame, score_cols: list[str]) -> pd.DataFrame:
    rows = []
    onset = (d["flags_last10"] == 0).to_numpy()
    for target in ("any", "occlusion", "near_edge"):
        if target == "any":
            y_all, valid = d["future_flag"].to_numpy(int), np.ones(len(d), bool)
        else:
            y_all = d[f"future_{target}"].to_numpy(int)
            valid = (d["future_flag"] == 0).to_numpy() | (y_all == 1)  # negatives: no flag of any kind
        for subset, mask in (("all", valid), ("onset", valid & onset)):
            y = y_all[mask]
            if y.sum() == 0 or y.sum() == len(y):
                continue
            for col in score_cols:
                p = d[col].to_numpy()[mask]
                rows.append({"target": target, "subset": subset, "model": col, "n": int(mask.sum()),
                             "positives": int(y.sum()), "auc": roc_auc_score(y, p),
                             "average_precision": average_precision_score(y, p)})
    return pd.DataFrame(rows)


def alarm_threshold(y: np.ndarray, p: np.ndarray) -> float:
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / np.maximum(prec[:-1] + rec[:-1], 1e-12)
    return float(thr[int(np.argmax(f1))])


def alarm_metrics(y: np.ndarray, p: np.ndarray, thr: float) -> dict:
    alarm = p >= thr
    tp, fp, fn = int((alarm & (y == 1)).sum()), int((alarm & (y == 0)).sum()), int((~alarm & (y == 1)).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {"threshold": thr, "alarms": int(alarm.sum()), "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(precision + recall, 1e-12)}


# ------------------------------------------------------------------ stages

def _coverage_tables(cal: pd.DataFrame):
    cells = (cal.groupby(["horizon", "method", "state", "flag"])
             .agg(n=("covered", "size"), coverage=("covered", "mean"), mean_radius_mm=("radius", "mean")).reset_index())
    overall = (cal.groupby(["horizon", "method"])
               .agg(n=("covered", "size"), coverage=("covered", "mean"), mean_radius_mm=("radius", "mean")).reset_index())
    cells["mean_radius_mm"] *= 1000
    overall["mean_radius_mm"] *= 1000
    return cells, overall


def _choose(cells: pd.DataFrame, overall: pd.DataFrame, methods: list[str]) -> tuple[str, dict]:
    worst = {m: float((cells[cells.method == m]["coverage"] - 0.95).abs().max()) for m in methods}
    best = min(worst.values())
    tied = [m for m in methods if worst[m] <= best + 0.005]
    rel_radius = {}
    for m in tied:
        ratios = []
        for h in HORIZONS:
            per_h = overall[(overall.horizon == h) & overall.method.isin(tied)].set_index("method")["mean_radius_mm"]
            ratios.append(per_h[m] / per_h.min())
        rel_radius[m] = float(np.mean(ratios))
    return min(tied, key=lambda m: rel_radius[m]), {"max_abs_deviation": worst, "relative_radius_among_tied": rel_radius}


def _severity_table(scores: pd.DataFrame) -> pd.DataFrame:
    t = scores.groupby(["horizon", "model", "sev"]).agg(n=("err", "size"), E_mean_mm=("err", "mean")).reset_index()
    t["E_mean_mm"] *= 1000
    return t


def stage_cv(canonical: Path, data_root: Path, out: Path, max_folds: int | None = None) -> dict:
    tracks = load2(canonical, data_root)
    dev = [tr for tr in tracks if tr["split"] in ("train", "validation")]
    envs = sorted({tr["environment_id"] for tr in dev})[:max_folds]
    cache = KalmanCache(dev)
    kf_cal, bst_cal, sev_parts = [], [], []
    for env in envs:
        fit_rows = np.array([i for i, tr in enumerate(dev) if tr["environment_id"] != env])
        fit_tracks = [dev[i] for i in fit_rows]
        held = [tr for tr in dev if tr["environment_id"] == env]
        params = cache.fit(fit_rows, dev)
        cuts = np.quantile(np.concatenate([tr["speed5"][np.isfinite(tr["speed5"])] for tr in fit_tracks]), [1 / 3, 2 / 3])
        boost = {h: Learned("boosting").fit(fit_tracks, h) for h in HORIZONS}
        train_kf = score2(fit_tracks, params, None, cache, cuts).query("model == 'cv_kalman'")
        held_scores = score2(held, params, boost, cache, cuts)
        sev_parts.append(held_scores.assign(fold=env))
        kf_cal.append(kf_calibration_apply(held_scores.query("model == 'cv_kalman'"), kf_calibration_fit(train_kf)))
        bst_cal.append(boost_uncertainty_apply(held_scores.query("model == 'boosting'"),
                                               boost_uncertainty_fit(fit_tracks, cuts)))
        print(f"fold {env}: done", flush=True)
    kf_cells, kf_overall = _coverage_tables(pd.concat(kf_cal, ignore_index=True))
    b_cells, b_overall = _coverage_tables(pd.concat(bst_cal, ignore_index=True))
    kf_choice, kf_why = _choose(kf_cells, kf_overall, [f"kf_{m}" for m in KF_METHODS])
    b_choice, b_why = _choose(b_cells, b_overall, [f"boost_{m}" for m in BOOST_METHODS])
    sev = _severity_table(pd.concat(sev_parts, ignore_index=True))
    # Reliability: head (round-1 adopted) vs head + frustum, out-of-fold over the development environments.
    table = reliability_table2(canonical, tracks)
    devt = table[table.split.isin(["train", "validation"])].copy()
    fit_pool = devt.copy()
    score_cols = []
    for fs, cols in FEATURE_SETS.items():
        for kind in ("logistic", "boosting"):
            name = f"{fs}_{kind}"
            score_cols.append(name)
            devt[name] = np.nan
            for env in envs:
                te = devt.environment_id == env
                devt.loc[te, name] = qi._fit_predict(fit_pool[fit_pool.environment_id != env], devt[te], cols, kind)
    devt = devt[devt.environment_id.isin(set(envs))].reset_index(drop=True)
    rel_cv = reliability_metrics(devt, score_cols)
    onset = (devt["flags_last10"] == 0).to_numpy()
    ci = {kind: qi.auc_by_bootstrap(devt, f"head_{kind}", f"frustum_{kind}", onset) for kind in ("logistic", "boosting")}
    ci_all = {kind: qi.auc_by_bootstrap(devt, f"head_{kind}", f"frustum_{kind}", np.ones(len(devt), bool))
              for kind in ("logistic", "boosting")}
    onset_auc = rel_cv[(rel_cv.target == "any") & (rel_cv.subset == "onset")].set_index("model")["auc"]
    adopt_frustum = all(onset_auc[f"frustum_{k}"] > onset_auc[f"head_{k}"] for k in ("logistic", "boosting"))
    fs_choice = "frustum" if adopt_frustum else "head"
    y = devt["future_flag"].to_numpy(int)
    thresholds, alarm_rows = {}, []
    for name in score_cols:
        thr = alarm_threshold(y, devt[name].to_numpy())
        thresholds[name] = thr
        for subset, mask in (("all", np.ones(len(devt), bool)), ("onset", onset)):
            alarm_rows.append({"model": name, "subset": subset, **alarm_metrics(y[mask], devt[name].to_numpy()[mask], thr)})
    decisions = {"reliability_features": fs_choice, "kf_calibration": kf_choice, "boost_uncertainty": b_choice,
                 "alarm_thresholds": thresholds, "onset_auc_gain_ci_frustum_minus_head": ci,
                 "all_auc_gain_ci_frustum_minus_head": ci_all, "kf_choice_detail": kf_why, "boost_choice_detail": b_why,
                 "rules": {"R1": "frustum if onset AUC improves for both logistic and boosting",
                           "R2/R3": "smallest max |coverage-0.95| over 24 cells; ties within 0.005 -> smaller radius",
                           "R4": "F1-maximising threshold on development out-of-fold predictions"}}
    out.mkdir(parents=True, exist_ok=True)
    for name, frame in {"cv2_kf_calibration_cells": kf_cells, "cv2_kf_calibration_overall": kf_overall,
                        "cv2_boost_uncertainty_cells": b_cells, "cv2_boost_uncertainty_overall": b_overall,
                        "cv2_severity_errors": sev, "cv2_reliability": rel_cv,
                        "cv2_alarm_metrics": pd.DataFrame(alarm_rows)}.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    (out / "cv2_decisions.json").write_text(json.dumps(decisions, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    return decisions


def stage_final(canonical: Path, data_root: Path, out: Path) -> dict:
    dec = json.loads((out / "cv2_decisions.json").read_text(encoding="utf-8"))
    tracks = load2(canonical, data_root)
    train = [tr for tr in tracks if tr["split"] == "train"]
    test = [tr for tr in tracks if tr["split"] == "test"]
    cache = KalmanCache(train)
    params = cache.fit(np.arange(len(train)), train)
    cuts = np.quantile(np.concatenate([tr["speed5"][np.isfinite(tr["speed5"])] for tr in train]), [1 / 3, 2 / 3])
    boost = {h: Learned("boosting").fit(train, h) for h in HORIZONS}
    train_kf = score2(train, params, None, cache, cuts).query("model == 'cv_kalman'")
    test_scores = score2(test, params, boost, cache, cuts)
    kf_cells, kf_overall = _coverage_tables(kf_calibration_apply(test_scores.query("model == 'cv_kalman'"),
                                                                 kf_calibration_fit(train_kf)))
    b_cells, b_overall = _coverage_tables(boost_uncertainty_apply(test_scores.query("model == 'boosting'"),
                                                                  boost_uncertainty_fit(train, cuts)))
    sev = _severity_table(test_scores)
    table = reliability_table2(canonical, tracks)
    tr_t, te_t = table[table.split == "train"], table[table.split == "test"].copy()
    names = []
    for fs, cols in FEATURE_SETS.items():
        for kind in ("logistic", "boosting"):
            te_t[f"{fs}_{kind}"] = qi._fit_predict(tr_t, te_t, cols, kind)
            names.append(f"{fs}_{kind}")
    rel_test = reliability_metrics(te_t, names)
    y = te_t["future_flag"].to_numpy(int)
    onset = (te_t["flags_last10"] == 0).to_numpy()
    alarms = [{"model": n, "subset": s, **alarm_metrics(y[m], te_t[n].to_numpy()[m], dec["alarm_thresholds"][n])}
              for n in names for s, m in (("all", np.ones(len(te_t), bool)), ("onset", onset))]
    fs_cols = FEATURE_SETS[dec["reliability_features"]]
    mean, std = tr_t[fs_cols].mean(), tr_t[fs_cols].std().replace(0, 1.0)
    from sklearn.linear_model import LogisticRegression
    lr = LogisticRegression(max_iter=3000).fit(((tr_t[fs_cols] - mean) / std).to_numpy(), tr_t["future_flag"].to_numpy(int))
    names_zh = {**rel.FEATURE_NAMES, **qi.HEAD_NAMES, **FRUSTUM_NAMES}
    coef = pd.DataFrame({"feature": fs_cols, "name_zh": [names_zh.get(c, c) for c in fs_cols],
                         "standardized_coef": lr.coef_[0]}).sort_values("standardized_coef")
    for name, frame in {"final2_kf_calibration_cells": kf_cells, "final2_kf_calibration_overall": kf_overall,
                        "final2_boost_uncertainty_cells": b_cells, "final2_boost_uncertainty_overall": b_overall,
                        "final2_severity_errors": sev, "final2_reliability": rel_test,
                        "final2_alarm_metrics": pd.DataFrame(alarms), "final2_reliability_coefficients": coef}.items():
        frame.to_csv(out / f"{name}.csv", index=False)
    return {"decisions": dec}


def write_summary(out: Path) -> Path:
    dec = json.loads((out / "cv2_decisions.json").read_text(encoding="utf-8"))
    L = ["# 第一问第二轮改进（文献驱动）：留一环境交叉验证与一次性测试", "",
         "规则在交叉验证之前写定（见 q1_improve2.py 文件头）；最终模型只用训练集拟合。", "",
         "## 规则给出的选择", "",
         f"- 可靠性特征：{dec['reliability_features']}（R1：两种模型的新遮挡 AUC 都提升才采用视野特征；"
         f"增益区间 逻辑回归 [{dec['onset_auc_gain_ci_frustum_minus_head']['logistic'][0]:+.3f}, "
         f"{dec['onset_auc_gain_ci_frustum_minus_head']['logistic'][1]:+.3f}]，梯度提升 "
         f"[{dec['onset_auc_gain_ci_frustum_minus_head']['boosting'][0]:+.3f}, "
         f"{dec['onset_auc_gain_ci_frustum_minus_head']['boosting'][1]:+.3f}]）",
         f"- 卡尔曼方差校准：{dec['kf_calibration']}（各方法 24 格最大偏离：{ {k: round(v, 3) for k, v in dec['kf_choice_detail']['max_abs_deviation'].items()} }）",
         f"- 学习预测器不确定范围：{dec['boost_uncertainty']}（{ {k: round(v, 3) for k, v in dec['boost_choice_detail']['max_abs_deviation'].items()} }）", ""]
    for stage, prefix in (("交叉验证（20 个开发环境）", "cv2"), ("测试集（一次性）", "final2")):
        if not (out / f"{prefix}_reliability.csv").is_file():
            continue
        r = pd.read_csv(out / f"{prefix}_reliability.csv")
        L += [f"## 可靠性模型：{stage}", "", "| 目标 | 子集 | 模型 | 样本 | 正例 | AUC | AP |", "|---|---|---|---:|---:|---:|---:|"]
        L += [f"| {x.target} | {x.subset} | {x.model} | {x.n} | {x.positives} | {x.auc:.3f} | {x.average_precision:.3f} |"
              for x in r.itertuples()]
        a = pd.read_csv(out / f"{prefix}_alarm_metrics.csv")
        L += ["", f"报警指标（阈值在开发集留出预测上按 F1 最大选定）：", "",
              "| 模型 | 子集 | 阈值 | 报警次数 | 精确率 | 召回率 | F1 |", "|---|---|---:|---:|---:|---:|---:|"]
        L += [f"| {x.model} | {x.subset} | {x.threshold:.3f} | {x.alarms} | {x.precision:.3f} | {x.recall:.3f} | {x.f1:.3f} |"
              for x in a.itertuples()]
        for label, file in (("卡尔曼方差校准", "kf_calibration_cells"), ("学习预测器不确定范围", "boost_uncertainty_cells")):
            c = pd.read_csv(out / f"{prefix}_{file}.csv")
            piv = c.pivot_table(index=["horizon", "state", "flag"], columns="method", values="coverage")
            L += ["", f"{label}：各格 95% 覆盖率（{stage}）", "", "| 步长 | 状态 | 近期标记 | " + " | ".join(piv.columns) + " |",
                  "|---:|---:|---|" + "---:|" * len(piv.columns)]
            L += [f"| {h} | {s} | {f} | " + " | ".join(f"{v:.3f}" for v in row) + " |" for (h, s, f), row in piv.iterrows()]
            o = pd.read_csv(out / f"{prefix}_{file.replace('cells', 'overall')}.csv")
            L += ["", "| 步长 | 方法 | 总覆盖率 | 平均半径 mm |", "|---:|---|---:|---:|"]
            L += [f"| {x.horizon} | {x.method} | {x.coverage:.3f} | {x.mean_radius_mm:.1f} |" for x in o.itertuples()]
        s = pd.read_csv(out / f"{prefix}_severity_errors.csv")
        piv = s.pivot_table(index=["horizon", "sev"], columns="model", values="E_mean_mm")
        cnt = s[s.model == "cv_kalman"].set_index(["horizon", "sev"])["n"]
        L += ["", f"按遮挡严重程度的预测误差 E（mm，{stage}）", "",
              "| 步长 | 严重程度 | 样本 | " + " | ".join(piv.columns) + " |", "|---:|---|---:|" + "---:|" * len(piv.columns)]
        L += [f"| {h} | {v} | {cnt.get((h, v), 0)} | " + " | ".join(f"{x:.2f}" for x in row) + " |"
              for (h, v), row in piv.iterrows()]
        L.append("")
    L += ["严重程度：clean 为前 10 帧无标记；recovering 为标记已结束不超过 10 帧；flag_run_k 为预测起点处已连续标记 k 帧。",
          "区间按环境重抽样；相邻帧相关，样本数不是独立样本数。", ""]
    path = out / "summary_improve2.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path


def main(argv=None) -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=("cv", "final", "summary"))
    parser.add_argument("--canonical", type=Path, default=here / "cache" / "canonical")
    parser.add_argument("--data", type=Path, default=here.parent / "Restocking_Problem_and_Data_EN")
    parser.add_argument("--out", type=Path, default=here / "results" / "q1_real")
    parser.add_argument("--max-folds", type=int, default=None, help="smoke test only")
    args = parser.parse_args(argv)
    if args.stage == "cv":
        print(json.dumps(stage_cv(args.canonical, args.data, args.out, args.max_folds), ensure_ascii=False, indent=2,
                         default=str)[:3000])
    elif args.stage == "final":
        stage_final(args.canonical, args.data, args.out)
    print(write_summary(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
