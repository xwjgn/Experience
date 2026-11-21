#!/usr/bin/env python3
"""Problem 1, second part: observation-reliability and operational (motion) state models.

Reads trajectory_adapter output and the fitted parameters written by q1_state.py.

Observation reliability: at frame k, predict whether the hand will be flagged
(occlusion or near edge) in any of frames k+1..k+10 (about 0.33 s), using only
information available at k: smoothed wrist speed and acceleration, wrist position
relative to the head, distance between hands, current and recent flags, hand side
and data source. Logistic regression (interpretable) and gradient boosting
(non-linear check) are fitted on TRAIN; validation and test are reported, overall
and on "onset" frames with no flag in the last 10 frames (a new occlusion).

Motion states: slow / medium / fast by 5-frame mean wrist speed, cut at the
TRAIN tertiles. Reported per state: share of frames, current and upcoming flag
rate, prediction error of the Q1 predictors at each horizon, and a 3-state Markov
transition model over 0.1 s with mean dwell times (train vs test).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

import q1_state as q1

LABEL = "REAL_DATA_Q1_RELIABILITY_AND_MOTION_STATES"
LOOKAHEAD = 10          # frames, about 0.33 s
SMOOTH = 5              # frames for speed / acceleration smoothing
TRANSITION_STEP = 3     # frames, about 0.1 s
STATES = ("slow", "medium", "fast")
FEATURES = ["speed5", "acc5", "dist_head", "dz_head", "hand_gap", "flag_now", "flags_last10",
            "since_last_flag", "is_right", "src_EgoPro", "src_EgoStandard"]
FEATURE_NAMES = {"speed5": "腕速（5 帧均值）", "acc5": "腕部加速度（5 帧均值）", "dist_head": "腕到头距离",
                 "dz_head": "腕相对头的高度", "hand_gap": "两腕距离", "flag_now": "当前帧已标记",
                 "flags_last10": "近 10 帧标记数", "since_last_flag": "距上次标记帧数", "is_right": "右手",
                 "src_EgoPro": "来源 EgoPro", "src_EgoStandard": "来源 EgoStandard"}


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Trailing mean of the last `window` values; NaN unless all of them are finite."""
    return pd.Series(values, dtype=float).rolling(window, min_periods=window).mean().to_numpy()


def build_features(canonical: Path) -> pd.DataFrame:
    frames = pd.read_parquet(canonical / "frames.parquet").sort_values(["sample", "frame_index"])
    parts = []
    for sample, g in frames.groupby("sample", sort=False):
        t = g["t_s"].to_numpy(float)
        head = g[["head_x", "head_y", "head_z"]].to_numpy(float)
        wrists = {h: g[[f"{h}_j00_{a}" for a in "xyz"]].to_numpy(float) for h in q1.HANDS}
        for hand in q1.HANDS:
            other = "right" if hand == "left" else "left"
            w = wrists[hand]
            flagged = (g[f"{hand}_quality"] != "ok").to_numpy()
            dt = np.r_[np.nan, np.diff(t)]
            vel = np.vstack([np.full((1, 3), np.nan), np.diff(w, axis=0)]) / dt[:, None]
            speed = np.linalg.norm(vel, axis=1)
            acc = np.r_[np.nan, np.linalg.norm(np.diff(vel, axis=0), axis=1)] / dt
            speed5 = _rolling_mean(speed, SMOOTH)
            acc5 = _rolling_mean(acc, SMOOTH)
            flags = flagged.astype(float)
            last10 = np.convolve(flags, np.ones(LOOKAHEAD))[:flags.size]
            since = np.empty(flags.size)
            counter = 30.0
            for i, f in enumerate(flagged):
                counter = 0.0 if f else min(counter + 1.0, 30.0)
                since[i] = counter
            future = np.full(flags.size, np.nan)
            c = np.cumsum(np.r_[0.0, flags])
            k = np.arange(flags.size - LOOKAHEAD)
            future[k] = (c[k + 1 + LOOKAHEAD] - c[k + 1]) > 0
            rel = w - head
            parts.append(pd.DataFrame({
                "sample": sample, "hand": hand, "dataset": g["dataset"].iloc[0], "split": g["split"].iloc[0],
                "environment_id": g["environment_id"].iloc[0], "frame_index": g["frame_index"].to_numpy(),
                "speed5": speed5, "acc5": acc5, "dist_head": np.linalg.norm(rel, axis=1), "dz_head": rel[:, 2],
                "hand_gap": np.linalg.norm(w - wrists[other], axis=1), "flag_now": flags, "flags_last10": last10,
                "since_last_flag": since, "is_right": float(hand == "right"),
                "src_EgoPro": float(g["dataset"].iloc[0] == "EgoPro"),
                "src_EgoStandard": float(g["dataset"].iloc[0] == "EgoStandard"), "future_flag": future,
            }))
    data = pd.concat(parts, ignore_index=True)
    return data.dropna(subset=["speed5", "acc5", "future_flag", "dist_head", "dz_head", "hand_gap"]).reset_index(drop=True)


def reliability_models(data: pd.DataFrame) -> dict:
    train = data[data.split == "train"]
    mean, std = train[FEATURES].mean(), train[FEATURES].std().replace(0, 1.0)
    z = lambda d, cols: ((d[cols] - mean[cols]) / std[cols]).to_numpy()
    y_train = train["future_flag"].to_numpy(int)
    recent = ["flag_now", "flags_last10", "since_last_flag"]
    models = {
        "recent_flags_logistic": (LogisticRegression(max_iter=2000).fit(z(train, recent), y_train), recent),
        "full_logistic": (LogisticRegression(max_iter=2000).fit(z(train, FEATURES), y_train), FEATURES),
        "gradient_boosting": (HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=0)
                              .fit(z(train, FEATURES), y_train), FEATURES),
    }
    prevalence = float(y_train.mean())
    rows = []
    for split in ("train", "validation", "test"):
        part = data[data.split == split]
        for subset, sel in (("all", np.ones(len(part), bool)), ("onset", (part["flags_last10"] == 0).to_numpy())):
            d = part[sel]
            y = d["future_flag"].to_numpy(int)
            if y.sum() == 0 or y.sum() == len(y):
                continue
            preds = {"base_rate": np.full(len(d), prevalence)}
            for name, (model, cols) in models.items():
                preds[name] = model.predict_proba(z(d, cols))[:, 1]
            for name, p in preds.items():
                rows.append({"split": split, "subset": subset, "model": name, "n": len(d), "positives": int(y.sum()),
                             "prevalence": float(y.mean()), "auc": float(roc_auc_score(y, p)),
                             "average_precision": float(average_precision_score(y, p)),
                             "brier": float(brier_score_loss(y, p))})
    logit = models["full_logistic"][0]
    coefficients = pd.DataFrame({"feature": FEATURES, "name_zh": [FEATURE_NAMES[f] for f in FEATURES],
                                 "standardized_coef": logit.coef_[0],
                                 "odds_ratio_per_sd": np.exp(logit.coef_[0])}).sort_values("standardized_coef")
    return {"metrics": pd.DataFrame(rows), "coefficients": coefficients, "train_prevalence": prevalence,
            "standardization": {"mean": mean.to_dict(), "std": std.to_dict()}}


def motion_states(data: pd.DataFrame) -> dict:
    cuts = np.quantile(data.loc[data.split == "train", "speed5"], [1 / 3, 2 / 3])
    data = data.assign(state=pd.cut(data["speed5"], [-np.inf, cuts[0], cuts[1], np.inf], labels=STATES).astype(str))
    summary = (data.groupby(["split", "state"])
               .agg(frames=("speed5", "size"), flag_rate=("flag_now", "mean"), upcoming_flag_rate=("future_flag", "mean"),
                    median_speed=("speed5", "median"))
               .reset_index())
    summary["share"] = summary["frames"] / summary.groupby("split")["frames"].transform("sum")
    transitions, dwell = [], []
    for split in ("train", "test"):
        counts = np.zeros((3, 3))
        runs = {s: [] for s in STATES}
        for _, g in data[data.split == split].groupby(["sample", "hand"], sort=False):
            s = g["state"].map({k: i for i, k in enumerate(STATES)}).to_numpy()
            frame = g["frame_index"].to_numpy()
            contiguous = frame[TRANSITION_STEP:] - frame[:-TRANSITION_STEP] == TRANSITION_STEP
            np.add.at(counts, (s[:-TRANSITION_STEP][contiguous], s[TRANSITION_STEP:][contiguous]), 1)
            start = 0
            for i in range(1, len(s) + 1):
                if i == len(s) or s[i] != s[start] or frame[i] != frame[i - 1] + 1:
                    runs[STATES[s[start]]].append((i - start) / 30.0)
                    start = i
        totals = counts.sum(axis=1, keepdims=True)
        probs = np.divide(counts, totals, out=np.full_like(counts, np.nan), where=totals > 0)  # NaN if state unseen
        for i, a in enumerate(STATES):
            transitions.append({"split": split, "from": a, **{f"to_{b}": probs[i, j] for j, b in enumerate(STATES)}})
            dwell.append({"split": split, "state": a, "runs": len(runs[a]),
                          "mean_dwell_s": float(np.mean(runs[a])) if runs[a] else np.nan,
                          "median_dwell_s": float(np.median(runs[a])) if runs[a] else np.nan})
    return {"cuts_mps": cuts.tolist(), "data": data, "summary": summary,
            "transitions": pd.DataFrame(transitions), "dwell": pd.DataFrame(dwell)}


def errors_by_state(canonical: Path, data_root: Path, fitted: dict, cuts) -> pd.DataFrame:
    tracks, _ = q1.load_tracks(canonical, data_root)
    q1.qualify(tracks)
    params = fitted["wrist"]
    crit = chi2.ppf(0.95, 3)
    rows = []
    for tr in tracks:
        z = tr["points"]["wrist"]
        speed = np.r_[np.nan, np.linalg.norm(np.diff(z, axis=0), axis=1) / np.diff(tr["t"])]
        speed5 = _rolling_mean(speed, SMOOTH)
        state = np.where(speed5 < cuts[0], 0, np.where(speed5 < cuts[1], 1, 2))
        valid_state = ~np.isnan(speed5)
        for h, item in q1.predictions(tr, "wrist", params).items():
            k, o = item["target"], item["origin"]
            keep = tr["qualified"][k] & valid_state[o]
            for model in ("hold", "observed_cv", "cv_kalman", "damped_kalman"):
                err = np.linalg.norm(item["pred"][model][keep] - z[k[keep]], axis=1)
                cover = (err ** 2 / item["var"]["cv_kalman"][keep] <= crit) if model == "cv_kalman" else None
                for si, name in enumerate(STATES):
                    sel = state[o[keep]] == si
                    if sel.any():
                        rows.append({"split": tr["split"], "state": name, "model": model, "horizon": h,
                                     "n": int(sel.sum()), "sum_err": float(err[sel].sum()),
                                     "covered": float(cover[sel].sum()) if cover is not None else np.nan})
    raw = pd.DataFrame(rows)
    out = raw.groupby(["split", "state", "model", "horizon"]).agg(n=("n", "sum"), sum_err=("sum_err", "sum"),
                                                                  covered=("covered", "sum")).reset_index()
    out["E_mean_mm"] = 1000 * out["sum_err"] / out["n"]
    out["coverage95"] = np.where(out["model"] == "cv_kalman", out["covered"] / out["n"], np.nan)
    return out.drop(columns=["sum_err", "covered"])


def markdown(rel: dict, ms: dict, err: pd.DataFrame) -> str:
    lines = ["# 第一问补充：观测可靠性模型与运动状态", "",
             f"目标：在第 k 帧预测第 k+1 至 k+{LOOKAHEAD} 帧（约 0.33 s）内是否出现遮挡或靠边标记，只用第 k 帧及之前的信息。"
             f"模型只在训练集拟合（训练集正例比例 {rel['train_prevalence']:.2%}）。onset 子集为近 10 帧没有任何标记的帧，即预测“新的遮挡”。", "",
             "## 1 可靠性预测效果", "", "| 划分 | 子集 | 模型 | 样本数 | 正例 | AUC | 平均精度 AP | Brier |", "|---|---|---|---:|---:|---:|---:|---:|"]
    m = rel["metrics"]
    for _, r in m[m.split.isin(["validation", "test"])].iterrows():
        lines.append(f"| {r.split} | {r.subset} | {r.model} | {r.n} | {r.positives} | {r.auc:.3f} | {r.average_precision:.3f} | {r.brier:.4f} |")
    lines += ["", "AP 的随机基线等于正例比例（prevalence）。", "", "## 2 哪些因素让遮挡更可能（逻辑回归标准化系数）", "",
              "| 因素 | 标准化系数 | 每增加 1 个标准差的优势比 |", "|---|---:|---:|"]
    for _, r in rel["coefficients"].sort_values("standardized_coef", ascending=False).iterrows():
        lines.append(f"| {r.name_zh} | {r.standardized_coef:+.3f} | {r.odds_ratio_per_sd:.2f} |")
    cuts = ms["cuts_mps"]
    lines += ["", "## 3 运动状态", "",
              f"按 5 帧平均腕速划分，阈值取训练集三分位数：慢 < {cuts[0]:.3f} m/s ≤ 中 < {cuts[1]:.3f} m/s ≤ 快。", "",
              "| 划分 | 状态 | 帧占比 | 中位腕速 m/s | 当前标记率 | 未来 0.33 s 内标记率 |", "|---|---|---:|---:|---:|---:|"]
    for _, r in ms["summary"].iterrows():
        lines.append(f"| {r.split} | {r.state} | {r.share:.1%} | {r.median_speed:.3f} | {r.flag_rate:.2%} | {r.upcoming_flag_rate:.2%} |")
    lines += ["", "状态转移（0.1 s 一步）与平均停留时间：", "", "| 划分 | 从 | 到 slow | 到 medium | 到 fast | 平均停留 s |", "|---|---|---:|---:|---:|---:|"]
    dwell = ms["dwell"].set_index(["split", "state"])
    for _, r in ms["transitions"].iterrows():
        lines.append(f"| {r.split} | {r['from']} | {r.to_slow:.3f} | {r.to_medium:.3f} | {r.to_fast:.3f} | {dwell.loc[(r.split, r['from']), 'mean_dwell_s']:.2f} |")
    lines += ["", "## 4 各运动状态下的预测误差（测试集，腕点，E 均值 mm；括号内为常速度卡尔曼 95% 覆盖率）", "",
              "| 步长 | 状态 | hold | observed_cv | cv_kalman | damped_kalman |", "|---:|---|---:|---:|---:|---:|"]
    e = err[err.split == "test"]
    for h in q1.HORIZONS:
        for s in STATES:
            sel = e[(e.horizon == h) & (e.state == s)].set_index("model")
            if len(sel):
                cov = sel.loc["cv_kalman", "coverage95"]
                lines.append(f"| {h} | {s} | {sel.loc['hold', 'E_mean_mm']:.2f} | {sel.loc['observed_cv', 'E_mean_mm']:.2f} | "
                             f"{sel.loc['cv_kalman', 'E_mean_mm']:.2f} ({cov:.3f}) | {sel.loc['damped_kalman', 'E_mean_mm']:.2f} |")
    lines += ["", "测试集与验证集环境少、正例少，结论按经验检验表述；相邻帧高度相关，样本数不等于独立样本数。", ""]
    return "\n".join(lines)


def run(canonical: Path, data_root: Path, results: Path) -> dict:
    fitted = json.loads((results / "summary.json").read_text(encoding="utf-8"))["fitted_parameters"]
    data = build_features(canonical)
    rel = reliability_models(data)
    ms = motion_states(data)
    err = errors_by_state(canonical, data_root, fitted, ms["cuts_mps"])
    rel["metrics"].to_csv(results / "reliability_metrics.csv", index=False)
    rel["coefficients"].to_csv(results / "reliability_coefficients.csv", index=False)
    ms["summary"].to_csv(results / "motion_state_summary.csv", index=False)
    ms["transitions"].to_csv(results / "motion_state_transitions.csv", index=False)
    ms["dwell"].to_csv(results / "motion_state_dwell.csv", index=False)
    err.to_csv(results / "motion_state_errors.csv", index=False)
    meta = {"label": LABEL, "lookahead_frames": LOOKAHEAD, "smoothing_frames": SMOOTH, "features": FEATURES,
            "train_prevalence": rel["train_prevalence"], "speed_cuts_mps": ms["cuts_mps"],
            "standardization": rel["standardization"],
            "information_rules": ["features use frames <= k only", "models, standardisation and speed cuts fitted on train only"]}
    (results / "reliability_summary.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    (results / "summary_reliability.md").write_text(markdown(rel, ms, err), encoding="utf-8")
    return {"reliability": rel, "motion": ms, "errors": err, "data": data}


def main(argv=None) -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--canonical", type=Path, default=here / "cache" / "canonical")
    parser.add_argument("--data", type=Path, default=here.parent / "Restocking_Problem_and_Data_EN")
    parser.add_argument("--results", type=Path, default=here / "results" / "q1_real")
    args = parser.parse_args(argv)
    run(args.canonical, args.data, args.results)
    print(f"Summary: {(args.results / 'summary_reliability.md').resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
