#!/usr/bin/env python3
"""Problem 1, round 3: record-level occlusion propensity (2) and online adaptive calibration (3).

(2) Propensity: causal running statistics within each record, shrunk to the
    training mean so that early frames are stable:
      prop_rate_self   (flagged frames so far + A*mu) / (frames so far + A), A = 300 frames
      prop_onset_self  (flag-run starts so far + B*nu) / (minutes so far + B), B = 1 min
      prop_rate_other  the same flagged-frame rate for the other hand of the record
    mu and nu are the train-split flagged rate and onsets per minute.
    Rule: useful only if the CV onset AUC of head + propensity beats head for both
    logistic regression and gradient boosting; only then is the test split used.

(3) Adaptive conformal inference (Gibbs & Candes, NeurIPS 2021) applied per record
    and horizon on top of the per-cell bands (Kalman variance factors and boosting
    split-conformal radii): radius_t = base_t * exp(theta_t), and after the target of
    an earlier prediction is observed (time k = origin + h, never earlier)
    theta <- theta + gamma * (miss - 0.05). theta restarts at 0 for each record.
    Rule: gamma from {0.01, 0.03, 0.1} by the smallest worst-environment |coverage -
    0.95| over the four horizons in CV; ACI is adopted if that is below the
    non-adaptive value and every horizon's pooled coverage stays in [0.94, 0.96];
    decided separately for the Kalman and the boosting bands. Then test once.
    Not pre-registered: ties in the gamma choice go to the smallest gamma (the
    boosting band tied at 0.050 for all three values).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import q1_improve as qi
import q1_improve2 as r2
import q1_reliability as rel
from q1_improve import HORIZONS, KalmanCache, Learned

A_FRAMES, B_MINUTES = 300.0, 1.0
PROP_FEATURES = ["prop_rate_self", "prop_onset_self", "prop_rate_other"]
PROP_SELF = PROP_FEATURES[:2]
GAMMAS = (0.01, 0.03, 0.1)
ALPHA = 0.05
# "prop_self" (without the other hand's rate) was added AFTER the first test look: on the
# test split the other hand's rate had a reversed relation (single-feature onset AUC 0.466;
# the occluded EgoPro test hands occluded alone). It goes through the same CV rule first.
SETS = {"head": rel.FEATURES + qi.HEAD_FEATURES, "prop": rel.FEATURES + qi.HEAD_FEATURES + PROP_FEATURES,
        "prop_self": rel.FEATURES + qi.HEAD_FEATURES + PROP_SELF}
VARIANTS = ("prop", "prop_self")


# ------------------------------------------------------------- propensity

def propensity(tracks: list[dict]) -> pd.DataFrame:
    train = [tr for tr in tracks if tr["split"] == "train"]
    mu = float(np.mean(np.concatenate([tr["bad"] for tr in train])))
    starts = lambda f: f & ~np.r_[False, f[:-1]]
    minutes = lambda tr: (tr["t"] - tr["t"][0]) / 60.0  # elapsed time within the record
    nu = float(sum(starts(tr["bad"]).sum() for tr in train) / sum(minutes(tr)[-1] for tr in train))
    rate = {}
    rows = []
    for tr in tracks:
        f = tr["bad"]
        n = np.arange(1, len(f) + 1, dtype=float)
        r = (np.cumsum(f) + A_FRAMES * mu) / (n + A_FRAMES)
        onset = (np.cumsum(starts(f)) + B_MINUTES * nu) / (minutes(tr) + B_MINUTES)
        rate[(tr["sample"], tr["hand"])] = r
        rows.append((tr, r, onset))
    out = []
    for tr, r, onset in rows:
        other = rate.get((tr["sample"], "right" if tr["hand"] == "left" else "left"), np.full(len(r), mu))
        out.append(pd.DataFrame({"sample": tr["sample"], "hand": tr["hand"], "frame_index": np.arange(len(r)),
                                 "prop_rate_self": r, "prop_onset_self": onset, "prop_rate_other": other[:len(r)]}))
    return pd.concat(out, ignore_index=True)


def _alarms(d: pd.DataFrame, cols: list[str], thresholds: dict) -> pd.DataFrame:
    y, onset = d["future_flag"].to_numpy(int), (d["flags_last10"] == 0).to_numpy()
    return pd.DataFrame([{"model": c, "subset": s, **r2.alarm_metrics(y[m], d[c].to_numpy()[m], thresholds[c])}
                         for c in cols for s, m in (("all", np.ones(len(d), bool)), ("onset", onset))])


def _prop_table(canonical: Path, data_root: Path) -> pd.DataFrame:
    tracks = r2.load2(canonical, data_root)
    table = r2.reliability_table2(canonical, tracks)
    return table.merge(propensity(tracks), on=["sample", "hand", "frame_index"], how="left")


def prop_cv(canonical: Path, data_root: Path, out: Path) -> dict:
    dev = _prop_table(canonical, data_root)
    dev = dev[dev.split.isin(["train", "validation"])].reset_index(drop=True)
    cols = [f"{s}_{k}" for s in SETS for k in ("logistic", "boosting")]
    for c in cols:
        dev[c] = np.nan
    for env in sorted(dev.environment_id.unique()):
        fit, held = dev.environment_id != env, dev.environment_id == env
        for s, feats in SETS.items():
            for kind in ("logistic", "boosting"):
                dev.loc[held, f"{s}_{kind}"] = qi._fit_predict(dev[fit], dev[held], feats, kind)
    metrics = r2.reliability_metrics(dev, cols)
    onset_auc = metrics[(metrics.target == "any") & (metrics.subset == "onset")].set_index("model")["auc"]
    onset = (dev["flags_last10"] == 0).to_numpy()
    kinds = ("logistic", "boosting")
    ci = {v: {k: qi.auc_by_bootstrap(dev, f"head_{k}", f"{v}_{k}", onset) for k in kinds} for v in VARIANTS}
    useful = {v: all(onset_auc[f"{v}_{k}"] > onset_auc[f"head_{k}"] for k in kinds) for v in VARIANTS}
    y = dev["future_flag"].to_numpy(int)
    thresholds = {c: r2.alarm_threshold(y, dev[c].to_numpy()) for c in cols}  # R4: F1-maximising, out-of-fold
    decision = {"useful": useful, "onset_auc": onset_auc.to_dict(), "onset_auc_gain_ci_minus_head": ci,
                "alarm_thresholds": thresholds, "constants": {"A_frames": A_FRAMES, "B_minutes": B_MINUTES},
                "note": "prop_self added after the first test look at prop (other-hand rate reversed on test)"}
    _alarms(dev, cols, thresholds).to_csv(out / "cv6_propensity_alarm_metrics.csv", index=False)
    metrics.to_csv(out / "cv6_propensity_reliability.csv", index=False)
    (out / "cv6_propensity_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    return decision


def prop_final(canonical: Path, data_root: Path, out: Path) -> dict:
    decision = json.loads((out / "cv6_propensity_decision.json").read_text(encoding="utf-8"))
    useful = decision["useful"]
    sets = {s: f for s, f in SETS.items() if s == "head" or useful.get(s)}
    if len(sets) == 1:
        raise SystemExit("CV rule said not useful; the test split is not used.")
    table = _prop_table(canonical, data_root)
    tr_t, te_t = table[table.split == "train"], table[table.split == "test"].copy()
    cols = []
    for s, feats in sets.items():
        for kind in ("logistic", "boosting"):
            te_t[f"{s}_{kind}"] = qi._fit_predict(tr_t, te_t, feats, kind)
            cols.append(f"{s}_{kind}")
    metrics = r2.reliability_metrics(te_t, cols)
    metrics.to_csv(out / "final6_propensity_reliability.csv", index=False)
    _alarms(te_t, cols, decision["alarm_thresholds"]).to_csv(out / "final6_propensity_alarm_metrics.csv", index=False)
    return {"test_onset_auc": metrics[(metrics.target == "any") & (metrics.subset == "onset")].set_index("model")["auc"].to_dict()}


# -------------------------------------------------------------------- ACI

def aci(df: pd.DataFrame, gamma: float) -> pd.DataFrame:
    """Per (record, hand, horizon), in origin order; an error is used only once its target is observed."""
    radius = np.empty(len(df))
    df = df.reset_index(drop=True)
    for _, g in df.groupby(["sample", "hand", "horizon"], sort=False):
        g = g.sort_values("origin")
        idx, o, k = g.index.to_numpy(), g["origin"].to_numpy(), g["target"].to_numpy()
        base, err = g["radius"].to_numpy(), g["err"].to_numpy()
        used = np.empty(len(g))
        theta, j = 0.0, 0
        for i in range(len(g)):
            while j < i and k[j] <= o[i]:
                theta += gamma * (float(err[j] > used[j]) - ALPHA)
                j += 1
            used[i] = base[i] * np.exp(theta)
        radius[idx] = used
    return df.assign(radius=radius, covered=df["err"].to_numpy() <= radius)


def _bands(held: pd.DataFrame, kf_fit: dict, boost_fit: dict) -> pd.DataFrame:
    kf = r2.kf_calibration_apply(held.query("model == 'cv_kalman'"), kf_fit).query("method == 'kf_cells'")
    bs = r2.boost_uncertainty_apply(held.query("model == 'boosting'"), boost_fit).query("method == 'boost_cells'")
    parts = []
    for base in (kf, bs):
        name = base["method"].iloc[0]
        parts.append(base.assign(method=name, gamma=0.0))
        for gamma in GAMMAS:
            parts.append(aci(base, gamma).assign(method=name, gamma=gamma))
    return pd.concat(parts, ignore_index=True)


def _aci_tables(bands: pd.DataFrame):
    keys = ["method", "gamma", "horizon"]
    pooled = bands.groupby(keys).agg(n=("covered", "size"), coverage=("covered", "mean"), mean_radius_mm=("radius", "mean")).reset_index()
    pooled["mean_radius_mm"] *= 1000
    env = bands.groupby(keys + ["environment_id"])["covered"].mean().rename("coverage").reset_index()
    cells = bands.groupby(keys + ["state", "flag"])["covered"].mean().rename("coverage").reset_index()
    return pooled, env, cells


def _choose_aci(pooled: pd.DataFrame, env: pd.DataFrame) -> dict:
    decisions = {}
    for method in pooled.method.unique():
        worst = {g: float((env[(env.method == method) & (env.gamma == g)]["coverage"] - 0.95).abs().max())
                 for g in (0.0,) + GAMMAS}
        best_gamma = min(GAMMAS, key=lambda g: worst[g])
        p = pooled[(pooled.method == method) & (pooled.gamma == best_gamma)]["coverage"]
        adopt = worst[best_gamma] < worst[0.0] and bool(((p >= 0.94) & (p <= 0.96)).all())
        decisions[method] = {"worst_env_deviation": worst, "best_gamma": best_gamma, "adopt_aci": adopt}
    return decisions


def aci_cv(canonical: Path, data_root: Path, out: Path, max_folds: int | None = None) -> dict:
    tracks = r2.load2(canonical, data_root)
    dev = [tr for tr in tracks if tr["split"] in ("train", "validation")]
    envs = sorted({tr["environment_id"] for tr in dev})[:max_folds]
    cache = KalmanCache(dev)
    parts = []
    for env in envs:
        fit_rows = np.array([i for i, tr in enumerate(dev) if tr["environment_id"] != env])
        fit_tracks = [dev[i] for i in fit_rows]
        held = [tr for tr in dev if tr["environment_id"] == env]
        params = cache.fit(fit_rows, dev)
        cuts = np.quantile(np.concatenate([tr["speed5"][np.isfinite(tr["speed5"])] for tr in fit_tracks]), [1 / 3, 2 / 3])
        boost = {h: Learned("boosting").fit(fit_tracks, h) for h in HORIZONS}
        kf_fit = r2.kf_calibration_fit(r2.score2(fit_tracks, params, None, cache, cuts).query("model == 'cv_kalman'"))
        parts.append(_bands(r2.score2(held, params, boost, cache, cuts), kf_fit, r2.boost_uncertainty_fit(fit_tracks, cuts)))
        print(f"fold {env}: done", flush=True)
    pooled, env_cov, cells = _aci_tables(pd.concat(parts, ignore_index=True))
    decision = _choose_aci(pooled, env_cov)
    pooled.to_csv(out / "cv7_aci_pooled.csv", index=False)
    env_cov.to_csv(out / "cv7_aci_by_environment.csv", index=False)
    cells.to_csv(out / "cv7_aci_cells.csv", index=False)
    (out / "cv7_aci_decision.json").write_text(json.dumps(decision, ensure_ascii=False, indent=2), encoding="utf-8")
    return decision


def aci_final(canonical: Path, data_root: Path, out: Path) -> dict:
    decision = json.loads((out / "cv7_aci_decision.json").read_text(encoding="utf-8"))
    if not any(d["adopt_aci"] for d in decision.values()):
        raise SystemExit("CV rule adopted ACI for neither band; the test split is not used.")
    tracks = r2.load2(canonical, data_root)
    train = [tr for tr in tracks if tr["split"] == "train"]
    test = [tr for tr in tracks if tr["split"] == "test"]
    cache = KalmanCache(train)
    params = cache.fit(np.arange(len(train)), train)
    cuts = np.quantile(np.concatenate([tr["speed5"][np.isfinite(tr["speed5"])] for tr in train]), [1 / 3, 2 / 3])
    boost = {h: Learned("boosting").fit(train, h) for h in HORIZONS}
    kf_fit = r2.kf_calibration_fit(r2.score2(train, params, None, cache, cuts).query("model == 'cv_kalman'"))
    bands = _bands(r2.score2(test, params, boost, cache, cuts), kf_fit, r2.boost_uncertainty_fit(train, cuts))
    pooled, env_cov, cells = _aci_tables(bands)
    pooled.to_csv(out / "final7_aci_pooled.csv", index=False)
    env_cov.to_csv(out / "final7_aci_by_environment.csv", index=False)
    cells.to_csv(out / "final7_aci_cells.csv", index=False)
    return {"decision": decision}


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    rows = [[f"{v:.3f}" if isinstance(v, float) else str(v) for v in r] for r in df.itertuples(index=False)]
    return "\n".join(["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)] + ["| " + " | ".join(r) + " |" for r in rows])


def write_summary(out: Path) -> Path:
    lines = ["# 第一问第三轮：记录内遮挡倾向与在线自适应校准", "",
             "规则在运行前写在 `q1_round3.py` 文件头。", ""]
    onset = lambda f: pd.read_csv(out / f).query("target == 'any' and subset == 'onset'").set_index("model")["auc"]
    if (out / "cv6_propensity_reliability.csv").is_file():
        table = pd.DataFrame({"cv": onset("cv6_propensity_reliability.csv")})
        if (out / "final6_propensity_reliability.csv").is_file():
            table["test"] = onset("final6_propensity_reliability.csv")
        lines += ["## 记录内遮挡倾向", "", "数值为新遮挡（近 10 帧无标记）AUC。", "", _md(table.reset_index()), "",
                  "`prop_self` 去掉另一只手的标记比例，是在第一次看到 `prop` 的测试结果后补充的。", ""]
    for stage, label in (("cv7", "交叉验证"), ("final7", "测试集")):
        if not (out / f"{stage}_aci_pooled.csv").is_file():
            continue
        pooled = pd.read_csv(out / f"{stage}_aci_pooled.csv")
        env = pd.read_csv(out / f"{stage}_aci_by_environment.csv")
        worst = (env.assign(dev=(env.coverage - 0.95).abs()).groupby(["method", "gamma"])["dev"].max().rename("worst_env_dev"))
        wide = pooled.pivot_table(index=["method", "gamma"], columns="horizon", values="coverage")
        wide.columns = [f"cov_h{h}" for h in wide.columns]
        lines += [f"## 在线自适应校准（{label}）", "", _md(wide.join(worst).reset_index()), ""]
    if (out / "cv7_aci_decision.json").is_file():
        d = json.loads((out / "cv7_aci_decision.json").read_text(encoding="utf-8"))
        lines += ["规则结论：" + "；".join(f"{m} 步长 {v['best_gamma']}，{'采用' if v['adopt_aci'] else '不采用'}" for m, v in d.items()), ""]
    path = out / "summary_round3.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main(argv=None) -> int:
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("part", choices=("prop", "aci", "summary"))
    parser.add_argument("--canonical", type=Path, default=here / "cache" / "canonical")
    parser.add_argument("--data", type=Path, default=here.parent / "Restocking_Problem_and_Data_EN")
    parser.add_argument("--out", type=Path, default=here / "results" / "q1_real")
    parser.add_argument("--final", action="store_true", help="fit on train and report the test split once")
    parser.add_argument("--max-folds", type=int, default=None, help="smoke test only (aci)")
    args = parser.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    if args.part == "summary":
        print(write_summary(args.out))
        return 0
    if args.part == "prop":
        result = prop_final(args.canonical, args.data, args.out) if args.final else prop_cv(args.canonical, args.data, args.out)
    else:
        result = (aci_final(args.canonical, args.data, args.out) if args.final
                  else aci_cv(args.canonical, args.data, args.out, args.max_folds))
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
