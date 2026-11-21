"""Choose a wrist endpoint using the distribution of real displacements in the training data."""
import json
from dataclasses import asdict, dataclass

import joblib
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

from acceleration_data import AccelerationData, RESULTS, SCALE
from acceleration_evaluation import Policy as PriorPolicy, error_cm, read_models
from acceleration_models import Batches, linear_inputs
from similar_motion import MotionLibrary, consensus, correct
from similar_motion_evaluation import nonoverlap_mask, prepare_split, verify_current


OUT = RESULTS / "检索修正选择"
ACTION_NAMES = ("keep", "turn", "reach", "shorten")
CM = SCALE * 100


def motion_features(data, ids):
    chunks = []
    for batch in np.array_split(ids, max(1, int(np.ceil(len(ids)/512)))):
        # The old linear baseline concatenates a learned forecast to history.
        # Train the outcome model on sensed history alone, independent of the forecasts it will compare.
        history = linear_inputs(data, batch)[:, :-data.base.shape[1]]
        now = data.sequence[batch]*data.feature_sd + data.feature_mean
        # After the two wrist positions, hand normal, finger offsets and openings
        # come 17 channels per lag: velocities of both wrists, head and head direction first.
        magnitudes = []
        for offset in (26, 43, 60):
            velocity = now[:, offset:offset+12].reshape(-1, 4, 3)
            magnitudes.append(np.log1p(10*np.linalg.norm(velocity, axis=2)))
        chunks.append(np.column_stack([history, *magnitudes]))
    return np.vstack(chunks)


def fit_outcomes(data, library):
    train = data.meta.iloc[data.train].copy()
    train["block"] = train.origin // 6
    thin = train.drop_duplicates(["sample", "hand", "block"])
    ids = thin.index.to_numpy()
    assert thin.split.eq("train").all()
    # Each environment has equal total weight, as do records within an environment
    # and hands within a record. A sustained slow motion cannot dominate simply by lasting longer.
    hand_count = thin.groupby("sample").hand.transform("nunique")
    record_count = thin.groupby("env")["sample"].transform("nunique")
    per_hand = thin.groupby(["sample", "hand"]).origin.transform("size")
    weights = 1. / (record_count*hand_count*per_hand).to_numpy()
    weights /= weights.mean()
    y = data.truth[ids].astype(float)
    radius = np.linalg.norm(y, axis=1)
    size = np.digitize(radius, [library.quiet, library.far])
    axis = np.abs(y).argmax(axis=1)
    side = (y[np.arange(len(y)), axis] >= 0).astype(int)
    bins = np.where(size == 0, 0, 1 + 6*(size-1) + 2*axis + side)
    model = LogisticRegression(C=.012, max_iter=800, random_state=1837)
    model.fit(motion_features(data, ids), bins, sample_weight=weights)
    assert model.n_iter_.max() < model.max_iter and np.array_equal(model.classes_, np.arange(13))
    order = np.argsort(bins, kind="stable")
    counts = np.bincount(bins)
    # Within each class preserve actual 3D reference displacements, including their lengths.
    conditional = weights / np.bincount(bins, weights=weights)[bins]
    return dict(model=model, outcomes=y[order], weights=conditional[order],
                offsets=np.r_[0, np.cumsum(counts)[:-1]], counts=counts, records=thin["sample"].nunique(),
                environments=thin.env.nunique(), n=len(ids))


def expected_error(point, probabilities, fitted):
    risk = np.empty(len(point))
    for start in range(0, len(point), 160):
        batch = slice(start, start+160)
        distances = cdist(point[batch], fitted["outcomes"])*fitted["weights"]
        conditional = np.add.reduceat(distances, fitted["offsets"], axis=1)
        risk[batch] = np.einsum("ij,ij->i", probabilities[batch], conditional)*CM
    return risk


def forecast_alternatives(data, pack, library, fitted, retrieval):
    ids, current, neighbors = pack
    target, strength, far, _ = consensus(neighbors[:, :retrieval["k"]], library.outcomes, current,
                                         library.quiet, library.far, retrieval["quiet_consensus"])
    reach = correct(current, target, strength, retrieval["fraction"])
    turn = current.copy()
    length = np.linalg.norm(reach, axis=1)
    directed = far & (length > 1e-8)
    turn[directed] = reach[directed] * (np.linalg.norm(current[directed], axis=1)/length[directed])[:, None]
    np.testing.assert_allclose(np.linalg.norm(turn, axis=1), np.linalg.norm(current, axis=1), atol=1e-6)
    points = np.stack([current, turn, reach, .5*current], axis=1)
    probabilities = fitted["model"].predict_proba(motion_features(data, ids))
    risk = np.column_stack([expected_error(points[:, j], probabilities, fitted) for j in range(len(ACTION_NAMES))])
    assert np.isfinite(risk).all()
    return dict(ids=ids, points=points, risk=risk, near=probabilities[:, 0],
                probabilities=probabilities, far_vote=far)


@dataclass(frozen=True)
class Rule:
    name: str
    margin_cm: float = 0.
    near_cut: float = 1.


def decisions(pred, rule):
    n = len(pred["ids"])
    if rule.name == "keep":
        return np.zeros(n, dtype=int)
    if rule.name in ("turn", "reach"):
        return np.where(pred["far_vote"], ACTION_NAMES.index(rule.name), 0)
    risk = pred["risk"].copy()
    allow_neighbors = pred["far_vote"] & (rule.name != "shorten_only")
    risk[~allow_neighbors, 1:3] = np.inf
    allow_shorten = (rule.name in ("both", "shorten_only")) & (pred["near"] >= rule.near_cut)
    risk[~allow_shorten, 3] = np.inf
    action = risk.argmin(axis=1)
    saving = risk[:, 0] - risk[np.arange(n), action]
    return np.where(saving > rule.margin_cm, action, 0)


def scored(data, pred, rule):
    ids = pred["ids"]
    action = decisions(pred, rule)
    issued = pred["points"][np.arange(len(ids)), action]
    mask = data.meta.score.to_numpy()[ids]
    scored_ids = ids[mask]
    g = data.meta.loc[scored_ids, ["sample", "hand", "origin", "env", "split", "end_bin"]].reset_index(drop=True)
    g["current_cm"] = error_cm(pred["points"][mask, 0], data.truth[scored_ids])
    g["model_cm"] = error_cm(issued[mask], data.truth[scored_ids])
    g["gain_cm"] = g.current_cm-g.model_cm
    g["action"] = np.asarray(ACTION_NAMES)[action[mask]]
    g["near_probability"] = pred["near"][mask]
    g["far_vote"] = pred["far_vote"][mask]
    g["nonoverlap"] = nonoverlap_mask(g)
    return g


def hindsight_diagnostics(data, pred, log):
    # Called only after the rule is locked. Give the hindsight selector all four
    # geometric candidates, including shortening even where the live rule disallows it.
    query = pred["ids"]
    mask = data.meta.score.to_numpy()[query]
    truth = data.truth[query[mask]]
    options = pred["points"][mask]
    errors = np.linalg.norm(options-truth[:, None, :], axis=2)*CM
    log["best_available_cm"] = errors.min(axis=1)
    log["all_candidates_too_short"] = np.linalg.norm(options, axis=2).max(axis=1) < np.linalg.norm(truth, axis=1)
    np.testing.assert_array_less(log.best_available_cm.to_numpy(), log.model_cm.to_numpy()+1e-6)


def metrics(log, total_by_env):
    env_gain = log.groupby("env").gain_cm.sum() / total_by_env
    assert np.isfinite(env_gain).all()
    return dict(slow_cm=log.loc[log.end_bin == 0, "model_cm"].mean(),
                fast_cm=log.loc[log.end_bin == 2, "model_cm"].mean(),
                all_slow_cm=log.model_cm.mean(), overall_macro_gain_cm=env_gain.mean(),
                changed=int(log.action.ne("keep").sum()))


def choose(data, pred, total_by_env):
    old = metrics(scored(data, pred, Rule("keep")), total_by_env)
    choices = [Rule("turn")]
    for margin in (0., .5, 1.):
        choices.append(Rule("neighbors", margin))
        choices.extend(Rule("both", margin, cutoff) for cutoff in (.5, .7))
    rows = []
    for rule in choices:
        values = metrics(scored(data, pred, rule), total_by_env)
        passes = (values["fast_cm"] < old["fast_cm"] and values["slow_cm"] <= old["slow_cm"] and
                  values["all_slow_cm"] <= old["all_slow_cm"] and values["overall_macro_gain_cm"] >= 0)
        rows.append({"rule": asdict(rule), **values, "passes": bool(passes)})
    feasible = [r for r in rows if r["passes"]]
    best = min(feasible or rows, key=lambda r: (r["fast_cm"], r["changed"]))
    return Rule(**best["rule"]), bool(feasible), old, rows


def summary(data, log, method):
    split = log.split.iat[0]
    rows = []
    for sampling, filtered in (("all", log), ("nonoverlap", log[log.nonoverlap])):
        for env in ["all", *sorted(filtered.env.unique())]:
            part = filtered if env == "all" else filtered[filtered.env == env]
            scopes = {"all_slow": part, "slow_to_slow": part[part.end_bin == 0],
                      "slow_to_middle": part[part.end_bin == 1], "slow_to_fast": part[part.end_bin == 2]}
            for name, group in scopes.items():
                if len(group):
                    rows.append(dict(method=method, split=split, sampling=sampling, env=env, group=name,
                                     n=len(group), current_cm=group.current_cm.mean(), model_cm=group.model_cm.mean(),
                                     gain_cm=group.gain_cm.mean(), changed=int(group.action.ne("keep").sum())))
    other = data.meta.index[(data.meta.split == split) & data.meta.score & ~data.meta.slow].to_numpy()
    prior = pd.Series(error_cm(data.base[other], data.truth[other]), index=data.meta.env.to_numpy()[other])
    for env in sorted(log.env.unique()):
        old = prior[prior.index == env]
        part = log[log.env == env]
        n = len(old)+len(part)
        rows.append(dict(method=method, split=split, sampling="all", env=env, group="all_origins",
                         n=n, current_cm=(old.sum()+part.current_cm.sum())/n,
                         model_cm=(old.sum()+part.model_cm.sum())/n, gain_cm=part.gain_cm.sum()/n,
                         changed=int(part.action.ne("keep").sum())))
    return pd.DataFrame(rows)


def report(results, detail, rule, feasible, valid_n):
    cases = {"slow_to_fast": "慢→快", "slow_to_slow": "慢→慢", "all_slow": "全部慢起点"}
    lines = ["根据历史运动选择 1 秒腕点修正", "",
             f"训练数据来自题目 16 个环境、55 条记录。每条记录每只手每 6 帧分段取首个合格慢起点，得到 {valid_n} 个训练窗口。",
             "用过去最多 2 秒的手腕、头、另一只手、手指和已知质量等信息，训练正则化逻辑回归，"
             "预测腕点位移大小和主要方向的 13 类概率。短位移合为一类，中、长位移各按头部坐标系三根轴的正负方向分为六类。"
             "长短边界沿用训练集位移三分位。每类保留真实训练位移，依预测概率计算每个候选落点的预期三维欧氏误差。",
             "可选落点包括原预测、保留原距离只转方向、沿用上一轮的远伸修正，以及将原位移减半。"
             "只转向与远伸修正均须满足近邻多数同向远伸。减半须达到近距离概率门槛。"
             "除独立的只转向对照外，仅在预期误差下降超过设定值时采用修正。",
             f"验证候选 {rule.name}，最小预期收益 {rule.margin_cm:.1f} 厘米，近距离概率阈值 {rule.near_cut:.2f}（仅含缩短的方案使用）。",
             ("候选在验证集满足预设条件。" if feasible else "全部候选均未同时满足验证条件，保留现有主模型。表内新结果为验证集慢→快误差最低的诊断候选。"),
             "条件要求慢→快改善、慢→慢和全部慢起点均不变差，且第一问全部起点按环境等权的平均误差不升。", ""]
    for split, title in (("validation", "验证"), ("test", "回顾测试")):
        part = results[(results.split == split) & (results.method == "challenger")]
        lines.extend([title, "分组与方法         起点数    现有误差    候选误差    下降量（厘米）"])
        for kind, label in cases.items():
            row = part.query("sampling=='all' and env=='all' and group==@kind").iloc[0]
            lines.append(f"{label:8s} {row.n:6d} {row.current_cm:11.3f} {row.model_cm:11.3f} {row.gain_cm:+9.3f}")
        macro = part[part.group == "all_origins"][["current_cm", "model_cm"]].mean()
        lines.append(f"第一问各环境等权的全部起点误差为 {macro.current_cm:.3f} → {macro.model_cm:.3f} 厘米。")
        for kind in ("slow_to_fast", "slow_to_slow"):
            row = part.query("sampling=='nonoverlap' and env=='all' and group==@kind").iloc[0]
            lines.append(f"每记录每手至少相隔 30 帧取样，{cases[kind]}有 {row.n} 个，"
                         f"{row.current_cm:.3f} → {row.model_cm:.3f} 厘米。")
        for row in part.query("sampling=='all' and env!='all' and group=='slow_to_fast'").itertuples():
            lines.append(f"  {row.env} 的慢→快样本有 {row.n} 个，{row.current_cm:.3f} → {row.model_cm:.3f} 厘米。")
        counts = detail[detail.split == split].action.value_counts()
        lines.append("预测选择次数    " + "    ".join(f"{a}  {counts.get(a, 0)}" for a in ACTION_NAMES))
        fast = detail[(detail.split == split) & detail.end_bin.eq(2)]
        lines.append(f"事后从四个候选中逐起点选最接近真实腕点的一个，慢→快误差为 "
                     f"{fast.best_available_cm.mean():.3f} 厘米。这是现有候选集合的误差下界，使用了未来坐标。"
                     f"{int(fast.all_candidates_too_short.sum())}/{len(fast)} 个慢→快起点（"
                     f"{fast.all_candidates_too_short.mean():.1%}）的四个候选位移均短于参考位移。")
        lines.append("")
    lines.extend(["验证选择先于回顾测试。本题 4 个验证与 4 个测试环境曾在前序实验中使用。",
                  "endpoint_errors.csv 还包含只转向、原检索修正、仅判断邻居、不使用邻居仅缩短等对照。",
                  "复算时在 c_restocking_toolkit/literature_experiments 目录运行 python endpoint_decision.py。"])
    (OUT/"实测结论.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run():
    torch.set_num_threads(3)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = AccelerationData(device)
    previous = json.loads((RESULTS/"相似动作检索"/"validation_lock.json").read_text(encoding="utf-8"))
    prior, retrieval = PriorPolicy(**previous["prior"]), previous["challenger"]
    assert retrieval == {"k": 5, "quiet_consensus": False, "fraction": .5}
    _, models = read_models(data, device)
    batches, library = Batches(data, device), MotionLibrary(data, device)
    OUT.mkdir(parents=True, exist_ok=True)
    with threadpool_limits(limits=3):
        fitted = fit_outcomes(data, library)
    joblib.dump(fitted, OUT/"displacement_distribution.joblib", compress=3)

    def prepare(split):
        pack = prepare_split(data, split, library, batches, models[prior.model], prior)
        verify_current(data, split, *pack[:2])
        return forecast_alternatives(data, pack, library, fitted, retrieval)

    validation = prepare("validation")
    total_by_env = data.meta[(data.meta.split == "validation") & data.meta.score].env.value_counts()
    best, feasible, ref, grid = choose(data, validation, total_by_env)
    lock = dict(rule=asdict(best), passes=feasible, current_validation=ref, candidates=grid,
                frozen_retrieval=retrieval, n_train=fitted["n"], class_counts=fitted["counts"].tolist(),
                selection="Improve slow-to-fast; protect slow-to-slow, pooled slow origins and environment-equal overall Q1.",
                test_use="Retrospective only; no test tuning")
    (OUT/"validation_lock.json").write_text(json.dumps(lock, indent=2, allow_nan=False), encoding="utf-8")
    print("Locked before retrospective test", best, feasible, flush=True)
    test = prepare("test")
    logs, results = [], []
    for pred in (validation, test):
        log = scored(data, pred, best)
        hindsight_diagnostics(data, pred, log)
        logs.append(log)
        results.append(summary(data, log, "challenger"))
        controls = {"old_retrieval": Rule("reach"), "turn_only": Rule("turn"),
                    "gated_neighbors": Rule("neighbors", best.margin_cm),
                    "shorten_only": Rule("shorten_only", best.margin_cm, best.near_cut if best.name == "both" else .5)}
        results.extend(summary(data, scored(data, pred, rule), key) for key, rule in controls.items())
    detail, results = pd.concat(logs, ignore_index=True), pd.concat(results, ignore_index=True)
    results.to_csv(OUT/"endpoint_errors.csv", index=False)
    detail.to_csv(OUT/"origin_predictions.csv.gz", compression="gzip", index=False)
    report(results, detail, best, feasible, fitted["n"])
    print((OUT/"实测结论.txt").read_text(encoding="utf-8"))


if __name__ == "__main__":
    run()
