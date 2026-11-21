"""Select a motion retrieval correction on validation; replay it on the old test environments."""
import json
from dataclasses import asdict

import numpy as np
import pandas as pd
import torch

from acceleration_data import AccelerationData, RESULTS, SCALE
from acceleration_evaluation import Policy as PriorPolicy, apply, error_cm, read_models
from acceleration_models import Batches, predict_one
from similar_motion import LAGS_S, MotionLibrary, Policy, consensus, correct


OUT = RESULTS / "相似动作检索"
COUNTS = (5, 11)
FRACTIONS = (.25, .5, 1.)
SHUFFLE_SEEDS = range(4700, 4724)


def prepare_split(data, split, library, batches, nets, prior):
    query = data.eval_ids[split]
    assert library.training_environments.isdisjoint(data.meta.env.to_numpy()[query])
    # Issue every slow-origin forecast before consulting future reference eligibility.
    members = [predict_one(net, batches, query) for net in nets]
    probs, point = (np.mean([pair[j] for pair in members], axis=0) for j in (0, 1))
    current = apply(prior, data.base[query], {prior.model: (probs, point)})
    neighbors = library.nearest(query, max(COUNTS))
    return query, current, neighbors


def verify_current(data, split, query, current):
    mask = data.meta.score.to_numpy()[query]
    ids = query[mask]
    old = pd.read_csv(RESULTS / "origin_predictions.csv.gz",
                      usecols=["split", "sample", "hand", "origin", "validation_selected_cm"])
    expected = old[old.split == split].set_index(["sample", "hand", "origin"]).validation_selected_cm
    keys = pd.MultiIndex.from_frame(data.meta.loc[ids, ["sample", "hand", "origin"]])
    assert expected.index.is_unique and len(expected) == len(keys)
    np.testing.assert_allclose(error_cm(current[mask], data.truth[ids]), expected.loc[keys], atol=1e-4)


def choose_on_validation(data, pack, library):
    query, current, neighbors = pack
    score = data.meta.score.to_numpy()[query]
    labels = data.end_class[query[score]]
    truth = data.truth[query[score]]

    def metrics(points):
        errors = error_cm(points[score], truth)
        return dict(slow_cm=float(errors[labels == 0].mean()), fast_cm=float(errors[labels == 2].mean()),
                    all_slow_origins_cm=float(errors.mean()))

    baseline = metrics(current)
    rows = []
    for k in COUNTS:
        for quiet in (False, True):
            target, strength, _, _ = consensus(neighbors[:, :k], library.outcomes, current,
                                               library.quiet, library.far, quiet)
            for fraction in FRACTIONS:
                policy = Policy(k, quiet, fraction)
                summary = metrics(correct(current, target, strength, fraction))
                passes = (summary["slow_cm"] <= baseline["slow_cm"] and
                          summary["all_slow_origins_cm"] <= baseline["all_slow_origins_cm"])
                rows.append(dict(policy=asdict(policy), **summary, passes_guards=passes))
    eligible = [r for r in rows if r["passes_guards"] and r["fast_cm"] < baseline["fast_cm"]]
    # Keep a validation-selected diagnostic challenger even if the incumbent is retained.
    best = min(eligible or rows, key=lambda r: (r["fast_cm"], r["policy"]["fraction"]))
    return Policy(**best["policy"]), bool(eligible), baseline, rows


def nonoverlap_mask(log):
    keep = np.zeros(len(log), dtype=bool)
    for positions in log.groupby(["sample", "hand"], sort=False).indices.values():
        last = -30
        origins = log.origin.to_numpy()[positions]
        assert (np.diff(origins) > 0).all()
        for pos, origin in zip(positions, origins):
            if origin >= last + 30:
                keep[pos] = True
                last = origin
    return keep


def score_predictions(data, pack, library, policy):
    query, current, neighbors = pack
    neighbors = neighbors[:, :policy.k]
    target, strength, far, quiet = consensus(neighbors, library.outcomes, current,
                                             library.quiet, library.far, policy.quiet_consensus)
    predicted = correct(current, target, strength, policy.fraction)
    # Fixed-policy ablation: mix only distances, preserve the current forecast's
    # direction, and retain zero displacements where that direction is undefined.
    current_radius = np.linalg.norm(current, axis=1)
    neighbor_radius = np.linalg.norm(target, axis=1)
    radial_target = current.copy()
    directed = current_radius > 1e-8
    radial_target[directed] *= (neighbor_radius[directed]/current_radius[directed])[:, None]
    radial = correct(current, radial_target, strength, policy.fraction)
    mask = data.meta.score.to_numpy()[query]
    ids = query[mask]
    log = data.meta.loc[ids, ["sample", "hand", "origin", "env", "split", "end_bin"]].reset_index(drop=True)
    for name, position in (("current", current[mask]), ("retrieval", predicted[mask]),
                           ("distance_only", radial[mask])):
        log[name+"_cm"] = error_cm(position, data.truth[ids])
        log[name+"_length_cm"] = np.linalg.norm(position, axis=1)*SCALE*100
    log["observed_length_cm"] = np.linalg.norm(data.truth[ids], axis=1)*SCALE*100
    log["gain_cm"] = log.current_cm-log.retrieval_cm
    log["far_vote"], log["quiet_vote"] = far[mask], quiet[mask]
    log["weight"] = policy.fraction*strength[mask]
    log["nonoverlap"] = nonoverlap_mask(log)
    donor_rows = library.bank_ids[neighbors[mask]]
    donor_names = library.names[donor_rows]
    donor_origins = data.meta.origin.to_numpy()[donor_rows]
    log["training_neighbors"] = ["|".join(f"{s}@{i}" for s, i in zip(n, o))
                                 for n, o in zip(donor_names, donor_origins)]
    assert np.isfinite(log[["current_cm", "retrieval_cm", "gain_cm"]]).all().all()
    assert (log.loc[log.weight == 0, "gain_cm"] == 0).all()
    return log


def group_results(data, split, log):
    rows = []
    for sampling, sample in (("all", log), ("nonoverlap", log[log.nonoverlap])):
        for env in ["all", *sorted(sample.env.unique())]:
            part = sample if env == "all" else sample[sample.env == env]
            groups = {"all_slow": part, "slow_to_slow": part[part.end_bin == 0],
                      "slow_to_middle": part[part.end_bin == 1], "slow_to_fast": part[part.end_bin == 2],
                      "far_consensus": part[part.far_vote], "quiet_consensus": part[part.quiet_vote]}
            for name, g in groups.items():
                if g.empty:
                    continue
                rows.append(dict(split=split, sampling=sampling, env=env, group=name, n=len(g),
                                 current_cm=g.current_cm.mean(), retrieval_cm=g.retrieval_cm.mean(),
                                 distance_only_cm=g.distance_only_cm.mean(),
                                 gain_cm=g.gain_cm.mean(), current_length_cm=g.current_length_cm.median(),
                                 retrieval_length_cm=g.retrieval_length_cm.median(),
                                 observed_length_cm=g.observed_length_cm.median(), fast_share=(g.end_bin == 2).mean(),
                                 far_votes=g.far_vote.sum(), modified=(g.gain_cm.abs() > 1e-7).sum()))
    other = data.meta.index[(data.meta.split == split) & data.meta.score & ~data.meta.slow].to_numpy()
    keep_error = pd.Series(error_cm(data.base[other], data.truth[other]), index=data.meta.env.to_numpy()[other])
    for env in ["all", *sorted(log.env.unique())]:
        previous = keep_error if env == "all" else keep_error[keep_error.index == env]
        g = log if env == "all" else log[log.env == env]
        n = len(previous)+len(g)
        rows.append(dict(split=split, sampling="all", env=env, group="all_origins", n=n,
                         current_cm=(previous.sum()+g.current_cm.sum())/n,
                         retrieval_cm=(previous.sum()+g.retrieval_cm.sum())/n,
                         distance_only_cm=(previous.sum()+g.distance_only_cm.sum())/n, gain_cm=g.gain_cm.sum()/n))
    return pd.DataFrame(rows)


def shuffled_results(data, split, pack, library, policy):
    query, current, neighbors = pack
    mask = data.meta.score.to_numpy()[query]
    ids = query[mask]
    groups = {"all_slow": np.ones(len(ids), dtype=bool), "slow_to_fast": data.end_class[ids] == 2,
              "slow_to_slow": data.end_class[ids] == 0}
    rows = []
    for seed in SHUFFLE_SEEDS:
        target, strength, _, _ = consensus(neighbors[:, :policy.k], library.shuffled(seed), current,
                                           library.quiet, library.far, policy.quiet_consensus)
        errors = error_cm(correct(current, target, strength, policy.fraction)[mask], data.truth[ids])
        rows.extend(dict(split=split, seed=seed, group=name, shuffled_cm=errors[select].mean())
                    for name, select in groups.items())
    return rows


def write_report(groups, controls, policy, adopted, library):
    mode = "同时允许近距离一致时缩短预测" if policy.quiet_consensus else "仅使用同向远伸的多数票"
    lines = ["相似动作检索与 1 秒腕点预测", "",
             f"检索库有 {len(library.bank_ids):,} 个慢速起点，来自 {len(library.training_names)} 条训练记录、"
             f"{len(library.training_environments)} 个训练环境，包含随后所有速度档。每个训练记录、每只手最多投一票。",
             "输入为起点当时及过去最多 2 秒两手腕点和头部的运动，统一到该起点的头部坐标系。"
             "同侧手匹配。短历史附有实际时长。按同手的每条训练记录取最近的一段，再取距离最近的若干条记录。",
             f"训练集中 1 秒直线位移的下、上三分位为 {library.quiet*SCALE*100:.2f} 和 "
             f"{library.far*SCALE*100:.2f} 厘米。大于等于上界且位于共同方向 60 度内的邻居超过半数时，"
             "用这些邻居的方向和中位距离修正落点。启用近距离修正时，低于等于下界的邻居超过半数可缩短当前距离。",
             f"验证选出的检索候选使用 {policy.k} 条记录、最大修正权重 {policy.fraction:.0%}，{mode}。"
             "实际权重还乘以多数比例超过一半的幅度（2 倍投票比例减 1）。",
             "验证判定条件为慢→快平均误差下降，同时慢→慢及全部慢起点的平均误差均不升。"
             + ("检索候选通过。" if adopted else "所有检索候选均未同时通过，锁定保留现有预测。下列检索列为验证集慢→快误差最低的诊断候选。"),
             "对照预测为原贝叶斯增益预测加上此前验证选定的 25% 轨迹 GRU 修正。"
             "只在慢起点应用本轮候选。另做固定参数对照，按同样权重修正原方向上的预测距离，"
             "保留原预测方向。此项不参与选择。以下数值为三维欧氏平均误差，单位厘米。", ""]
    titles = {"slow_to_fast": "慢→快", "slow_to_slow": "慢→慢", "slow_to_middle": "慢→中",
              "all_slow": "全部慢起点", "all_origins": "全部起点"}
    for split, title in (("validation", "验证"), ("test", "回顾测试")):
        sub = groups[(groups.split == split) & (groups.env == "all")]
        lines.extend([title, "分组        数量     现有预测     检索候选    误差下降     只改距离"])
        for row in sub[(sub.sampling == "all") & sub.group.isin(titles)].itertuples():
            lines.append(f"{titles[row.group]:<8} {row.n:5d} {row.current_cm:11.3f} "
                         f"{row.retrieval_cm:11.3f} {row.gain_cm:+10.3f} {row.distance_only_cm:11.3f}")
        for group in ("slow_to_fast", "slow_to_slow"):
            row = sub[(sub.sampling == "nonoverlap") & (sub.group == group)].iloc[0]
            lines.append(f"按同记录同手至少相隔 30 帧取起点，{titles[group]}有 {row.n} 个，"
                         f"{row.current_cm:.3f} → {row.retrieval_cm:.3f} 厘米。")
        fast = sub[(sub.group == "slow_to_fast") & (sub.sampling == "all")].iloc[0]
        slow = sub[(sub.group == "all_slow") & (sub.sampling == "all")].iloc[0]
        if slow.far_votes:
            lines.append(f"全部慢起点中，{fast.n/slow.n:.1%} 在 1 秒后变快。"
                         f"出现同向远伸多数票时，这个比例为 {fast.far_votes/slow.far_votes:.1%}。")
        lines.append(f"同向远伸多数票覆盖慢起点 {int(slow.far_votes)}/{int(slow.n)}，"
                     f"其中真正慢→快 {int(fast.far_votes)} 个。慢→快组预测位移中位数 "
                     f"{fast.current_length_cm:.2f} → {fast.retrieval_length_cm:.2f} 厘米，"
                     f"参考位移中位数 {fast.observed_length_cm:.2f} 厘米。")
        shuffled = controls[(controls.split == split) & (controls.group == "slow_to_fast")].shuffled_cm
        lines.append(f"固定近邻来源与参数，只在同一训练记录、同一只手中打乱后续位移，"
                     f"{len(shuffled)} 次的慢→快平均误差为 {shuffled.mean():.3f} 厘米，"
                     f"范围 {shuffled.min():.3f}～{shuffled.max():.3f}。")
        lines.append("各环境的慢→快结果")
        by_env = groups[(groups.split == split) & (groups.env != "all") &
                        (groups.group == "slow_to_fast") & (groups.sampling == "all")]
        for row in by_env.itertuples():
            lines.append(f"  {row.env}  {row.n} 个，{row.current_cm:.3f} → {row.retrieval_cm:.3f} 厘米。")
        overall = groups[(groups.split == split) & (groups.env != "all") &
                         (groups.group == "all_origins") & (groups.sampling == "all")]
        lines.append(f"第一问全部起点按环境等权的 1 秒平均误差为 "
                     f"{overall.current_cm.mean():.3f} → {overall.retrieval_cm.mean():.3f} 厘米。")
        lines.append("")
    lines += ["样本单位为预测起点。相邻起点的 1 秒窗口重叠。抽稀先按慢起点取样，再看未来速度。",
              "参数在本轮回顾测试前按 4 个验证环境锁定。4 个测试环境曾在先前实验中使用。",
              "复算时在 c_restocking_toolkit/literature_experiments 下运行 python similar_motion_evaluation.py。"]
    (OUT / "实测结论.txt").write_text("\n".join(lines)+"\n", encoding="utf-8")


def run():
    torch.set_num_threads(3)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = AccelerationData(device)
    old_lock = json.loads((RESULTS / "validation_decision.json").read_text(encoding="utf-8"))
    prior = PriorPolicy(**old_lock["point_policy"])
    assert prior == PriorPolicy("motion_gru", .25, -1.)
    _, models = read_models(data, device)
    batches = Batches(data, device)
    library = MotionLibrary(data, device)
    validation = prepare_split(data, "validation", library, batches, models[prior.model], prior)
    verify_current(data, "validation", *validation[:2])
    policy, adopted, baseline, grid = choose_on_validation(data, validation, library)
    OUT.mkdir(parents=True, exist_ok=True)
    lock = dict(training_records=len(library.training_names), training_environments=len(library.training_environments),
                training_windows=len(library.bank_ids), training_speed_bin_counts=np.bincount(data.end_class[data.train]).tolist(),
                distance_tertiles_cm=[library.quiet*SCALE*100, library.far*SCALE*100], history_lags_s=LAGS_S.tolist(),
                neighbors_per_record=1, same_hand=True, prior=asdict(prior), validation_baseline=baseline,
                challenger=asdict(policy), adopted_on_validation=adopted, validation_candidates=grid,
                rule="Minimize slow-to-fast with slow-to-slow and all-slow mean errors at most the current model. Keep current on ties.",
                diagnostic_if_none="Lowest validation slow-to-fast error; for numerical ties, smaller maximum fraction.",
                shuffled_controls=len(SHUFFLE_SEEDS))
    (OUT / "validation_lock.json").write_text(json.dumps(lock, indent=2, allow_nan=False), encoding="utf-8")
    print("Validation locked before test:", policy, "adopted:", adopted, flush=True)
    test = prepare_split(data, "test", library, batches, models[prior.model], prior)
    verify_current(data, "test", *test[:2])
    logs, metrics, shuffled = [], [], []
    for split, pack in (("validation", validation), ("test", test)):
        detail = score_predictions(data, pack, library, policy)
        logs.append(detail)
        metrics.append(group_results(data, split, detail))
        shuffled.extend(shuffled_results(data, split, pack, library, policy))
    metrics, controls = pd.concat(metrics, ignore_index=True), pd.DataFrame(shuffled)
    metrics.to_csv(OUT / "endpoint_errors.csv", index=False)
    controls.to_csv(OUT / "shuffled_controls.csv", index=False)
    pd.concat(logs, ignore_index=True).to_csv(OUT / "origin_predictions.csv.gz", index=False, compression="gzip")
    write_report(metrics, controls, policy, adopted, library)
    print((OUT / "实测结论.txt").read_text(encoding="utf-8"), flush=True)


if __name__ == "__main__":
    run()
