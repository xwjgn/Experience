"""Paired wrist/palm representative-point and palm-input comparisons on the frozen Q1 split."""
import gc
import json
from dataclasses import dataclass

import joblib
import numpy as np
import pandas as pd
import torch
from threadpoolctl import threadpool_limits

from acceleration_data import AccelerationData, RESULTS
from acceleration_evaluation import Policy as PriorPolicy, error_cm
from acceleration_models import AnticipationGRU, Batches, SEEDS, train_one
from bayesian_gain import SEED as GAIN_SEED, fit_filter
from endpoint_decision import CM, OUT as PRIOR, Rule, decisions, fit_outcomes, forecast_alternatives
from motion_data import KIT, MotionData
from similar_motion import MotionLibrary
from similar_motion_evaluation import nonoverlap_mask, prepare_split


OUT = RESULTS / "腕点与掌心对照"
MODEL_DIR = OUT / "模型"
VARIANTS = {"wrist": ("wrist", None), "palm": ("palm_center", None),
            "wrist_palm_input": ("wrist", "measured"), "wrist_zero_input": ("wrist", "zero")}
TITLES = {"wrist": "腕点→腕点", "palm": "掌心→掌心", "wrist_palm_input": "腕点加掌心历史→腕点",
          "wrist_zero_input": "同维零通道→腕点", "palm_to_wrist": "掌心位移换算腕点",
          "hold_wrist": "保持腕点", "hold_palm": "保持掌心"}


@dataclass
class Arm:
    data: AccelerationData
    batches: Batches
    library: MotionLibrary
    nets: list
    fitted: dict


def alternate_speed(motion, point="palm_center"):
    """Second, descriptive grouping only. These labels do not tune or drive forecasts."""
    speeds, terminal, training = [], [], []
    for tr, _, _ in motion.parts:
        p = tr["points"][point]
        speed = np.r_[np.nan, np.linalg.norm(np.diff(p,axis=0),axis=1)/np.diff(tr["t"])]
        avg = pd.Series(speed).rolling(5,min_periods=5).mean().to_numpy()
        speeds.append(avg)
        terminal.append(np.r_[avg[30:],np.full(30,np.nan)])
        if tr["split"] == "train":
            training.append(avg[np.isfinite(avg)])
    cuts = np.quantile(np.concatenate(training),[1/3,2/3])
    return np.concatenate(speeds),np.concatenate(terminal),cuts


def make_arm(name, device, *, variant=None, model_dir=MODEL_DIR, speed_point=None):
    point, context = VARIANTS[name] if variant is None else variant
    motion = MotionData(visual=False, point=point, palm_context=context)
    secondary = alternate_speed(motion, speed_point or "palm_center") if name == "wrist" or speed_point else None
    gain = None
    if name != "wrist":
        folder = model_dir/name
        folder.mkdir(parents=True,exist_ok=True)
        tensors = {k:torch.as_tensor(motion.a[k],device=device) for k in ("x","hist","dt","hist_flags","y")}
        gain = fit_filter(motion,tensors,device,GAIN_SEED)
        torch.save({"state":gain.state_dict(),"seed":GAIN_SEED,"point":point,"palm_context":context},folder/"bayesian_gain.pt")
        del tensors
    d = AccelerationData(device,motion=motion,gain_model=gain)
    del gain, motion
    batches,library = Batches(d,device),MotionLibrary(d,device)
    if name == "wrist":
        nets = []
        for seed in SEEDS:
            weights = torch.load(RESULTS/f"motion_gru_{seed}.pt",map_location=device,weights_only=True)
            assert weights["n_features"] == d.sequence.shape[1] and not weights["visual"]
            model = AnticipationGRU(d.sequence.shape[1],False).to(device)
            model.load_state_dict(weights["state_dict"])
            nets.append(model.eval())
        fitted = joblib.load(PRIOR/"displacement_distribution.joblib")
    else:
        nets = [train_one(d,batches,"motion_gru",False,seed,output_dir=folder) for seed in SEEDS]
        with threadpool_limits(limits=3):
            fitted = fit_outcomes(d,library)
        joblib.dump(fitted,folder/"displacement_distribution.joblib",compress=3)
    print(f"{name}: inputs={d.sequence.shape[1]}, train slow origins={len(d.train)}",flush=True)
    return Arm(d,batches,library,nets,fitted),secondary


def issue(arm,split,prior,rule,retrieval,*,scored_only=True):
    d = arm.data
    pack = prepare_split(d,split,arm.library,arm.batches,arm.nets,prior)
    choices = forecast_alternatives(d,pack,arm.library,arm.fitted,retrieval)
    action = decisions(choices,rule)
    slow_ids = choices["ids"]
    slow_prediction = choices["points"][np.arange(len(action)),action]
    # An unscorable origin receives the same forward computation as a scored one.
    ids = d.meta.index[d.meta.split.eq(split)].to_numpy()
    points = d.base[ids].astype(float)
    slots = np.searchsorted(ids,slow_ids)
    np.testing.assert_array_equal(ids[slots],slow_ids)
    points[slots] = slow_prediction
    if not scored_only:
        return ids,points
    eligible = d.meta.score.to_numpy()[ids]
    return ids[eligible],points[eligible]


def mark_spaced(g, slow):
    mask = np.zeros(len(g),bool)
    positions = np.flatnonzero(slow)
    mask[positions] = nonoverlap_mask(g.iloc[positions].reset_index(drop=True))
    return mask


def evaluate(arms,split,secondary,prior,rule,retrieval):
    ids,points = {},{}
    for name,arm in arms.items():
        ids[name], points[name] = issue(arm,split,prior,rule,retrieval)
        np.testing.assert_array_equal(ids[name],ids["wrist"])
    idx=ids["wrist"]
    d=arms["wrist"].data
    g=d.meta.loc[idx,["sample","hand","origin","env","split","slow","end_bin"]].reset_index(drop=True)
    p0,p1,cuts=secondary
    g["palm_slow"] = (p0[idx] < cuts[0]) & (g.origin.to_numpy() >= 10)
    g["palm_end_bin"] = np.digitize(p1[idx],cuts)
    assert np.isfinite(p1[idx]).all()
    g["nonoverlap_wrist"] = mark_spaced(g,g.slow.to_numpy())
    g["nonoverlap_palm"] = mark_spaced(g,g.palm_slow.to_numpy())
    wrist_y, palm_y = d.truth[idx], arms["palm"].data.truth[idx]
    for name,predicted in points.items():
        reference = palm_y if name == "palm" else wrist_y
        g[name] = error_cm(predicted,reference)
    # Apply the predicted displacement of the palm to the current wrist. This
    # transfer uses only the current palm-to-wrist offset, never its future value.
    g["palm_to_wrist"] = error_cm(points["palm"],wrist_y)
    g["hold_wrist"] = np.linalg.norm(wrist_y,axis=1)*CM
    g["hold_palm"] = np.linalg.norm(palm_y,axis=1)*CM
    g["relative_offset_change_cm"] = error_cm(palm_y,wrist_y)
    old = pd.read_csv(PRIOR/"origin_predictions.csv.gz")
    old = old[old.split == split]
    matching = g[g.slow].merge(old,on=["sample","hand","origin"],validate="one_to_one")
    assert len(matching) == len(old) == int(g.slow.sum())
    np.testing.assert_allclose(matching.wrist,matching.model_cm,atol=1e-4)
    # The triangle inequality bounds the cost of changing reference for the same forecast.
    assert (np.abs(g.palm_to_wrist-g.palm) <= g.relative_offset_change_cm+1e-5).all()
    return g


def tables(g):
    rows, kin = [], []
    split=g.split.iat[0]
    for grouping, slow, end, spaced in (("wrist_speed",g.slow,g.end_bin,g.nonoverlap_wrist),
                                        ("palm_speed",g.palm_slow,g.palm_end_bin,g.nonoverlap_palm)):
        for sampling, eligible in (("all",np.ones(len(g),bool)),("nonoverlap",spaced.to_numpy())):
            groups = {"all_slow":slow, "slow_to_slow":slow & (end==0), "slow_to_fast":slow & (end==2)}
            if sampling == "all" and grouping == "wrist_speed":
                groups["all_origins"] = np.ones(len(g),bool)
            for scope, mask in groups.items():
                selected = g[eligible & mask]
                kin.append(dict(split=split,grouping=grouping,sampling=sampling,scope=scope,n=len(selected),
                                wrist_move_median_cm=selected.hold_wrist.median(),
                                palm_move_median_cm=selected.hold_palm.median(),
                                offset_change_mean_cm=selected.relative_offset_change_cm.mean(),
                                offset_change_median_cm=selected.relative_offset_change_cm.median(),
                                offset_change_p90_cm=selected.relative_offset_change_cm.quantile(.9)))
                for env, sub in [("all",selected),*selected.groupby("env",sort=True)]:
                    macro=sub.groupby("env")[list(TITLES)].mean().mean()
                    for name in TITLES:
                        rows.append(dict(split=split,grouping=grouping,sampling=sampling,scope=scope,env=env,
                                         method=name,n=len(sub),pooled_cm=sub[name].mean(),macro_cm=macro[name]))
    return pd.DataFrame(rows),pd.DataFrame(kin)


def metrics(table, split, name):
    cells=table.query("split==@split and grouping=='wrist_speed' and sampling=='all' and env=='all' and method==@name").set_index("scope")
    return dict(slow_slow=float(cells.loc["slow_to_slow","pooled_cm"]),fast=float(cells.loc["slow_to_fast","pooled_cm"]),
                slow=float(cells.loc["all_slow","pooled_cm"]),overall=float(cells.loc["all_origins","macro_cm"]))


def report(grouped,kin,cutoffs,lock):
    old = pd.read_csv(KIT/"results/q1_real/prediction_metrics.csv")
    old = old.query("split=='test' and dataset=='all' and horizon==30 and model=='damped_kalman'").set_index("point")
    lines=["腕点与掌心的 1 秒预测对照", "",
           "腕点采用 j00。掌心沿用本项目 q1_state.py 中的固定定义，将 j00 及食指、中指、无名指、小指的指根 "
           "j05、j09、j13、j17 三维坐标等权平均。无论可见性如何均使用同一组点和权重。",
           "四候选无图像选择器使用既有规则。按原环境划分，55 条训练记录来自 16 个环境，验证和回顾测试分别为 7 条记录、4 个环境。"
           "所有对照逐起点配对，参考帧质量、预测时距、训练样本、超参数、数据可见截止时刻与随机种子相同。"
           "掌心目标的两手运动历史、滤波和 GRU 输出、检索历史与训练落点分布均按该目标重建或重新拟合，"
           "保留相同的解剖手指特征。常规无图像腕点结果已逐起点与之前存档交叉核对。",
           "主分析沿用训练腕速三分位定义慢、快，使两种代表点的比较聚焦同一批操作。"
           f"另按训练集掌速重新划组的三分位为 {cutoffs[0]:.5f} 和 {cutoffs[1]:.5f} 米/秒；这是给同一组预测重新分层的描述性核对。",
           "额外的腕点输入含当前腕到掌心的向量及相对过去 3、9、24 帧的变化，以当前头部坐标表达。"
           "同维零输入对照使用相同网络大小和种子。掌心位移换算腕点假设未来 1 秒腕到掌心的向量维持当前值。",
           f"第一版中已有代表点敏感性实验。按第一版的起点汇总规则，30 帧速度衰减卡尔曼的回顾测试误差为"
           f"腕点 {old.loc['wrist','E_mean_mm']/10:.2f} 厘米，掌心 {old.loc['palm_center','E_mean_mm']/10:.2f} 厘米。",
           "误差单位为厘米。总体为全体合格起点按环境等权的 1 秒三维平均误差；慢→快与慢→慢按相同起点等权。", ""]
    for split,title in (("validation","验证"),("test","回顾测试")):
        lines.extend([title,"方法                         总体          慢→快        慢→慢         全部慢起点"])
        for name in TITLES:
            r=metrics(grouped,split,name)
            lines.append(f"{TITLES[name]:20s} {r['overall']:8.3f} {r['fast']:12.3f} {r['slow_slow']:12.3f} {r['slow']:13.3f}")
        for tag,scheme in (("wrist_speed","按腕速"),("palm_speed","按掌速")):
            fast=grouped.query("split==@split and grouping==@tag and env=='all' and scope=='slow_to_fast'")
            for sampling in ("all","nonoverlap"):
                t=fast[fast.sampling==sampling].set_index("method")
                sample_name = "全部起点" if sampling == "all" else "同记录同手相隔至少 30 帧"
                lines.append(f"{scheme}的慢→快，{sample_name}共 {int(t.loc['wrist','n'])} 个。腕点 {t.loc['wrist','pooled_cm']:.3f}，"
                             f"掌心 {t.loc['palm','pooled_cm']:.3f}，腕点加掌心输入 {t.loc['wrist_palm_input','pooled_cm']:.3f}。")
        h=kin.query("split==@split and grouping=='wrist_speed' and sampling=='all' and scope=='slow_to_fast'").iloc[0]
        lines.append(f"慢→快时，腕点与掌心 1 秒净位移的中位数分别为 {h.wrist_move_median_cm:.2f} 与 {h.palm_move_median_cm:.2f} 厘米。"
                     f"二者位移向量之差的中位数为 {h.offset_change_median_cm:.2f} 厘米，90% 分位为 {h.offset_change_p90_cm:.2f} 厘米。")
        envs=grouped.query("split==@split and grouping=='wrist_speed' and sampling=='all' and scope=='slow_to_fast' and env!='all'")
        for env,t in envs.groupby("env"):
            t=t.set_index("method")
            lines.append(f"  {env}  共 {int(t.loc['wrist','n'])} 个，腕点 {t.loc['wrist','pooled_cm']:.3f}，"
                         f"掌心 {t.loc['palm','pooled_cm']:.3f}，腕点加掌心输入 {t.loc['wrist_palm_input','pooled_cm']:.3f}。")
        lines.append("")
    lines+=["将掌心直接作为目标时，误差以它自己的记录坐标为参考。辅助输入和掌心位移换算则仍按原腕点参考评分。"
            "独立物理真值未提供。现有四个测试环境已用于过去的回顾比较。",
            ("新增掌心输入通过了预设的四项验证条件。" if lock["validation_palm_input_passes"] else
             "新增掌心输入未通过预设的四项验证条件。当前保留无图像腕点方案，掌心作为代表点敏感性对照。"),
            "复算时在 c_restocking_toolkit/literature_experiments 中运行 python palm_point_evaluation.py。"]
    (OUT/"实测结论.txt").write_text("\n".join(lines)+"\n",encoding="utf8")


def run():
    torch.set_num_threads(3)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    old=json.loads((RESULTS/"相似动作检索"/"validation_lock.json").read_text(encoding="utf8"))
    prior,retrieval=PriorPolicy(**old["prior"]),old["challenger"]
    rule=Rule(**json.loads((PRIOR/"validation_lock.json").read_text(encoding="utf8"))["rule"])
    OUT.mkdir(parents=True,exist_ok=True)
    protocol=dict(representative_points={"wrist":[0],"palm":[0,5,9,13,17]}, palm_weights=[.2]*5,
                  variants=VARIANTS, original_slow_groups="wrist 5-frame training tertiles", seeds=list(SEEDS),
                  gain_seed=GAIN_SEED, epochs=24, retained_target="same quality mask, 30 future frames",
                  selector=vars(rule),prior=vars(prior),retrieval=retrieval,
                  policy="Descriptive target comparison; wrist-input challenger must improve slow-to-fast without worsening slow-to-slow, all slow origins or overall macro mean.")
    (OUT/"protocol.json").write_text(json.dumps(protocol,indent=2),encoding="utf8")
    arms={}
    for name in VARIANTS:
        arms[name],alt=make_arm(name,device)
        if name=="wrist":
            secondary=alt
        else:
            pd.testing.assert_frame_equal(arms[name].data.meta,arms["wrist"].data.meta)
            np.testing.assert_array_equal(arms[name].data.train,arms["wrist"].data.train)
        gc.collect()
    valid=evaluate(arms,"validation",secondary,prior,rule,retrieval)
    vtab,vkin=tables(valid)
    scores={name:metrics(vtab,"validation",name) for name in VARIANTS}
    baseline,aux=scores["wrist"],scores["wrist_palm_input"]
    passed=aux["fast"] < baseline["fast"] and all(aux[k]<=baseline[k] for k in ("slow_slow","slow","overall"))
    lock=dict(validation_palm_input_passes=bool(passed),scores=scores,palm_speed_tertiles=secondary[2].tolist(),
              test_use="Previously examined environments. No test tuning.")
    (OUT/"validation_decision.json").write_text(json.dumps(lock,indent=2,allow_nan=False),encoding="utf8")
    print("Validation locked before retrospective test:",scores,flush=True)
    test=evaluate(arms,"test",secondary,prior,rule,retrieval)
    ttab,tkin=tables(test)
    results,kin=pd.concat([vtab,ttab],ignore_index=True),pd.concat([vkin,tkin],ignore_index=True)
    results.to_csv(OUT/"endpoint_errors.csv",index=False)
    kin.to_csv(OUT/"paired_displacements.csv",index=False)
    pd.concat([valid,test],ignore_index=True).to_csv(OUT/"per_origin.csv.gz",index=False,compression="gzip")
    report(results,kin,secondary[2],lock)
    print((OUT/"实测结论.txt").read_text(encoding="utf8"))


if __name__=="__main__":
    run()
