"""GRU trajectories, returned-feedback adaptive bands and budgeted observation on the Q2 replay."""
import argparse
import json
import sys
import time
from zipfile import ZipFile

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from video_inputs import KIT, ROOT

H, L, DELAY, BUDGET = 12, 30, 3, 2
ALPHA, GAMMA = .05, .04
SEEDS = (8501, 8502, 8503)
KNOTS = (0., .10, .30, .60)
OUT = ROOT / "分析报告" / "第二问GRU轨迹与整段校准"
ARCHIVE = ROOT / "分析报告" / "C题分析复现材料.zip"
sys.path.insert(0, str(ARCHIVE))
sys.path.insert(0, str(KIT))
import q1_state
import q2_replay as original


def write_json(name, obj):
    (OUT/name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf8")


def part(data, mask):
    meta, t, z, stage = data
    return meta[mask].reset_index(drop=True), t[mask], z[mask], stage[mask]


def fit_windows(envs):
    tracks, _ = q1_state.load_tracks(KIT/"cache/canonical", ROOT/"Restocking_Problem_and_Data_EN")
    tracks = [tr for tr in tracks if tr["split"] == "train" and tr["environment_id"] in envs]
    cut = np.quantile(np.concatenate([tr["bone_deviation"][~tr["bad"]] for tr in tracks]), q1_state.GEOMETRY_QUANTILE)
    times, positions = [], []
    span = H + 2*L
    for tr in tracks:
        t, z = tr["t"], tr["points"]["wrist"]
        good = ~tr["bad"] & (tr["bone_deviation"] <= cut) & np.isfinite(z).all(axis=1)
        bad = np.r_[0, np.cumsum(~good)]
        gaps = np.r_[0, np.cumsum(np.diff(t) > .05)]
        starts = np.arange(0, len(t)-span+1, 4)
        starts = starts[(bad[starts+span] == bad[starts]) & (gaps[starts+span-1] == gaps[starts])]
        idx = starts[:, None] + np.arange(span)
        times.append(t[idx] - t[starts, None])
        positions.append(z[idx] - z[starts, None])
    return np.concatenate(times).astype("float32"), np.concatenate(positions).astype("float32")


class PathGRU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gru = nn.GRU(8, 40, batch_first=True)
        self.read = nn.Sequential(nn.Linear(40, 40), nn.SiLU(), nn.Linear(40, 16))

    def forward(self, t, z):
        gap = torch.diff(t, dim=1, prepend=t[:, :1])
        v = torch.diff(z, dim=1, prepend=z[:, :1]) / gap.clamp_min(.001)[:, :, None]
        fields = torch.cat(((z-z[:, -1:])/.15, v.clamp(-4., 4.)/.6,
                            (t-t[:, -1:])[:, :, None], gap[:, :, None]*5), dim=2)
        _, hidden = self.gru(fields)
        out = self.read(hidden[-1])
        return out[:, :12].reshape(-1, 4, 3), .35*F.softplus(out[:, 12:])+.004


def elapsed_basis(age):
    pieces = [(age-a).clip(0, b-a) for a,b in zip(KNOTS[:-1], KNOTS[1:])]
    pieces.append((age-KNOTS[-1]).clip(0, None))
    return torch.stack(pieces, -1) if isinstance(age, torch.Tensor) else np.stack(pieces, -1)


def fit_models(t, z, reuse):
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    times, pts = torch.from_numpy(t).to(dev), torch.from_numpy(z).to(dev)
    models = []
    for seed in SEEDS:
        torch.manual_seed(seed)
        rng = np.random.default_rng(seed)
        model = PathGRU().to(dev)
        file = OUT/f"path_gru_{seed}.pt"
        if reuse:
            model.load_state_dict(torch.load(file, map_location=dev, weights_only=True))
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=.0025, weight_decay=.001)
            lr = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 26, eta_min=.0003)
            for epoch in range(26):
                for _ in range(32):
                    n = 160
                    rows = torch.as_tensor(rng.integers(len(t), size=n), device=dev)[:, None]
                    count = rng.choice(3, n, p=[.4,.35,.25])
                    gap1, gap2 = rng.integers(1,16,size=(2,n))
                    history = np.broadcast_to(np.arange(H), (n,H)).copy()
                    # The warm prefix is followed by zero, one or two genuinely observed sparse samples.
                    for c in (1,2):
                        take = count == c
                        history[take,:H-c] = np.arange(c,H)
                        history[take,H-c] = H-1+gap1[take]
                        if c == 2:
                            history[take,-1] = H-1+gap1[take]+gap2[take]
                    past = torch.as_tensor(history, device=dev)
                    fut = past[:,-1:] + torch.arange(1,L+1,device=dev)
                    ht, hz = times[rows,past], pts[rows,past]
                    elapsed = times[rows,fut]-ht[:,-1:]
                    basis = elapsed_basis(elapsed)
                    vel, width = model(ht,hz)
                    pred = hz[:,-1:] + basis @ vel
                    err = ((pred-pts[rows,fut]).square().sum(-1)+.0005**2).sqrt()
                    scale = .002+(basis*width[:,None]).sum(-1)
                    loss = (err/(.025+.15*elapsed)).mean() + .15*(err.detach()/scale+scale.log()).mean()
                    opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), 2.)
                    opt.step()
                lr.step()
            torch.save(model.state_dict(), file)
        models.append(model.cpu().eval())
        print(f"GRU seed {seed} ready", flush=True)
    return models


@torch.inference_mode()
def encode(models, times, wrist):
    t = torch.from_numpy((times-times[:,-1:]).astype("float32"))
    z = torch.from_numpy((wrist-wrist[:,-1:]).astype("float32"))
    values = [net(t,z) for net in models]
    return tuple(np.mean([pair[j].numpy() for pair in values],axis=0) for j in range(2))


def rank95(scores):
    x = np.sort(np.asarray(scores))
    rank = int(np.ceil((len(x)+1)*(1-ALPHA)))
    return float(x[rank-1]) if rank <= len(x) else np.inf


def base_quantile(sorted_scores, alpha):
    if sorted_scores is None:
        return np.ones_like(alpha)
    return sorted_scores[np.ceil((len(sorted_scores)-1)*(1-alpha)).astype(int)]


def simulate(models, data, marginal, decision_factor, kind, setting):
    """Predictions stay filed at issue time. Access to masked wrist samples is restricted to delivery."""
    _, t, wrist, _ = data
    n = len(t)
    rows = np.arange(n)
    ht, hz = t[:,:H].copy(), wrist[:,:H].copy()
    vel, scale = encode(models, ht, hz)
    pending, count = np.full(n,-1), np.zeros(n,int)
    alpha = np.full(n,ALPHA)
    centers, base = np.empty((n,L,3)), np.empty((n,L))
    asked, delivery = np.zeros((n,L),bool), np.zeros((n,L),bool)
    for j in range(L):
        k = H+j
        arrived = rows[(pending >= 0) & (pending+DELAY == j)]
        if len(arrived):
            src = pending[arrived]
            returned = wrist[arrived,H+src]
            old_error = np.linalg.norm(returned-centers[arrived,src],axis=1)
            missed = old_error > base[arrived,src]
            alpha[arrived] = np.clip(alpha[arrived]+GAMMA*(ALPHA-missed),.01,.20)
            ht[arrived] = np.column_stack([ht[arrived,1:], t[arrived,H+src]])
            hz[arrived] = np.concatenate([hz[arrived,1:],returned[:,None]],axis=1)
            vel[arrived], scale[arrived] = encode(models, ht[arrived], hz[arrived])
            pending[arrived] = -1
            delivery[arrived,j] = True
        age = t[:,k]-ht[:,-1]
        basis = elapsed_basis(age)
        centers[:,j] = hz[:,-1]+np.einsum("nk,nkd->nd",basis,vel)
        q = base_quantile(marginal, alpha)
        base[:,j] = q*(.002+(basis*scale).sum(1))
        can = (pending < 0) & (count < BUDGET) & (j+1+DELAY <= L)
        if kind == "fixed":
            request = can & (j+1 in setting)
        elif kind in ("threshold", "paced"):
            # Public timing through the current tick only; account for travel time of a request.
            cadence = np.median(np.diff(t[:,k-5:k+1],axis=1),axis=1)
            at_return = elapsed_basis(age+DELAY*cadence)
            risk = decision_factor*q*(.002+(at_return*scale).sum(1))
            request = can & (risk >= setting)
            if kind == "paced":
                slot = 0 if j+1 <= L//2 else 1
                last = (slot+1)*(L//2)-DELAY
                request = can & (count == slot) & (j+1 <= last) & ((risk >= setting) | (j+1 == last))
        else:
            assert kind == "none"
            request = np.zeros(n,bool)
        pending[request] = j
        count += request
        asked[:,j] = request
    assert np.all(pending == -1) and np.all(count<=BUDGET)
    np.testing.assert_array_equal(asked[:,:-DELAY],delivery[:,DELAY:])
    if kind in ("fixed","paced"):
        assert np.all(count==BUDGET)
    return centers,base,asked


def errors(run, data):
    return np.linalg.norm(run[0]-data[2][:,H:],axis=2)


def outer_factors(run, data):
    scores = (errors(run,data)/run[1]).max(axis=1)
    per_record = pd.Series(scores).groupby(data[0]["sample"].to_numpy()).max()
    return max(1.,rank95(scores)), max(1.,rank95(per_record))


def from_old(data, mode, radii, qr):
    err, band, _, _, requested, _ = original.replay(data,mode,radii,*qr,budget=BUDGET,delay=DELAY)
    return err,band,requested


def metrics(data, name, err, base, requests, factor, record_factor, elapsed):
    meta,t,_,boundary = data
    radius = base*factor
    covered = err <= radius
    waiting = ((t[:,H+DELAY:]-t[:,H:-DELAY])*requests[:,:-DELAY]).sum(1)
    logs = meta.copy()
    logs["method"] = name
    logs["error_cm"], logs["endpoint_cm"] = err.mean(1)*100, err[:,-1]*100
    logs["path_covered"], logs["frame_coverage"] = covered.all(1),covered.mean(1)
    logs["radius_cm"], logs["base_radius_cm"] = radius.mean(1)*100, base.mean(1)*100
    logs["path_covered_base"] = (err <= base).all(1)
    logs["base_frame_coverage"] = (err <= base).mean(1)
    logs["within_10cm"] = (err <= .1).mean(1)
    logs["requests"],logs["waiting_s"] = requests.sum(1),waiting
    logs["cost"] = logs.requests  # One experimental cost unit per request.
    logs["request_frames"] = ["|".join(str(j+1) for j in np.flatnonzero(v)) for v in requests]
    logs["record_band_path_covered"] = (err<=base*record_factor).all(1)
    logs["record_band_radius_cm"] = base.mean(1)*record_factor*100
    cols = ["error_cm","endpoint_cm","path_covered","frame_coverage","radius_cm","base_radius_cm",
            "path_covered_base","base_frame_coverage","within_10cm","requests","cost","waiting_s",
            "record_band_path_covered","record_band_radius_cm"]
    group = logs.groupby("env")[cols].mean()
    summary = dict(method=name,split=meta.split.iat[0],windows=len(meta),records=meta["sample"].nunique(),
                   environments=len(group),**logs[cols].mean().to_dict(),env_macro_cm=group.error_cm.mean(),
                   p90_error_cm=100*np.quantile(err,.9),microsec_per_frame_batch=elapsed*1e6/err.size)
    horizon = pd.DataFrame(dict(method=name,split=meta.split.iat[0],frame=np.arange(1,L+1),
                                error_cm=100*err.mean(0),radius_cm=100*radius.mean(0),
                                coverage=covered.mean(0),within_10cm=(err<=.1).mean(0)))
    summary["boundary_error_cm"] = err[boundary].mean()*100 if boundary.any() else None
    summary["other_error_cm"] = err[~boundary].mean()*100 if (~boundary).any() else None
    return summary, group.reset_index().assign(method=name,split=meta.split.iat[0]), logs, horizon


def check_visibility(models, val, marginal, decision_factor, setting):
    d = part(val,np.arange(len(val[0]))<36)
    ref = simulate(models,d,marginal,decision_factor,"paced",setting)
    perturbed = d[2].copy()
    rng = np.random.default_rng(740)
    noise = rng.normal(0,2,size=perturbed[:,H:].shape)
    perturbed[:,H:] += noise*(~ref[2])[:,:,None]
    alternate = simulate(models,(d[0],d[1],perturbed,d[3]),marginal,decision_factor,"paced",setting)
    for a,b in zip(ref,alternate):
        np.testing.assert_array_equal(a,b)
    # Corrupt an actual queried sample; outputs and requests through the tick before its arrival must agree.
    perturbed = d[2].copy()
    first = np.argmax(ref[2],axis=1)
    perturbed[np.arange(len(first)),H+first] += rng.normal(0,.4,size=(len(first),3))
    alternate = simulate(models,(d[0],d[1],perturbed,d[3]),marginal,decision_factor,"paced",setting)
    unchanged = np.arange(L)[None,:] < first[:,None]+DELAY
    for a,b in zip(ref,alternate):
        np.testing.assert_array_equal(a[unchanged],b[unchanged])


TITLES = {"hold":"保持腕点、无请求","damped":"速度衰减外推、无请求","fixed":"原衰减模型、固定请求",
          "trigger":"原衰减模型、原半径触发","gru_none":"GRU、无请求","gru_fixed":"GRU、原固定时刻",
          "gru_fixed_tuned":"GRU、验证选择固定时刻","gru_direct":"GRU、直接按 10 厘米触发",
          "gru_paced":"GRU、按整段校准半径并分配两次请求"}


def report(summary, lock):
    v=summary[summary.split.eq("validation")].set_index("method")
    t=summary[summary.split.eq("test")].set_index("method")
    lines=["第二问　GRU 腕点轨迹预测与整段误差校准", "",
           f"同一 GRU 在验证选定的固定时刻请求时，环境等权平均位置误差为验证 {v.at['gru_fixed_tuned','env_macro_cm']:.3f}、"
           f"回顾测试 {t.at['gru_fixed_tuned','env_macro_cm']:.3f} 厘米。按预测范围给两个半窗各分配一次请求时，"
           f"对应误差为 {v.at['gru_paced','env_macro_cm']:.3f}、{t.at['gru_paced','env_macro_cm']:.3f} 厘米。"
           f"两者回顾测试整窗覆盖分别为 {t.at['gru_fixed_tuned','path_covered']:.1%}、{t.at['gru_paced','path_covered']:.1%}。",
           ("分段触发通过了验证采用条件。" if lock["validation_adopt_paced"] else
            "验证结果支持用固定时刻请求作为 GRU 的主方案，整窗校准可用于报告误差范围。"), "",
           "在合格真实腕点记录上人工遮挡。每窗前 12 帧可见，随后遮挡 30 帧，最多请求两次，每次花费 1 个实验成本单位，"
           "采样后三帧返回。各方法只获取腕点历史、公共时钟及已返回的请求。人工遮挡窗口长度预先公布。"
           "验证和回顾测试分别有 200、376 个非重叠手部窗口，均来自 7 条记录、4 个环境。",
           "完整轨迹覆盖要求一个窗口的 30 帧三维腕点全部落入逐帧发出的球形范围。校准目标 95%。"
           "误差对照合格记录，平均半径为所有时刻所报半径的算术平均。",
           f"GRU 使用训练集内 9 个环境拟合，另 3 个训练环境建立逐帧尺度及开发用无请求预测带。剩余 4 个训练环境的 "
           f"{lock['calibration_windows']} 个窗口仅用于最终整路径校准。外层使用每窗最大的归一化三维误差。"
           "相同记录和环境内的窗口有关联，跨新环境的 95% 有限样本覆盖保证不能由本回放推出。",
           "模型在收到延迟坐标后，把它与该采样时刻已发出的预测比较，更新名义误差水平和 GRU 的观测历史。"
           "从不回改已发出的历史预测。时机由预计返回时的无新观测半径决定。",
           f"原固定时刻为第 8、19 帧，验证择优固定时刻为第 {lock['fixed_frames'][0]}、{lock['fixed_frames'][1]} 帧。"
           f"分段策略使用最终校准后的 {lock['deployed_threshold_cm']:.2f} 厘米触发阈值，每半窗留一次请求，最迟在各半窗能按时返回的最后时刻发出。"
           "独立外层校准后把触发阈值作同尺度换算，以保持校准时和部署时的请求规则一致。"
           "直接按 10 厘米触发的对照使用开发集已冻结的整段半径，到达阈值且有预算便请求。", ""]
    for sp,title in (("validation","验证"),("test","回顾测试")):
        t=summary[summary.split.eq(sp)].set_index("method")
        lines += [title,"方案｜平均误差（厘米）｜环境等权误差（厘米）｜整窗覆盖｜平均半径（厘米）｜请求次数/窗｜等待秒数/窗"]
        for name,label in TITLES.items():
            v=t.loc[name]
            lines.append(f"{label}｜{v.error_cm:.3f}｜{v.env_macro_cm:.3f}｜{v.path_covered:.1%}｜{v.radius_cm:.2f}｜{v.requests:.3f}｜{v.waiting_s:.3f}")
        v=t.loc["gru_paced"]
        lines.append(f"同一组分段请求和点预测，外层整窗校准前后，逐帧覆盖 {v.base_frame_coverage:.1%} → {v.frame_coverage:.1%}，"
                     f"整窗覆盖 {v.path_covered_base:.1%} → {v.path_covered:.1%}，平均半径 {v.base_radius_cm:.2f} → {v.radius_cm:.2f} 厘米。")
        lines.append("")
    lines += ["所有阈值和请求规则在回顾测试前冻结。95% 指本实验的目标水平，30 帧的整窗覆盖、逐帧覆盖和腕点误差分别计量。"
              "原衰减模型使用既有逐帧半径。保持腕点对照采用原回放的保持法逐帧半径。"
              "另一项保守的核对用每条校准记录跨窗口的最大误差，见数值表的 record_band 列。"
              "计算耗时是本机按批回放的每手帧时间，不包含离线训练。",
              "自然遮挡区间的坐标缺少独立参考，本试验的误差及覆盖率限于合格记录的人工遮挡。",
              "方法依据为 Zhou、Lindemann、Sesia 在 ICML 2024 发表的 Conformalized Adaptive Forecasting of Heterogeneous Trajectories。"
              "本题使用延迟反馈、三维欧氏球、预算约束与请求调度，属于针对本题的改造。",
              "运行 q2_gru_cafht.py 可复算主实验，使用 --from-weights 可加载已保存的三个 GRU。运行 q2_gru_cafht_controls.py 可复算补充对照。"]
    (OUT/"实测结论.txt").write_text("\n".join(lines)+"\n",encoding="utf8")


def run(reuse=False):
    OUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(3)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    data,_ = original.prepare(ROOT)
    meta = data[0]
    assert (H,L,BUDGET,DELAY)==(original.HISTORY,original.LENGTH,2,3)
    train_envs=np.random.default_rng(80523).permutation(sorted(meta.loc[meta.split.eq("train"),"env"].unique()))
    sets={"fit":train_envs[:9],"pilot":train_envs[9:12],"calibration":train_envs[12:]}
    pilot,cal=[part(data,meta.env.isin(sets[name]).to_numpy()) for name in ("pilot","calibration")]
    val=part(data,meta.split.eq("validation").to_numpy())
    assert len(val[0])==200 and len(cal[0])==296
    thresholds=[.10,.15,.20,.25,.30,.40,.50,.65]
    calendars=[(a,b) for a in (5,8,11) for b in (17,19,22,25)]
    write_json("protocol.json",dict(history_frames=H,hidden_frames=L,budget=BUDGET,return_delay_frames=DELAY,
        cost_per_request=1,position_tolerance_cm=10,nominal_window_coverage=.95,feedback_gamma=GAMMA,
        split_seed=80523,environment_sets={k:v.tolist() for k,v in sets.items()},gru_seeds=SEEDS,
        gru_epochs=26,risk_thresholds_m=thresholds,fixed_calendars=calendars,
        objective="minimize mean Euclidean wrist error, equal weight per validation environment, under a two-request cap",
        adoption="Prefer the paced policy if it improves validation environment-macro error over both tuned fixed GRU and archived fixed damped, attains 95% empirical path coverage and respects budget. Test is retrospective.",
        scope="Controlled full wrist masking. The primary comparator shares the exact GRU, returned inputs, budget, delay and window rules."))
    tfit,zfit=fit_windows(sets["fit"])
    print(f"Fitting on {len(tfit)} training clips / {len(sets['fit'])} environments",flush=True)
    models=fit_models(tfit,zfit,reuse)
    raw=simulate(models,pilot,None,1.,"none",None)
    normalized=errors(raw,pilot)/raw[1]
    marginal=np.sort(normalized.ravel())
    q0=float(base_quantile(marginal,np.array([ALPHA]))[0])
    decision_factor=max(1.,rank95(normalized.max(1)/q0))
    np.save(OUT/"pilot_point_scores.npy",marginal)
    grid=[]
    for kind, choices in (("fixed",calendars),("paced",thresholds)):
        for choice in choices:
            issued=simulate(models,val,marginal,decision_factor,kind,choice)
            mae=100*errors(issued,val).mean(1)
            macro=float(pd.Series(mae).groupby(val[0].env.to_numpy()).mean().mean())
            grid.append(dict(kind=kind,setting=choice,macro_cm=macro,pooled_cm=float(mae.mean()),
                             mean_queries=float(issued[2].sum(1).mean())))
    grid=pd.DataFrame(grid)
    grid.to_csv(OUT/"validation_policy_grid.csv",index=False)
    best=grid.sort_values(["macro_cm","pooled_cm"],kind="stable").groupby("kind",sort=False).head(1).set_index("kind")
    calendar,threshold=best.at["fixed","setting"],best.at["paced","setting"]
    policies={"gru_none":("none",None),"gru_fixed":("fixed",(8,19)),"gru_fixed_tuned":("fixed",calendar),
              "gru_direct":("threshold",.10),"gru_paced":("paced",threshold)}
    factors={}
    for name,(kind,setting) in policies.items():
        issued=simulate(models,cal,marginal,decision_factor,kind,setting)
        window_q,record_q=outer_factors(issued,cal)
        factors[name]=(decision_factor if name=="gru_direct" else window_q,record_q)
    deployed=threshold*factors["gru_paced"][0]/decision_factor
    check_visibility(models,val,marginal,factors["gru_paced"][0],deployed)
    with ZipFile(ARCHIVE) as zipfile:
        stored=pd.read_csv(zipfile.open("results/q2_windows.csv"),keep_default_na=False)
        radii=pd.read_csv(zipfile.open("results/q2_radii.csv"))
    def radius_table(mode):
        rr=radii[radii["mode"].eq(mode)].assign(g=lambda x:x.regime.map(dict(slow=0,middle=1,fast=2)))
        return rr.sort_values(["g","age_frames"]).radius95_mm.to_numpy().reshape(3,L)/1000
    params=json.loads((KIT/"results/q1_real/summary.json").read_text(encoding="utf8"))["fitted_parameters"]["wrist"]["cv_kalman"]
    qr=(params["q"],params["r"])
    lock=dict(fit_clips=len(tfit),pilot_windows=len(pilot[0]),calibration_windows=len(cal[0]),
        calibration_records=cal[0]["sample"].nunique(),fixed_frames=calendar,development_threshold_cm=threshold*100,
        deployed_threshold_cm=100*deployed,pilot_operational_factor=decision_factor,
        policy_factors=factors,grid_selection_metric="validation environment-equal mean wrist error")
    summaries,envs,logs,horizons=[],[],[],[]
    for split in ("validation","test"):
        dataset=val if split=="validation" else part(data,meta.split.eq("test").to_numpy())
        assert dataset[0].env.nunique()==4
        for name in TITLES:
            if name in policies:
                kind,setting=policies[name]
                final_q,record_q=factors[name]
                operational=final_q if name=="gru_paced" else decision_factor
                start=time.perf_counter()
                issued=simulate(models,dataset,marginal,operational,kind,deployed if name=="gru_paced" else setting)
                runtime=time.perf_counter()-start
                err,band,asks=errors(issued,dataset),issued[1],issued[2]
                if name=="gru_paced":
                    old_run=simulate(models,dataset,marginal,decision_factor,kind,setting)
                    for a,b in zip(issued,old_run):
                        np.testing.assert_array_equal(a,b)
            else:
                radius=radius_table("hold" if name=="hold" else "damped")
                start=time.perf_counter()
                err,band,asks=from_old(dataset,name,radius,qr)
                runtime=time.perf_counter()-start
                final_q=record_q=1.
                past=stored[stored.split.eq(split)&stored["mode"].eq(name)&stored.budget.eq(BUDGET)&stored.delay_frames.eq(DELAY)]
                match=dataset[0].merge(past,on=list(dataset[0].columns),how="left",validate="one_to_one")
                np.testing.assert_allclose(err.mean(1)*1000,match.MAE3D_mm,atol=1e-7)
                exact=["|".join(str(i+1) for i in np.flatnonzero(rr)) for rr in asks]
                assert exact==match.request_frames.astype(str).tolist()
            s,env,log,hor=metrics(dataset,name,err,band,asks,final_q,record_q,runtime)
            summaries.append(s)
            envs.append(env)
            logs.append(log)
            horizons.append(hor)
        recent=pd.DataFrame(summaries).query("split==@split").set_index("method")
        if split=="validation":
            m=recent.loc["gru_paced"]
            lock["validation_adopt_paced"]=bool(m.env_macro_cm < min(recent.at["fixed","env_macro_cm"],
                    recent.at["gru_fixed_tuned","env_macro_cm"]) and m.path_covered>=1-ALPHA and m.requests<=BUDGET)
            write_json("validation_lock.json",lock)
        print(split,recent[["error_cm","env_macro_cm","path_covered","radius_cm","requests"]].round(3).to_string(),flush=True)
    table=pd.DataFrame(summaries)
    table.to_csv(OUT/"summary.csv",index=False)
    pd.concat(envs,ignore_index=True).to_csv(OUT/"by_environment.csv",index=False)
    pd.concat(logs,ignore_index=True).to_csv(OUT/"per_window.csv.gz",index=False,compression="gzip")
    pd.concat(horizons,ignore_index=True).to_csv(OUT/"by_horizon.csv",index=False)
    report(table,lock)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-weights",action="store_true")
    run(parser.parse_args().from_weights)
