"""Q1: thirty wrist predictions issued together, scored against the original point model."""
import argparse
import json

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F

from acceleration_data import OFFSETS
from acceleration_evaluation import Policy
from endpoint_decision import Rule
from motion_data import KIT, ROOT, SCALE, local, predictors, rotation_matrices
from palm_point_evaluation import OUT as PALM, issue, make_arm
from similar_motion_evaluation import nonoverlap_mask
from q2_gru_cafht import (ALPHA, GAMMA, H as Q2_H, OUT as Q2_OUT, PathGRU, SEEDS as Q2_SEEDS,
                          elapsed_basis, encode, rank95)
from q1_state import kalman_grid

OUT = ROOT / "分析报告" / "第一问GRU轨迹与整段校准"
L = 30
MIN = predictors.MIN_ORIGIN
STEPS = np.arange(1,L+1)
SEEDS = (7341,7342,7343)
PATHS = ("gru","linear_current","anchored_gru","anchored_damped")
NAMES = {"current":"现有腕点模型","hold":"保持腕点","q2_transfer":"直接沿用第二问腕点 GRU",
         "gru":"第一问专训轨迹 GRU","linear_current":"现有终点、直线补出中间位置",
         "anchored_gru":"现有终点、GRU 提供轨迹形状","damped":"原速度衰减卡尔曼外推",
         "anchored_damped":"现有终点、速度衰减提供轨迹形状"}
BANDS = {"pointwise":"逐帧分别校准","joint_static":"整段校准、固定分位数",
         "joint_adaptive":"CAFHT 式整段校准、自适应分位数","flat":"整段恒定半径"}


def dump(name, obj):
    (OUT/name).write_text(json.dumps(obj,indent=2,ensure_ascii=False,allow_nan=False),encoding="utf8")


class Paths:
    def __init__(self, data):
        self.meta = data.meta
        n = len(self.meta)
        self.y = np.empty((n,L,3),np.float32)
        self.mask = np.zeros((n,L),bool)
        self.ok = np.zeros(n,bool)
        self.dense = np.empty((n,54),np.float32)
        self.cadence = np.empty(n,np.float32)
        self.tracks = {}
        columns=["sample","frame_index","head_qx","head_qy","head_qz","head_qw"]
        frames=pd.read_parquet(KIT/"cache/canonical/frames.parquet",columns=columns)
        head={key:rotation_matrices(g.sort_values("frame_index")) for key,g in frames.groupby("sample",sort=False)}
        tracks=predictors.load(KIT/"cache/canonical",ROOT/"Restocking_Problem_and_Data_EN")
        by_track=self.meta.groupby(["sample","hand"],sort=False).indices
        for tr in tracks:
            key=(tr["sample"],tr["hand"])
            rows=by_track[key]
            o=np.arange(len(rows))
            np.testing.assert_array_equal(rows,rows[0]+o)
            np.testing.assert_array_equal(self.meta.origin.to_numpy()[rows],o)
            t,z,rot=tr["t"],tr["points"]["wrist"],head[key[0]]
            future=o[:,None]+STEPS
            safe=np.minimum(future,len(rows)-1)
            self.y[rows]=local(z[safe]-z[:,None],rot)
            self.mask[rows]=(future<len(rows)) & tr["qualified"][safe] & (o[:,None]>=MIN) & np.isfinite(tr["speed5"][:,None])
            self.ok[rows]=~tr["bad"]
            hist=np.maximum(0,o[:,None]-np.arange(10,-1,-1))
            disp=local(z[hist]-z[:,None],rot)/SCALE
            gaps=np.diff(t[hist],axis=1)
            self.dense[rows]=np.column_stack((disp.reshape(len(o),-1),np.clip(gaps*30,0,5),tr["bad"][hist]))
            self.cadence[rows]=pd.Series(np.r_[np.nan,np.diff(t)]).rolling(5,min_periods=1).median().fillna(1/30).to_numpy()
            self.tracks[key]=(rows,t,z,rot)
        np.testing.assert_array_equal(self.mask[:,-1],data.meta.score)
        np.testing.assert_allclose(self.y[:,-1]/SCALE,data.truth,atol=2e-6)
        self.grid=(self.meta.origin.to_numpy()-MIN)%L==0


class WristPath(nn.Module):
    """Decode a continuous, origin-anchored displacement in the head axes at issue time."""
    def __init__(self, ndim, dense_dim):
        super().__init__()
        self.encode=nn.Sequential(nn.Linear(ndim,48),nn.LayerNorm(48),nn.SiLU())
        self.gru=nn.GRU(48,56,num_layers=2,dropout=.12,batch_first=True)
        self.recent=nn.Sequential(nn.Linear(dense_dim,32),nn.LayerNorm(32),nn.SiLU())
        self.output=nn.Sequential(nn.Linear(56*2+32,80),nn.LayerNorm(80),nn.SiLU(),nn.Dropout(.10),nn.Linear(80,16))

    def forward(self, seq, dense, basis):
        history,_=self.gru(self.encode(seq))
        out=self.output(torch.cat((history[:,-1],history.mean(1),self.recent(dense)),dim=1))
        pred=basis @ out[:,:12].reshape(-1,4,3)
        scale=.002+(basis*(.35*F.softplus(out[:,12:])+.004)[:,None]).sum(2)
        return pred,scale


class Batches:
    def __init__(self, data, path, device):
        self.device=device
        self.x=torch.as_tensor(data.sequence,device=device)
        self.left=torch.as_tensor(data.start,device=device)
        self.offsets=torch.as_tensor(OFFSETS,device=device)
        self.dense=torch.as_tensor(path.dense,device=device)
        self.y=torch.as_tensor(path.y,device=device)
        self.mask=torch.as_tensor(path.mask,device=device)
        cadence=torch.as_tensor(path.cadence,device=device)
        self.basis=elapsed_basis(cadence[:,None]*torch.arange(1,L+1,device=device))

    def inputs(self,ids):
        timeline=torch.maximum(ids[:,None]-self.offsets,self.left[ids,None])
        return self.x[timeline],self.dense[ids],self.basis[ids]


def train_net(batches, ids, seed, reuse):
    torch.manual_seed(seed)
    rng=np.random.default_rng(seed)
    dev=batches.device
    model=WristPath(batches.x.shape[1],batches.dense.shape[1]).to(dev)
    filename=OUT/f"q1_path_gru_{seed}.pt"
    if reuse:
        model.load_state_dict(torch.load(filename,map_location=dev,weights_only=True))
        return model.eval()
    opt=torch.optim.AdamW(model.parameters(),lr=.0014,weight_decay=.003)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,24,eta_min=.00018)
    # Every sampled origin has a qualified 30th frame. Other horizons have independent masks.
    horizon_weight=1/(.025+.16*torch.arange(1,L+1,device=dev)/L)
    for epoch in range(24):
        model.train()
        for choose in np.array_split(rng.choice(ids,9216,replace=False),36):
            ix=torch.as_tensor(choose,device=dev)
            pred,scale=model(*batches.inputs(ix))
            err=((pred-batches.y[ix]).square().sum(2)+.0005**2).sqrt()
            mask=batches.mask[ix]
            path_loss=(err*horizon_weight)[mask].mean()
            end_loss=err[:,-1].mean()/.185
            scale_loss=(err.detach()/scale+scale.log())[mask].mean()
            loss=.5*(path_loss+end_loss)+.12*scale_loss
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(),3.)
            opt.step()
        scheduler.step()
    torch.save(model.state_dict(),filename)
    return model.eval()


@torch.inference_mode()
def predict(nets,batches,ids):
    paths,scales=[],[]
    for batch in np.array_split(ids,max(1,int(np.ceil(len(ids)/384)))):
        args=batches.inputs(torch.tensor(batch,device=batches.device,dtype=torch.long))
        outputs=[net(*args) for net in nets]
        paths.append(torch.stack([v[0] for v in outputs]).mean(0).cpu().numpy())
        scales.append(torch.stack([v[1] for v in outputs]).mean(0).cpu().numpy())
    return np.concatenate(paths),np.concatenate(scales)


def transfer(paths, ids, q2_models):
    result=np.empty((len(ids),L,3),np.float32)
    for rows,t,z,rot in paths.tracks.values():
        lo,hi=np.searchsorted(ids,[rows[0],rows[-1]+1])
        if lo==hi:
            continue
        local_ids=ids[lo:hi]-rows[0]
        for batch in np.array_split(np.arange(lo,hi),max(1,int(np.ceil((hi-lo)/384)))):
            origins=local_ids[batch-lo]
            hist=np.maximum(0,origins[:,None]-np.arange(Q2_H-1,-1,-1))
            vel,_=encode(q2_models,t[hist],z[hist])
            basis=elapsed_basis(paths.cadence[rows[origins],None]*STEPS)
            result[batch]=local(basis @ vel,rot[origins])
    return result


def damped_control(paths,ids):
    params=json.loads((KIT/"results/q1_real/summary.json").read_text(encoding="utf8"))["fitted_parameters"]["wrist"]
    cv,tau=params["cv_kalman"],params["damped_kalman"]["tau"]
    result=np.empty((len(ids),L,3),dtype=np.float32)
    for rows,t,z,rot in paths.tracks.values():
        lo,hi=np.searchsorted(ids,[rows[0],rows[-1]+1])
        if lo==hi:
            continue
        _,(p,v,_)=kalman_grid(t,z,np.zeros(len(t),bool),cv["q"],cv["r"],1.,store=True)
        o=ids[lo:hi]-rows[0]
        elapsed=paths.cadence[ids[lo:hi],None]*STEPS
        delta=(p[o]-z[o])[:,None]+v[o,None]*(-tau*np.expm1(-elapsed/tau))[:,:,None]
        result[lo:hi]=local(delta,rot[o])
    return result


def candidates(split, arm, paths, batch, nets, q2_nets, frozen):
    prior,rule,retrieval=frozen
    ids,original=issue(arm,split,prior,rule,retrieval,scored_only=False)
    raw,scale=predict(nets,batch,ids)
    true=paths.y[ids]
    endpoint=original*SCALE
    linear=(STEPS/L)[None,:,None]*endpoint[:,None,:]
    corrected=raw+(STEPS/L)[None,:,None]*(endpoint-raw[:,-1])[:,None,:]
    damped=damped_control(paths,ids)
    damped_shape=damped+(STEPS/L)[None,:,None]*(endpoint-damped[:,-1])[:,None,:]
    np.testing.assert_allclose(corrected[:,-1],endpoint,atol=2e-7)
    np.testing.assert_allclose(damped_shape[:,-1],endpoint,atol=2e-7)
    preds={"gru":raw,"linear_current":linear,"anchored_gru":corrected,"hold":np.zeros_like(raw),
           "q2_transfer":transfer(paths,ids,q2_nets),"damped":damped,"anchored_damped":damped_shape}
    err={k:np.linalg.norm(x-true,axis=2) for k,x in preds.items()}
    err["current"]=np.linalg.norm(endpoint-true[:,-1],axis=1)[:,None]
    g=arm.data.meta.loc[ids].reset_index(drop=True)
    archive=pd.read_csv(PALM/"per_origin.csv.gz",usecols=["split","env","sample","hand","origin","wrist"])
    mask=paths.mask[ids,-1]
    expected=archive[archive.split.eq(split)].reset_index(drop=True)
    keys=["split","env","sample","hand","origin"]
    pd.testing.assert_frame_equal(g.loc[mask,keys].reset_index(drop=True),expected[keys])
    np.testing.assert_allclose(err["current"][mask,-1]*100,expected.wrist,atol=1e-4)
    return ids,g,err,scale


def macro(x, env):
    return float(pd.Series(x).groupby(np.asarray(env)).mean().mean())


def point_metrics(paths,ids,g,err):
    mask=paths.mask[ids]
    full=mask.all(1)
    fast=(g.slow & g.end_bin.eq(2)).to_numpy() & mask[:,-1]
    slow=(g.slow & g.end_bin.eq(0)).to_numpy() & mask[:,-1]
    allslow=g.slow.to_numpy() & mask[:,-1]
    spaced=full & paths.grid[ids]
    rows=[]
    for name,v in err.items():
        last=100*v[:,-1]
        row=dict(method=name,split=g.split.iat[0],n=int(mask[:,-1].sum()),fast_n=int(fast.sum()),
                 endpoint_macro_cm=macro(last[mask[:,-1]],g.env[mask[:,-1]]),endpoint_pooled_cm=float(last[mask[:,-1]].mean()),
                 fast_cm=float(last[fast].mean()),slow_cm=float(last[slow].mean()),allslow_cm=float(last[allslow].mean()))
        if v.shape[1]==L:
            row.update(path_n=int(full.sum()),path_macro_cm=macro(100*v[full].mean(1),g.env[full]),
                       spaced_n=int(spaced.sum()),spaced_path_cm=macro(100*v[spaced].mean(1),g.env[spaced]))
            for h in (1,3,10,30):
                ok=mask[:,h-1]
                row[f"frame{h}_cm"]=macro(v[ok,h-1]*100,g.env[ok])
        rows.append(row)
    return pd.DataFrame(rows)


def adaptive_bases(g, error, scale, ok, reference):
    """At o, update only with the endpoint issued at o-30 and observed by o."""
    nref=len(reference)
    initial=int(np.ceil((1-ALPHA)*(nref-1)))
    static=scale*reference[initial]
    levels=np.full(len(g),ALPHA)
    for rows in g.groupby(["sample","hand"],sort=False).indices.values():
        np.testing.assert_array_equal(g.origin.to_numpy()[rows],np.arange(len(rows)))
        for o in range(MIN+L,len(rows),L):
            old,now=rows[o-L],rows[o]
            level=levels[old]
            if ok[now]:
                q=reference[int(np.ceil((1-level)*(nref-1))),-1]
                miss=error[old,-1] > scale[old,-1]*q
                level=np.clip(level+GAMMA*(ALPHA-float(miss)),.01,.20)
            levels[rows[o:o+L]]=level
    q=reference[np.ceil((1-levels)*(nref-1)).astype(int)]
    return static,scale*q


def calibration_parameters(mask,error,static,adaptive):
    n=int(np.sum(mask))
    rank=int(np.ceil((n+1)*(1-ALPHA)))-1
    if rank>=n:
        raise ValueError("There are too few windows for a finite empirical 95% radius.")
    ratio=error[mask]/static[mask]
    return dict(n=n,pointwise=np.sort(ratio,axis=0)[rank].tolist(),
                joint_static=max(1.,rank95(ratio.max(1))),
                joint_adaptive=max(1.,rank95((error[mask]/adaptive[mask]).max(1))),
                flat=rank95(error[mask].max(1)))


def band_metrics(path_name,ids,g,error,scale,paths,reference,frozen=None):
    full=paths.mask[ids].all(1)
    grid=full & paths.grid[ids]
    static,adaptive=adaptive_bases(g,error,scale,paths.ok[ids],reference)
    # An edited future observation may affect later forecasts but never a band issued by the cutoff.
    track=next(iter(g.groupby(["sample","hand"],sort=False).indices.values()))
    cut=MIN+2*L+7
    future=g.origin.to_numpy()[track,None]+STEPS>cut
    altered=np.where(future,1000.,error[track])
    altered_ok=paths.ok[ids[track]].copy()
    altered_ok[cut+1:]=False
    _,check=adaptive_bases(g.iloc[track].reset_index(drop=True),altered,scale[track],altered_ok,reference)
    np.testing.assert_array_equal(check[:cut+1],adaptive[track[:cut+1]])
    radii={key:np.empty_like(error,dtype=np.float64) for key in BANDS}
    env=g.env.to_numpy()
    if frozen is None:
        # Outer calibration for an evaluated validation environment excludes that entire environment.
        folds=[(env==value,calibration_parameters(grid & (env!=value),error,static,adaptive)) for value in sorted(set(env))]
        full_parameters=calibration_parameters(grid,error,static,adaptive)
    else:
        folds=[(np.ones(len(g),bool),frozen)]
        full_parameters=frozen
    for mask,params in folds:
        radii["pointwise"][mask]=static[mask]*params["pointwise"]
        radii["joint_static"][mask]=static[mask]*params["joint_static"]
        radii["joint_adaptive"][mask]=adaptive[mask]*params["joint_adaptive"]
        radii["flat"][mask]=params["flat"]
    rows=[]
    logs=g.loc[grid,["split","env","sample","hand","origin","slow","end_bin"]].reset_index(drop=True).copy()
    logs["method"]=path_name
    logs["max_error_cm"]=100*error[grid].max(1)
    logs["path_mean_error_cm"]=error[grid].mean(1)*100
    for scheme,band in radii.items():
        hit=error<=band
        covered=hit.all(1)
        logs[f"{scheme}_covered"]=covered[grid]
        logs[f"{scheme}_radius_cm"]=band[grid].mean(1)*100
        for sample,use in (("grid",grid),("all_complete",full)):
            for scope,subset in (("all",use),("slow_to_fast",use & g.slow.to_numpy() & g.end_bin.eq(2).to_numpy())):
                indices=np.flatnonzero(subset)
                for group,ix in [("all",indices),*((v,indices[env[indices]==v]) for v in sorted(set(env[indices])))]:
                    rows.append(dict(split=g.split.iat[0],method=path_name,band=scheme,sampling=sample,scope=scope,env=group,
                                n=len(ix),covered=int(covered[ix].sum()),path_coverage=covered[ix].mean(),
                                frame_coverage=hit[ix].mean(),mean_radius_cm=100*band[ix].mean(),
                                endpoint_coverage=hit[ix,-1].mean(),endpoint_radius_cm=100*band[ix,-1].mean(),
                                env_macro_coverage=macro(covered[ix],env[ix]),path_error_cm=100*error[ix].mean()))
    return pd.DataFrame(rows),logs,full_parameters


def slow_fast_sensitivity(endpoints):
    rows=[]
    for split,group in endpoints.groupby("split",sort=False):
        slow=group[group.slow].reset_index(drop=True)
        keep=nonoverlap_mask(slow)
        for sample,mask in (("all",slow.end_bin.eq(2)),("spaced_slow_starts",keep & slow.end_bin.eq(2))):
            sub=slow[mask]
            names=["current","gru"]
            by_env=sub.groupby("env")[names].mean()
            by_record=sub.groupby("sample")[names].mean()
            rows.append(dict(split=split,sampling=sample,n=len(sub),current_cm=sub.current.mean(),gru_cm=sub.gru.mean(),
                             record_macro_current_cm=by_record.current.mean(),record_macro_gru_cm=by_record.gru.mean(),
                             environments=len(by_env),gru_better_envs=int((by_env.gru<by_env.current).sum())))
    return pd.DataFrame(rows)


def report(points,bands,lock,sensitivity):
    chosen=lock["path"]
    with_control=lock["path_with_physical_control"]
    v=points[points.split.eq("validation")].set_index("method")
    t=points[points.split.eq("test")].set_index("method")
    lines=["第一问　单次发出未来 30 帧的腕点轨迹与误差范围","",
           f"专训 GRU 把慢→快的终点误差从验证 {v.at['current','fast_cm']:.2f}、回顾测试 {t.at['current','fast_cm']:.2f} 厘米，"
           f"变为 {v.at['gru','fast_cm']:.2f}、{t.at['gru','fast_cm']:.2f} 厘米。验证的慢→慢误差从 "
           f"{v.at['current','slow_cm']:.2f} 变为 {v.at['gru','slow_cm']:.2f} 厘米。"
           f"主实验在独立 GRU 与保留原终点的两种路径之间按验证规则选定「{NAMES[chosen]}」。",
           f"加入原速度衰减的补充对照后，按同一验证路径误差标准可采用「{NAMES[with_control]}」。"
           f"固定同一个终点时，GRU 轨迹形状的整段误差为验证 {v.at['anchored_gru','path_macro_cm']:.3f}、"
           f"回顾测试 {t.at['anchored_gru','path_macro_cm']:.3f} 厘米，速度衰减形状分别为 "
           f"{v.at['anchored_damped','path_macro_cm']:.3f}、{t.at['anchored_damped','path_macro_cm']:.3f} 厘米。"
           "各形状的随步长变化的误差尺度都取自专训 GRU 的尺度输出。","",
           "状态为左、右腕点。每个起点只读取当前及过去记录，一次给出未来 30 帧全部位置与半径。"
           "预测期间保持这一整条输出。第一问可连续获得原始观测，因此新的起点可以使用当时已经获得的数据。",
           "现有模型是质量感知的神经增益预测，加慢速起点的运动 GRU 与相似动作修正，原生输出为第 30 帧。"
           "直线对照在当前腕点和该预测终点间等分 30 步。形状对照给专训 GRU 的每一步加上按步数线性增长的终点差，"
           "从而让它的第 30 帧精确重合于现有模型预测。三个方案都没有使用将来的实际腕点作锚。",
           "主实验完成后，以第一版已拟合的速度衰减卡尔曼外推（时间常数 0.45 秒）补做了简单运动模型的形状对照。"
           "采用完全相同的预测终点和按步数分配的终点差，无新增可调参数。",
           "专训 GRU 使用全部 16 个训练环境、55 条记录，输入为最多 2 秒的头部、双手、手指、质量和近期腕点观测。"
           "它只从运动历史预测连续轨迹，不输入旧模型终点。沿用 3 个固定种子，逐帧速度和尺度按预测起点之前的采样间隔换算。"
           "另对照第二问原权重直接迁移（仅用腕点、只在部分训练环境拟合）。",
           "第 30 帧终点只要求该参考帧合格，维持原有全部评分起点。总体误差以环境等权，慢→快和慢→慢以起点等权。"
           "整段误差与覆盖只统计未来 30 帧的参考均合格的窗口。主覆盖表在每条记录、每只手从第 10 号起点开始每隔 30 帧固定取窗。",
           "训练集提供各预测步的归一化误差经验分布。自适应更新只在过去选定起点的第 30 帧已到达、且该时刻原始质量正常时进行。"
           "每种预测路径分别用窗口内最大标准化三维误差进行外层校准。验证覆盖按环境留一；回顾测试前使用全部 4 个验证环境锁定比例。"
           "这里报告 30 帧固定起点的经验同时覆盖。7 条验证记录来自 4 个环境，窗口仍有组内相关性。",
           ""]
    for split,title in (("validation","验证"),("test","回顾测试")):
        p=points[points.split.eq(split)].set_index("method")
        lines.extend([f"{title}　共有 {int(p.at['current','n'])} 个原终点评分起点，其中 {int(p.at['current','fast_n'])} 个慢→快。",
                      "方法｜第 30 帧总体｜慢→快｜慢→慢｜整段平均（环境等权，厘米）"])
        for name in ("current","hold","q2_transfer","gru","linear_current","anchored_gru","damped","anchored_damped"):
            cell=p.loc[name]
            path_value=f"{cell.path_macro_cm:.3f}" if np.isfinite(cell.path_macro_cm) else "原生模型只出终点"
            lines.append(f"{NAMES[name]}｜{cell.endpoint_macro_cm:.3f}｜{cell.fast_cm:.3f}｜{cell.slow_cm:.3f}｜{path_value}")
        line=p.loc[with_control]
        sens=sensitivity[sensitivity.split.eq(split)].set_index("sampling")
        thin=sens.loc["spaced_slow_starts"]
        all_fast=sens.loc["all"]
        lines.append(f"对同手慢速起点间隔至少 30 帧取样，再看其中慢→快的 {int(thin.n)} 个起点，"
                     f"现有模型 {thin.current_cm:.3f}、专训 GRU {thin.gru_cm:.3f} 厘米。"
                     f"全部慢→快起点按环境分开，专训 GRU 在 {int(all_fast.gru_better_envs)}/{int(all_fast.environments)} 个环境误差较低。")
        lines.append(f"含补充对照后采用的轨迹在未来第 1、3、10、30 帧的环境等权误差依次为 {line.frame1_cm:.3f}、{line.frame3_cm:.3f}、"
                     f"{line.frame10_cm:.3f}、{line.frame30_cm:.3f} 厘米，各自使用该预测步的合格参考。")
        lines.append(f"共有 {int(line.path_n)} 个整段可评分起点，固定网格留 {int(line.spaced_n)} 个非重叠手部窗口。")
        for path in dict.fromkeys([with_control,chosen,"gru","anchored_damped"]):
            t=bands[(bands.split==split)&(bands.method==path)&(bands.sampling=="grid")&(bands.scope=="all")&(bands.env=="all")]
            lines.append(f"{NAMES[path]}　范围方法｜整段实测覆盖｜逐帧实测覆盖｜平均半径｜第 30 帧平均半径")
            for v in t.itertuples():
                lines.append(f"{BANDS[v.band]}｜{v.covered}/{v.n}（{v.path_coverage:.2%}）｜{v.frame_coverage:.2%}｜"
                             f"{v.mean_radius_cm:.2f} 厘米｜{v.endpoint_radius_cm:.2f} 厘米")
        lines.append("")
    lines.extend(["逐帧法在同一批校准窗口内给每个步长分别取 95% 保形分位数。整段法按单个窗口内最大标准化误差膨胀范围。"
                  "固定分位数对照保留相同学习尺度和轨迹级校准，仅关闭在线调整分位水平。"
                  "校准只改变半径，不改变预测中心，因此任何同一模型的点误差在校准前后完全一致。",
                  "原题的观测质量与遮挡可靠性分析仍是第一问的另一项内容。本次比较仅覆盖腕点运动预测和它的误差范围。",
                  "运行 q1_gru_path_calibration.py 可复算，--from-weights 可加载已训练的 3 个第一问 GRU。"
                  "CAFHT 的原论文为 Zhou、Lindemann 与 Sesia，ICML 2024。"])
    (OUT/"实测结论.txt").write_text("\n".join(lines)+"\n",encoding="utf8")


def run(reuse):
    OUT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(3)
    torch.backends.cudnn.benchmark=False
    torch.backends.cudnn.deterministic=True
    dev=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg=json.loads((PALM/"protocol.json").read_text(encoding="utf8"))
    frozen=(Policy(**cfg["prior"]),Rule(**cfg["selector"]),cfg["retrieval"])
    dump("protocol.json",dict(target="Wrist in 3D; 30 forecasts issued together at the origin; no observation requests",
        train="16 supplied train environments only, 3 seeds, 24 epochs",seeds=SEEDS,
        features="Same 2s causal motion channels as the current Q1 wrist model plus the original 10-step dense wrist context, no incumbent endpoint supplied to the new standalone GRU",
        calibration="Train normalized marginal residuals; outer calibration on environment-disjoint validation windows. Leave one validation environment out for reported validation coverage.",
        coverage=.95,calibration_grid="Origin 10,40,... per record and hand; all 30 reference frames must be eligible",
        adapt="One earlier endpoint matures at each 30-frame grid tick; update alpha only when current recorded wrist quality is ok",
        point_rule="Replace current endpoint only if standalone GRU improves validation macro endpoint and slow->fast, and does not worsen slow->slow or all slow origins.",
        supplemental_control="Added after main comparison: original trained damped Kalman, tau=.45 s, both native and matched to the exact frozen incumbent endpoint. No parameters searched.",
        path_rule="For the retained endpoint choose the available path with the lowest mean path error, macro-averaged by validation environment."))
    arm,_=make_arm("wrist",dev)
    paths=Paths(arm.data)
    batches=Batches(arm.data,paths,dev)
    train=paths.meta.split.eq("train").to_numpy()
    train_ids=np.flatnonzero(train & paths.mask[:,-1])
    nets=[train_net(batches,train_ids,seed,reuse) for seed in SEEDS]
    q2_models=[]
    for seed in Q2_SEEDS:
        model=PathGRU()
        model.load_state_dict(torch.load(Q2_OUT/f"path_gru_{seed}.pt",map_location="cpu",weights_only=True))
        q2_models.append(model.eval())
    # This reference distribution is fitted before any validation outcomes are used.
    grid_ids=np.flatnonzero(train & paths.grid & paths.mask.all(1))
    tr_pred,tr_scale=predict(nets,batches,grid_ids)
    reference=np.sort(np.linalg.norm(tr_pred-paths.y[grid_ids],axis=2)/tr_scale,axis=0)
    np.save(OUT/"training_marginal_scores.npy",reference)
    point_tabs,band_tabs,window_logs,endpoint_logs,env_rows=[],[],[],[],[]
    for split in ("validation","test"):
        ids,g,err,scale=candidates(split,arm,paths,batches,nets,q2_models,frozen)
        points=point_metrics(paths,ids,g,err)
        point_tabs.append(points)
        mask=paths.mask[ids,-1]
        keys=["split","env","sample","hand","origin","slow","end_bin"]
        details=g.loc[mask,keys].reset_index(drop=True).copy()
        for name,value in err.items():
            details[name]=value[mask,-1]*100
        endpoint_logs.append(details)
        for env,group in details.groupby("env"):
            for name in err:
                env_rows.append(dict(split=split,env=env,method=name,n=len(group),mean_cm=group[name].mean(),
                                     fast_cm=group.loc[group.slow & group.end_bin.eq(2),name].mean()))
        if split=="validation":
            p=points.set_index("method")
            old,new=p.loc["current"],p.loc["gru"]
            adopt=(new.endpoint_macro_cm<old.endpoint_macro_cm and new.fast_cm<old.fast_cm and
                   new.slow_cm<=old.slow_cm and new.allslow_cm<=old.allslow_cm)
            chosen="gru" if adopt else min(("linear_current","anchored_gru"),key=lambda n:p.at[n,"path_macro_cm"])
            with_control=chosen if adopt else min((chosen,"anchored_damped"),key=lambda n:p.at[n,"path_macro_cm"])
            lock=dict(path=chosen,path_with_physical_control=with_control,adopt_gru_endpoint=bool(adopt),training_origins=len(train_ids),
                      training_grid_n=len(grid_ids),validation_point_scores=p[["endpoint_macro_cm","fast_cm","slow_cm","allslow_cm"]].to_dict("index"),
                      outer={},test_use="previously seen test environments, frozen retrospective comparisons")
        for name in PATHS:
            locked=lock["outer"][name] if split=="test" else None
            bands,logs,params=band_metrics(name,ids,g,err[name],scale,paths,reference,locked)
            band_tabs.append(bands)
            window_logs.append(logs)
            if split=="validation":
                lock["outer"][name]=params
        if split=="validation":
            dump("validation_lock.json",lock)
        print(split,points[["method","endpoint_macro_cm","fast_cm","path_macro_cm"]].round(3).to_string(index=False),flush=True)
    points=pd.concat(point_tabs,ignore_index=True)
    bands=pd.concat(band_tabs,ignore_index=True)
    points.to_csv(OUT/"point_errors.csv",index=False)
    bands.to_csv(OUT/"band_coverage.csv",index=False)
    pd.DataFrame(env_rows).to_csv(OUT/"point_by_environment.csv",index=False)
    pd.concat(window_logs,ignore_index=True).to_csv(OUT/"nonoverlap_windows.csv.gz",index=False,compression="gzip")
    endpoints=pd.concat(endpoint_logs,ignore_index=True)
    endpoints.to_csv(OUT/"endpoint_origins.csv.gz",index=False,compression="gzip")
    sensitivity=slow_fast_sensitivity(endpoints)
    sensitivity.to_csv(OUT/"slow_fast_sensitivity.csv",index=False)
    report(points,bands,lock,sensitivity)
    print("Selected path; including the fixed physical control",lock["path"],lock["path_with_physical_control"],flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--from-weights",action="store_true")
    run(parser.parse_args().from_weights)
