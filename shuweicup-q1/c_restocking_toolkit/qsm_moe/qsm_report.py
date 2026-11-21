"""Verify pairing, audit an independent prefix inference and write the Chinese comparison report."""
import base64
import hashlib
import io
import itertools
import json
import os
import platform
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR",str(Path(tempfile.gettempdir())/"qsm_mpl_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
import torch
from scipy.spatial.transform import Rotation

from qsm_data import ROOT,KIT,OUT,CACHE,SNAP,write_json
from qsm_predict import issue_prefix
from qsm_q1 import KEY,interpolate
from qsm_q2 import KEY as KEY2

OLD1=ROOT/"分析报告"/"第一问GRU轨迹与整段校准"
OLD2=ROOT/"分析报告"/"第二问GRU轨迹与整段校准"
LABEL={"hold":"保持腕点","A0":"A0 质量标记 Kalman","A1":"A1 短窗 GRU 残差","B1":"B1 长短窗残差",
       "B2":"B2 加运动阶段","B3":"B3 加学习式观测质量","B4":"B4 加从头训练的视觉特征","B5":"B5 自适应概率混合",
       "anchored_damped":"已有方案 · 保留终点、衰减轨迹","gru":"已有方案 · 专训轨迹 GRU"}


def file_hash(path):
    h=hashlib.sha256()
    with path.open("rb") as f:
        for part in iter(lambda:f.read(1<<20),b""):
            h.update(part)
    return h.hexdigest()


def read(name):
    return pd.read_csv(OUT/name)


def table(headers,rows):
    frame=pd.DataFrame(rows,columns=headers)
    return frame.to_html(index=False,border=0,justify="left",escape=True)


def percent(x):
    return f"{x:.1%}"


def match_archives():
    issued=read("q1_issued_windows.csv.gz")
    prev=pd.read_csv(OLD1/"nonoverlap_windows.csv.gz").query("method=='anchored_damped'")
    cols=["path_mean_error_cm","max_error_cm","joint_static_covered","joint_adaptive_covered"]
    windows=issued.merge(prev[KEY+cols],on=KEY,how="inner",validate="one_to_one")
    assert len(windows)==len(issued)==len(prev)==1161
    windows.to_csv(OUT/"q1_paired_windows.csv.gz",index=False)
    ep=read("q1_issued_endpoints.csv.gz")
    old_ep=pd.read_csv(OLD1/"endpoint_origins.csv.gz",usecols=KEY+["anchored_damped","gru"])
    endpoints=ep.merge(old_ep,on=KEY,validate="one_to_one")
    assert len(endpoints)==len(ep)==len(old_ep)==36466
    endpoints.to_csv(OUT/"q1_paired_endpoints.csv.gz",index=False)
    old=pd.read_csv(OLD2/"per_window.csv.gz")
    old=old[old.method.isin(["gru_fixed_tuned","gru_paced"])]
    damped=pd.read_csv(OLD2/"schedule_control_windows.csv.gz").query("method=='damped_gru_calendar'")
    q2=pd.concat((read("q2_issued_windows.csv.gz"),old,damped),ignore_index=True)
    values=["error_cm","radius_cm","path_covered","requests","waiting_s","request_frames"]
    q2=q2.pivot(index=KEY2,columns="method",values=values)
    q2.columns=[f"{model}_{col}" for col,model in q2.columns]
    assert len(q2)==576 and not q2.filter(like="error_cm").isna().any().any()
    q2.reset_index().to_csv(OUT/"q2_paired_windows.csv.gz",index=False)
    return windows


def movement_figure(issued,logs):
    test=np.flatnonzero(logs.split.eq("test"))
    ordered=test[np.argsort(logs.B5_path_cm.to_numpy()[test])]
    ix=ordered[np.array([int(.5*(len(ordered)-1)),int(.9*(len(ordered)-1))])]
    paths=interpolate(issued["diffusion_knots"][ix])*100
    fig,axes=plt.subplots(2,2,figsize=(9.0,6.1),constrained_layout=True)
    for row,slot in enumerate(ix):
        for col,(x,y,ylab) in enumerate(((0,1,"Head Y (cm)"),(0,2,"Head Z (cm)"))):
            ax=axes[row,col]
            def trace(curve,**kwargs):
                c=np.vstack((np.zeros(3),curve))
                ax.plot(c[:,x],c[:,y],**kwargs)
            for k,line in enumerate(paths[row]):
                trace(line,color="#9cb9cc",alpha=.66,lw=1,label="Diffusion sample" if k==0 else None)
            trace(100*issued["center"][slot],color="#226586",lw=2.4,label="Issued medoid")
            trace(100*issued["truth"][slot],color="#a34b33",lw=2.4,label="Recorded wrist")
            ax.scatter([0],[0],s=15,c="#202a31",zorder=5)
            v=logs.iloc[slot]
            ax.set_title(f"{v['sample']} / {v.hand} / frame {v.origin}",loc="left",fontsize=9)
            ax.set(xlabel="Head X (cm)",ylabel=ylab,aspect="equal")
            ax.grid(alpha=.2)
            if row==col==0: ax.legend(fontsize=7)
    buffer=io.BytesIO()
    fig.savefig(buffer,format="png",dpi=145)
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def audit(issued,logs):
    ids=issued["canonical_row"]
    meta=pd.read_parquet(CACHE/"meta.parquet")
    ordered=meta.loc[ids,KEY].reset_index(drop=True)
    pd.testing.assert_frame_equal(ordered,logs[KEY],check_dtype=False)
    results=[]
    for sp in ("validation","test"):
        group=logs[logs.split.eq(sp)]
        for pos in (0,len(group)*2//3):
            i=group.index[pos]
            v=logs.loc[i]
            live=issue_prefix(v["sample"],v.hand,int(v.origin))
            frame=pd.read_parquet(KIT/"cache/canonical/frames.parquet",
                      columns=["head_qx","head_qy","head_qz","head_qw"],
                      filters=[("sample","==",v["sample"]),("frame_index","==",int(v.origin))])
            rot=Rotation.from_quat(frame.iloc[0].to_numpy()).as_matrix()
            diff=(np.asarray(live["center_world"])-live["last_wrist_world"])@rot-issued["center"][i]
            np.testing.assert_allclose(diff,0,atol=5e-5)
            np.testing.assert_allclose(live["radius_m"],issued["radius"][i],atol=8e-5)
            np.testing.assert_allclose(live["expert_weights"][1]*8,issued["diffusion_total_weight"][i],atol=1e-4)
            assert live["selected_support"]==int(issued["medoid"][i])
            results.append(dict(sample=v["sample"],hand=v.hand,origin=int(v.origin),
                         max_coordinate_delta_m=float(np.abs(diff).max()),
                         max_radius_delta_m=float(np.abs(live["radius_m"]-issued["radius"][i]).max())))
    return results


def paired_stats(logs):
    # Four held-out environments are the unit of variation. Quantiles describe sensitivity to their weights.
    rows=[]
    endpoint=read("q1_paired_endpoints.csv.gz")
    q2=read("q2_issued_windows.csv.gz")
    old2=pd.read_csv(OLD2/"per_window.csv.gz").query("method=='gru_fixed_tuned'")
    for sp in ("validation","test"):
        g=logs[logs.split.eq(sp)]
        e=endpoint[endpoint.split.eq(sp)]
        q=q2[q2.split.eq(sp)].pivot(index=KEY2,columns="method",values="error_cm").reset_index()
        q=q.merge(old2[old2.split.eq(sp)][KEY2+["error_cm"]],on=KEY2,validate="one_to_one")
        samples=[("B5 minus incumbent, endpoint",e.B5-e.anchored_damped,e.env),
                 ("B3 minus incumbent, endpoint",e.B3-e.anchored_damped,e.env),
                 ("B5 minus incumbent, spaced path",g.B5_path_cm-g.path_mean_error_cm,g.env),
                 ("new active minus new fixed",q.new_active-q.new_fixed,q.env),
                 ("new active minus old fixed GRU",q.new_active-q.error_cm,q.env)]
        for label,delta,env in samples:
            by=delta.groupby(env).mean()
            res=np.array([np.mean(v) for v in itertools.product(by.to_numpy(),repeat=len(by))])
            rows.append(dict(split=sp,contrast=label,macro_delta_cm=by.mean(),environments=len(by),
                   envs_improved=int((by<0).sum()),min_env_delta_cm=by.min(),max_env_delta_cm=by.max(),
                   env_resampling_p025_cm=np.quantile(res,.025),env_resampling_p975_cm=np.quantile(res,.975)))
    result=pd.DataFrame(rows)
    result.to_csv(OUT/"paired_environment_sensitivity.csv",index=False)
    return result


def report():
    logs=match_archives()
    p=read("q1_points.csv")
    oldp=pd.read_csv(OLD1/"point_errors.csv")
    allp=pd.concat([oldp[oldp.method.isin(["anchored_damped","gru"])],p],ignore_index=True)
    q2=read("q2_compare.csv")
    oldq=pd.read_csv(OLD2/"summary.csv").rename(columns={"path_covered":"path_coverage","windows":"n"})
    oldq=oldq[oldq.method.isin(["gru_fixed_tuned","gru_paced"])]
    oldphys=pd.read_csv(OLD2/"schedule_controls.csv").query("method=='damped_gru_calendar'").rename(columns={"path_covered":"path_coverage"})
    qs=pd.concat([oldq,oldphys,q2],ignore_index=True)
    oldbands=pd.read_csv(OLD1/"band_coverage.csv").query("sampling=='grid' and scope=='all' and env=='all' and method=='anchored_damped'")
    bands=read("q1_bands.csv")
    energy=read("q1_energy.csv")
    phase=read("q1_phase.csv"); quality=read("q1_quality.csv")
    cadence=read("record_cadence.csv")
    q1cfg=json.loads((CACHE/"ready.json").read_text(encoding="utf8"))
    q1lock=json.loads((OUT/"q1_locked.json").read_text(encoding="utf8"))
    q2lock=json.loads((OUT/"q2_locked.json").read_text(encoding="utf8"))
    natural=read("q2_natural_flag_runs.csv")
    comparison=paired_stats(logs)
    with np.load(OUT/"q1_issued_trajectories.npz") as pack:
        prefix=audit(pack,logs)
        fig=movement_figure(pack,logs)
    common=[]
    for sp,g in logs.groupby("split"):
        for cm in (10,20,30):
            common.append(dict(split=sp,radius_cm=cm,n=len(g),QSM_coverage=(g.B5_max_error_cm<=cm).mean(),
                               incumbent_coverage=(g.max_error_cm<=cm).mean()))
    pd.DataFrame(common).to_csv(OUT/"q1_equal_radius.csv",index=False)
    plotsrc="data:image/png;base64,"+fig
    tests=allp[allp.split.eq("test")].set_index("method")
    vals=allp[allp.split.eq("validation")].set_index("method")
    points=table(["方法","验证终点 cm","验证整段 cm","测试终点 cm","测试整段 cm"],
                [[LABEL[k],f"{vals.at[k,'endpoint_macro_cm']:.2f}",f"{vals.at[k,'path_macro_cm']:.2f}",
                 f"{tests.at[k,'endpoint_macro_cm']:.2f}",f"{tests.at[k,'path_macro_cm']:.2f}"]
                 for k in ("anchored_damped","gru","hold","A0","A1","B1","B2","B3","B4","B5")])
    b_rows=[]
    for k,label in (("joint_static","已有方案 · 固定整段校准"),("joint_adaptive","已有方案 · 自适应整段校准"),
                    ("pooled","QSM · 全局整段校准"),("stage","QSM · 按阶段校准"),("adaptive","QSM · 按返回误差与阶段更新")):
        base=oldbands if k.startswith("joint_") else bands[bands.group.eq("all")]
        tmp=base[base.band.eq(k)].set_index("split")
        radius="mean_radius_cm" if k.startswith("joint_") else "radius_cm"
        b_rows.append([label,*[v for sp in ("validation","test") for v in (percent(tmp.at[sp,"path_coverage"]),f"{tmp.at[sp,radius]:.2f}")]])
    bands_table=table(["轨迹范围","验证覆盖","验证半径 cm","测试覆盖","测试半径 cm"],b_rows)
    names={"dynamics_only":"动力学单点","diffusion_only":"八条扩散轨迹","fixed_25pct":"固定 25% 生成权重",
           "fitted_constant":"训练集拟合单一权重","motion_gate":"运动门控","quality_stage_gate":"质量与阶段门控"}
    e=energy.pivot(index="method",columns="split",values="energy_score_cm")
    es_table=table(["同一组专家的输出分布","验证能量分数 cm","测试能量分数 cm"],
                   [[v,f"{e.at[k,'validation']:.3f}",f"{e.at[k,'test']:.3f}"] for k,v in names.items()])
    qnames={"damped_gru_calendar":"已有衰减模型 · 第 11、19 帧","gru_fixed_tuned":"已有 GRU · 第 11、19 帧",
            "gru_paced":"已有 GRU · 分段触发","no_request":"QSM · 无请求","one_request":"QSM · 第 11 帧",
            "new_fixed":"QSM · 验证选定固定时刻","new_active":"QSM · 按预计误差收益请求"}
    q_rows=[]
    for k,name in qnames.items():
        tmp=qs[qs.method.eq(k)].set_index("split")
        rad=tmp.at["test","radius_cm"]
        q_rows.append([name,f"{tmp.at['validation','env_macro_cm']:.3f}",f"{tmp.at['test','env_macro_cm']:.3f}",
                       "未归档" if pd.isna(rad) else percent(tmp.at["test","path_coverage"]),
                       "未归档" if pd.isna(rad) else f"{rad:.2f}",f"{tmp.at['test','requests']:.0f}",f"{tmp.at['test','waiting_s']:.3f}"])
    q_table=table(["第二问方法","验证误差 cm","测试误差 cm","测试整窗覆盖","测试半径 cm","请求/窗","累计等待 s/窗"],q_rows)
    for_df=tests.loc["B5"]; prev=tests.loc["anchored_damped"]
    qtest=qs[qs.split.eq("test")].set_index("method")
    qt=quality[quality.split.eq("test")].iloc[0]; pt=phase[phase.split.eq("test")].iloc[0]
    test_logs=logs[logs.split.eq("test")]
    g=read("q1_groups.csv")
    slow=[]
    for name in ("anchored_damped","B2","B3","B4","B5"):
        if name=="anchored_damped":
            f,s=oldp.query("method=='anchored_damped' and split=='test'")[["fast_cm","slow_cm"]].iloc[0]
        else:
            v=g[(g.method==name)&g.split.eq("test")].set_index("group")
            f,s=v.at["slow_to_fast","pooled_endpoint_cm"],v.at["slow_to_slow","pooled_endpoint_cm"]
        slow.append([LABEL[name],f"{s:.2f}",f"{f:.2f}"])
    groups_table=table(["回顾测试起点分层","慢→慢终点 cm","慢→快终点 cm"],slow)
    sens=comparison[comparison.split.eq("test")].set_index("contrast")
    sensitivity=sens.loc["B5 minus incumbent, spaced path"]
    curve=read("q2_horizons.csv").query("split=='test' and method=='no_request'").sort_values("frame")
    lost=curve.loc[curve.within_10cm<.9,"frame"]
    ok_until=(int(lost.iloc[0])-1) if len(lost) else 30
    qnow=natural.query("split=='test' and quality=='occlusion'")
    protocol=json.loads((OUT/"q2_protocol.json").read_text(encoding="utf8"))
    q2log=read("q2_issued_windows.csv.gz").query("split=='test' and method=='new_active'")
    pair=read("q1_paired_endpoints.csv.gz")
    assert len(pair)==36466 and len(test_logs)==760 and len(q2log)==376
    source_dir=Path(__file__).resolve().parent
    files=[*sorted(source_dir.glob("qsm_*.py")),*sorted(OUT.glob("*.pt")),*sorted(OUT.glob("*.joblib")),
           OUT/"feature_normalization.npz",OUT/"q1_sampler_noise.npy",OUT/"q1_feedback_reference.npy",
           SNAP/"03-LSTTN-68185aa8.zip",SNAP/"05-Uni-Hand-7f3edd59.zip",KIT/"cache/canonical/frames.parquet",
           ROOT/"Restocking_Problem_and_Data_EN/Restocking_Tasks_and_Constraints/record_split.csv"]
    write_json(OUT/"verification.json",dict(protocol_version=q1cfg["version"],python=platform.python_version(),numpy=np.__version__,
           torch=torch.__version__,sklearn=sklearn.__version__,device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
           q1_prefix_check=prefix,q1_endpoint_pairs=len(pair),q1_window_pairs=len(logs),q2_window_pairs=576,
           preexisting_parameters_used_for_training=0,source_and_parameter_sha256={str(v.relative_to(ROOT)):file_hash(v) for v in files}))
    references=table(["论文或作者快照","代码与本题的对应关系"],[
        ["GNSS 观测质量建模","借鉴可靠性预测。独立训练树模型，用腕点几何、运动与历史标记估计当前异常概率，当前标记只用于滤波和反馈。"],
        ["LSTTN · 68185aa8","直接调用作者的 TransformerLayers。输入为同手最多 61 帧的过去轨迹，每 5 帧取一个长窗 token，配合最近 9 帧的时序卷积。"],
        ["EgoMAN","借鉴阶段概率。三类运动代理由位移速度与转角定义，利用训练记录未来第 10 帧的代理标签训练，在线只使用历史。"],
        ["Uni-Hand · 7f3edd59","直接调用作者的 GaussianDiffusion 反推干净样本公式。新建腕点专用 GRU、双帧视觉编码器与去噪器，在本题上随机初始化训练。"],
        ["CAFHT、CPTC","借鉴整条路径的最大误差、按阶段归集反馈。每次反馈只读取已到达且质量合格标记可用的端点，按三维球半径校准。"],
        ["LASER","借鉴按预期信息收益选择观测的思想。针对两次请求和三帧延迟，在训练窗口回放不同请求日历，拟合相对误差成本。"]])
    css="body{font:16px/1.65 'Microsoft YaHei','Segoe UI',sans-serif;color:#182a35;background:#fff;margin:0}main{max-width:1040px;margin:40px auto;padding:0 30px}h1{font-size:32px;line-height:1.25}h2{margin-top:34px;font-size:23px}p{margin:10px 0}table{border-collapse:collapse;width:100%;font-size:14px;margin:12px 0 18px;font-variant-numeric:tabular-nums}td,th{padding:8px 10px;border-bottom:1px solid #d3dce0;text-align:right}td:first-child,th:first-child{text-align:left}th{background:#e9f1f3}a{color:#17587b}img{width:100%;height:auto}code{background:#edf1f4;padding:2px 4px;word-break:break-all}.caption{color:#475967;font-size:14px}@media print{main{margin:0;max-width:none}h2{break-after:avoid}tr,img{break-inside:avoid}}"
    html=f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>C题 · QSM-MoE 独立建模实测</title><style>{css}</style><main>
<h1>C 题 QSM-MoE 独立建模与配对实测</h1>
<p>第一问以左右手各自的第 0 号关键点为腕点，预测在同一起点发出的 30 帧三维轨迹。第二问沿用同一腕点定义，在观测缺失时决定何时请求。主线融合后回顾测试整段误差为 <b>{for_df.path_macro_cm:.2f} cm</b>，已有方案为 <b>{prev.path_macro_cm:.2f} cm</b>。第二问在两次请求下，新主动策略为 <b>{qtest.at['new_active','env_macro_cm']:.3f} cm</b>，同一新模型固定请求为 <b>{qtest.at['new_fixed','env_macro_cm']:.3f} cm</b>。</p>
<h2>数据、时间与拟合边界</h2>
<p>使用 EgoDemo、EgoPro、EgoStandard 的 69 条记录和 24 个环境。训练、验证、测试分别有 55、7、7 条记录，对应 16、4、4 个互斥环境。每条记录按自身坐标系处理，位移用预测起点的头部坐标轴表示。j00 是每只手都有的一致解剖标志，第二问遮挡前后可以按同一坐标评估；其位置表示手腕。合格参考沿用公共评分的质量标记及骨长筛选。训练与评测使用同一原始来源，参考位置含测量误差。</p>
<p>66 条记录的中位采样率约 30 Hz，另 3 条约 25 Hz，三条都在训练集。各记录未来 30 帧的中位时长介于 {cadence.h30_median_s.min():.3f} 和 {cadence.h30_median_s.max():.3f} 秒。预测时用截至当前的最近五次实际时间间隔估计步长。为与归档结果配对，主表统一以第 30 帧为终点。</p>
<p>第一问把官方训练环境分为 10 个专家拟合环境、3 个门控环境、3 个整段校准环境。第二问独立划为 {len(protocol['environments']['fit'])} 个拟合、{len(protocol['environments']['policy'])} 个策略学习和 {len(protocol['environments']['calibration'])} 个外层校准环境。所有神经网络随机初始化，滤波参数和阈值在本实验拟合。旧方案只提供评分参考、相同窗口及归档成绩。两方案内部训练资源分配不同，本次结果比较各自完整流程。测试环境在项目历史中已使用，本报告按回顾测试解释。</p>
<h2>第一问　在线发出一条轨迹时的误差</h2>
<p>每个环境先算平均误差，四个环境等权。终点使用验证 12,817、测试 23,649 个合格起点；整段平均误差使用验证 11,861、测试 22,704 个连续 30 帧全部合格的起点。A0 用常速度状态与衰减外推，A1 使用自建短窗 GRU 学习残差。B1 同时读取长短历史，B2 引入运动阶段，B3 在原始质量标记基础上加入学习式质量估计，B4 加入可见视频特征。B5 保留动力学分支与八条生成轨迹，在线点输出取在加权分布内期望路径距离最小的一条。</p>{points}
<p>按相同非重叠窗口配对，新完整模型相对已有方案的整段环境等权差值为 {sensitivity.macro_delta_cm:+.2f} cm，四个测试环境中有 {int(sensitivity.envs_improved)} 个误差下降，环境差值范围 {sensitivity.min_env_delta_cm:+.2f} 至 {sensitivity.max_env_delta_cm:+.2f} cm。慢速起点的事后分层如下，每列按起点数汇总。</p>{groups_table}
<h2>观测质量、运动阶段与概率预测</h2>
<p>学习式质量分数以当前非正常标记为代理监督，分类器输入不含当前同手标记。回顾测试 24,482 个手帧中有 {int(qt.flagged)} 个标记异常，频率 {percent(qt.prevalence)}，平均精确率 AP 为 {qt.AP:.3f}，Brier 分数为 {qt.Brier:.4f}。对可见通道按风险调整 Kalman 测量噪声，对缺失通道跳过更新。估计对象是已记录标记的风险。合格评分网格中只有 {int(test_logs.recent_flag.sum())} 个测试窗口的起点在近期出现过异常，复杂遮挡条件下的外推误差仍缺少充分样本。</p>
<p>当前运动由五帧位移速度及前后两个五帧位移向量的夹角划分为停留、转向或移动。B4 对未来十帧运动模式的判对率为 {percent(pt.phase_accuracy)}，沿用当前模式为 {percent(pt.persistence_accuracy)}。回顾测试时模型把 {int((test_logs.stage==1).sum())} 个窗口的最高概率分配给转向，参考运动规则标示了 {int((test_logs.actual_mode_at_10==1).sum())} 个转向窗口。阶段输出用于描述运动，接触、抓取成功等任务语义仍需要专门标注。</p>
<p>对动力学轨迹与八条扩散样本组成的联合分布，按完整 90 维轨迹计算能量分数，再除以 √30 并按环境等权。分数越低越好。验证与测试分别有 401 和 760 个非重叠窗口。固定 25% 一项将四分之一概率给生成专家。训练集拟合的常数为 {q1lock['gate']['constant']:.1%}。每个窗口的九条支持轨迹及权重均可恢复。</p>{es_table}
<p>下图选取新完整模型测试窗口中按误差排序的第 50 和第 90 百分位。左列与右列为同一动作在起点头部坐标系的两个投影。蓝线是当时可实际发出的轨迹，淡线是条件扩散样本，红线是随后记录到的位置。</p><img src="{plotsrc}" alt="两条实测腕点轨迹与八个候选轨迹的头部坐标投影">
<h2>第一问　整条轨迹的误差范围</h2>
<p>目标为 95% 整段覆盖。每步用三维欧氏球，30 步全部包含参考腕点即记一次覆盖。覆盖率与半径均按窗口汇总。外层保留 {q1lock['calibration']['n']} 个独立于专家和门控训练环境的校准窗口；同一环境内的窗口仍有关联。按阶段校准时，小于 25 窗的类回退全局分位。反馈版只在先前发出的终点到达时更新该运动模式的误差水平。</p>{bands_table}
<p>归档方案与 QSM 各用自身的校准协议。另附统一固定半径下两套中心的覆盖，便于排除半径取值差异。<a href="q1_equal_radius.csv">等半径覆盖表</a></p>
<h2>第二问　延迟请求与相同成本回放</h2>
<p>每窗先观察腕点 12 帧，然后隐藏 30 帧。请求采样后三帧返回，单次花费 1 单位。新稀疏模型用实际时间间隔、已释放腕点及速度估计运动专家的权重。主动策略在训练环境中比较 16 对候选时刻的未来误差，第一次请求由初始可见历史决定，第一次观测返回后再根据新坐标和当时已发出的预测差异选择第二次时刻。固定方案在验证集选定第 {q2lock['fixed']['first']}、{q2lock['fixed']['second']} 帧。回放用相同的 200 个验证窗口和 376 个测试窗口。等待列为每窗各请求延迟实际秒数之和。</p>{q_table}
<p>以 10 cm 为分析容差，回顾测试无请求时，至少 90% 窗口的单步误差都在容差内的连续前缀为 {ok_until} 帧。两次主动请求被分配到 {q2log.request_frames.nunique()} 种时刻组合。自然遮挡另行按原标记统计，测试集中有 {len(qnow)} 段遮挡，中位 {qnow.frames.median():.0f} 帧，最长 {int(qnow.frames.max())} 帧。可观测恢复的区段，从最后一个遮挡帧到下一个合格参考帧的中位时长为 {qnow.recovery_s.median():.3f} 秒。自然遮挡内部没有独立位置真值，误差与覆盖率来自人工遮挡回放。</p>
<h2>论文、作者代码与复算</h2>{references}
<p>全部实验在项目的独立 qsm_moe 目录中运行。QSM 专用参数文件由本次训练生成。归档结果只在核对和制表时读取。运行 <code>python qsm_run.py</code> 会从数据准备、随机初始化训练一直执行到配对验证与本报告。运行 <code>python qsm_predict.py {prefix[0]['sample']} {prefix[0]['hand']} {prefix[0]['origin']}</code> 可仅从指定前缀发出世界坐标下 30 步中心、九条备选轨迹、权重与半径。使用本机含 PyTorch、SciPy、pandas、scikit-learn、PyArrow 和 matplotlib 的 cws 环境。</p>
<p>已用四个独立截断前缀复算第一问输出。第二问通过未请求坐标扰动和返回前扰动两项核验。逐环境差值、逐起点误差和逐窗口动作均已保存。<a href="verification.json">核验与版本</a> · <a href="paired_environment_sensitivity.csv">逐环境差值汇总</a> · <a href="q1_paired_endpoints.csv.gz">逐起点误差</a> · <a href="q2_paired_windows.csv.gz">逐窗口请求</a> · <a href="../../参考文献/论文与代码仓库/README.md">论文原文与作者仓库索引</a> · <a href="../../c_restocking_toolkit/qsm_moe/qsm_run.py">复算入口</a></p></main></html>"""
    (OUT/"QSM-MoE实测对比.html").write_text(html,encoding="utf8")
    print("Report and prefix inference verified.",flush=True)


if __name__=="__main__":
    report()
