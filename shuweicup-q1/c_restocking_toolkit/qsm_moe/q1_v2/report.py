"""Generate the Chinese Q1 diagnosis from the paired tables and verify the 99 proxy labels from positions."""
import base64
import io
import json
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR",os.path.join(tempfile.gettempdir(),"qsm_mpl_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

from settings import OUT,WORK,PREVIOUS,KIT,KEY,MODES


def read(name):
    return pd.read_csv(OUT/name)


def table(headers,values):
    return pd.DataFrame(values,columns=headers).to_html(index=False,border=0,escape=True,justify="left")


def label_audit(logs):
    test=logs[logs.split.eq("test")].copy().reset_index(drop=True)
    meta=pd.read_parquet(WORK/"cache"/"meta.parquet")
    meta=meta.assign(row=np.arange(len(meta)))
    test=test.merge(meta[KEY+["row"]],on=KEY,validate="one_to_one")
    truth=np.load(WORK/"cache"/"y.npy",mmap_mode="r")[test.row.to_numpy()]
    one=truth[:,4]
    two=truth[:,9]-truth[:,4]
    cosine=(one*two).sum(1)/np.maximum(np.linalg.norm(one,axis=1)*np.linalg.norm(two,axis=1),1e-9)
    angle=np.degrees(np.arccos(np.clip(cosine,-1,1)))
    delta=np.empty(len(test))
    for sample,g in test.groupby("sample"):
        t=pd.read_parquet(KIT/"cache/canonical/frames.parquet",columns=["t_s","frame_index"],
                         filters=[("sample","==",sample)]).sort_values("frame_index").t_s.to_numpy()
        o=g.origin.to_numpy()
        delta[g.index]=t[o+10]-t[o+5]
    speed=np.linalg.norm(two,axis=1)/delta
    with np.load(WORK/"feature_normalization.npz") as f:
        cut=float(f["mode_cut"])
    derived=np.where(speed<cut,0,np.where(angle>45,1,2))
    np.testing.assert_array_equal(derived,test.target_phase)
    test["angle_deg"]=angle
    test["last_5_speed_mps"]=speed
    test["first_5_displacement_mm"]=np.linalg.norm(one,axis=1)*1000
    test["second_5_displacement_mm"]=np.linalg.norm(two,axis=1)*1000
    test["speed_cut_mps"]=cut
    columns=KEY+["current_phase","target_phase","v1_turn_prob","v2_turn_prob","v1_turn","v2_turn","v2_calibrated_argmax",
                 "angle_deg","last_5_speed_mps","first_5_displacement_mm","second_5_displacement_mm","speed_cut_mps",
                 "v2_fde_cm","v2_path_cm","v1_path_cm","original_path_cm"]
    cases=test.loc[test.turn,columns].copy()
    labels={0:"停留",1:"转向",2:"移动"}
    for field in ("current_phase","target_phase","v2_calibrated_argmax"):
        cases[field]=cases[field].map(labels)
    for field in ("v1_turn","v2_turn"):
        cases[field]=cases[field].map({True:"命中",False:"漏报"})
    cases.hand=cases.hand.map({"left":"左手","right":"右手"})
    cases=cases.rename(columns=dict(split="划分",env="环境",sample="记录",hand="手",origin="起点帧_从0计数",
        current_phase="起点运动阶段",target_phase="实际第10帧阶段",v1_turn_prob="当前版转向概率",
        v2_turn_prob="改进版校准转向概率",v1_turn="当前版判定",v2_turn="改进版预警",
        v2_calibrated_argmax="改进版校准概率最高类别",angle_deg="两段位移夹角_度",
        last_5_speed_mps="后5帧速度_米每秒",first_5_displacement_mm="前5帧位移_毫米",
        second_5_displacement_mm="后5帧位移_毫米",speed_cut_mps="速度门槛_米每秒",v2_fde_cm="改进版终点误差_厘米",
        v2_path_cm="改进版整段误差_厘米",v1_path_cm="当前版整段误差_厘米",original_path_cm="原方案整段误差_厘米"))
    cases.to_csv(OUT/"99个转向窗口.csv",index=False,encoding="utf-8-sig")
    return test


def figure(g,v):
    fig,axes=plt.subplots(1,2,figsize=(9.4,3.55),constrained_layout=True)
    ax=axes[0]
    for col,label,color in (("v1_turn_prob","QSM v1","#9d583a"),("v2_turn_prob","QSM v2","#1a6881")):
        p,r,_=precision_recall_curve(g.turn,g[col])
        ax.step(r,p,where="post",color=color,lw=1.6,label=label)
    ax.axhline(g.turn.mean(),ls="--",lw=1,color="#767676",label="Prevalence")
    chosen=g[g.v2_turn]
    ax.scatter([chosen.turn.sum()/g.turn.sum()],[chosen.turn.mean()],color="#143c53",s=45,label="Locked v2 alert",zorder=4)
    ax.set(xlabel="Recall",ylabel="Precision",xlim=(0,1),ylim=(0,1.02),title="Ten-frame look-ahead turns")
    ax.grid(alpha=.18);ax.legend(fontsize=8)
    ax=axes[1]
    names=("fit","gate","validation","test")
    for k,col,shift,name in (("aligned","#ad674a",-.18,"Full image"),("zero","#35788c",.18,"Visual channel zero")):
        sub=v[(v.train_input==k)&(v.issue_input==k)].groupby("role").path_cm.mean()
        ax.bar(np.arange(4)+shift,sub.loc[list(names)],width=.34,color=col,label=name)
    ax.set_xticks(range(4),["Expert fit","Gate fit","Validation","Test"])
    ax.set(ylabel="Mean path error (cm)",title="Same architecture and training batches")
    ax.grid(axis="y",alpha=.18);ax.legend(fontsize=8)
    buf=io.BytesIO()
    fig.savefig(buf,format="png",dpi=150)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def report():
    m=read("three_way_metrics.csv").set_index(["split","method"])
    phases=read("turn_metrics.csv")
    phase=phases[phases.split.eq("test")].set_index("method")
    labels={"original":"原方案 · 原终点与衰减轨迹","v1":"当前版 QSM-MoE","v2":"改进版 QSM-MoE"}
    metric_rows=[]
    for name,label in labels.items():
        a,b=m.loc[("validation",name)],m.loc[("test",name)]
        es="未归档" if pd.isna(b.energy_cm) else f"{b.energy_cm:.3f}"
        metric_rows.append([label,f"{a.path_macro_cm:.3f}",f"{a.endpoint_macro_cm:.3f}",f"{b.path_macro_cm:.3f}",
                          f"{b.endpoint_macro_cm:.3f}",es,f"{b.coverage:.2%}",f"{b.radius_cm:.2f}"])
    summary=table(["方案","验证整段 cm","验证终点 cm","测试整段 cm","测试终点 cm","测试能量分数 cm","测试整段覆盖","平均半径 cm"],metric_rows)
    plabels={"v1_argmax":"当前版 · 概率最大类","v1_threshold_only":"当前版 · 仅换预警规则","persistence":"沿用当前运动类别",
             "v2_raw_argmax":"改进版 · 类别加权训练原始分数最大类","v2_calibrated_argmax":"改进版 · 校准概率最大类","v2_decision":"改进版 · 冻结的转向预警"}
    pt=[]
    for name,label in plabels.items():
        r=phase.loc[name]
        pt.append([label,int(r.tp),int(r.fp),int(r.fn),f"{r.precision:.1%}",f"{r.recall:.1%}",f"{r.turn_f1:.3f}",f"{r.AP:.3f}"])
    phase_table=table(["规则","命中","误报","漏报","精确率","召回率","转向 F1","平均精确率 AP"],pt)
    logs=read("paired_windows.csv.gz")
    test=label_audit(logs)
    turns=test[test.turn]
    original=read("v1_stage_diagnosis.csv").query("role=='test' and model=='B4'").iloc[0]
    counts=json.loads((OUT/"v1_training_counts.json").read_text(encoding="utf8"))
    v=read("visual_interventions.csv")
    c=read("turn_loss_interventions.csv")
    two=c[c.role.eq("test")].groupby(["balanced","detached"])[["tp","fp","turn_f1"]].mean()
    loss_table=table(["阶段损失按类别重加权","阻断轨迹误差对阶段门控的直接梯度","命中数 · 两种子均值","误报数 · 两种子均值","转向 F1"],
        [["是" if bal else "否","是" if det else "否",f"{r.tp:.1f}",f"{r.fp:.1f}",f"{r.turn_f1:.3f}"]
         for (bal,det),r in two.iterrows()])
    vv=v.groupby(["train_input","issue_input","role"]).path_cm.mean()
    vr=[]
    for k,j,title in (("aligned","aligned","对齐全图训练与预测"),("aligned","one_second_lag","同一模型 · 图像特征滞后约一秒"),
                  ("aligned","first_second","同一模型 · 冻结起始一秒的图像特征"),("aligned","zero","同一模型 · 预测时视觉置零"),
                  ("zero","zero","架构与批次相同 · 全程视觉置零")):
        vr.append([title,*[f"{vv.loc[(k,j,s)]:.3f}" for s in ("fit","gate","validation","test")]])
    visual_table=table(["视觉对照 · 环境等权整段误差","专家训练 cm","门控训练 cm","验证 cm","测试 cm"],vr)
    diag=read("v1_visual_environment_probe.csv")
    dpi=diag[diag.other_record_in_env]
    local=read("local_visual_ablation.csv").set_index(["split","alpha"])
    raw=read("v1_vision_by_environment.csv").query("env=='ALL'").set_index(["role","model"])
    camera=read("camera_alignment.csv")
    by=read("paired_by_environment.csv").query("split=='test'")
    env_table=table(["测试环境","窗数","实际转向","预警命中","预警误报","原方案整段 cm","当前版整段 cm","改进版整段 cm"],
         [[r.env,r.n,r.turns,r.turn_tp,r.turn_fp,f"{r.original_spaced_ade:.2f}",f"{r.v1_spaced_ade:.2f}",f"{r.v2_spaced_ade:.2f}"]
          for r in by.itertuples()])
    lock=json.loads((OUT/"frozen_before_test.json").read_text(encoding="utf8"))
    ver=json.loads((OUT/"verification.json").read_text(encoding="utf8"))
    val_stage=phases.query("split=='validation' and method=='v2_decision'").iloc[0]
    onset=turns[turns.current_phase.ne(1)]
    visually=read("v1_vision_by_environment.csv").query("role=='test' and env!='ALL'").pivot(index="env",columns="model",values="path_cm")
    visual_gap=visually.B4-visually.B3
    mistaken=test[~test.turn & test.v2_turn].target_phase.value_counts()
    tt=phase.loc["v2_decision"]
    v2=m.loc[("test","v2")];v1=m.loc[("test","v1")];base=m.loc[("test","original")]
    gate_text=(f"训练集拟合的常数 {lock['gate']['constant']:.1%}" if lock["distribution"]=="constant"
               else "依赖阶段、质量与运动状态的自适应权重")
    readout_label={"motion":"动力学中心","mean":"加权均值","medoid":"支持轨迹中的加权中心轨迹",
                   "geometric_median":"逐帧加权几何中位数"}[lock["point"]]
    coverage_turn=read("v2_subgroups.csv").query("split=='test' and group=='turn'").iloc[0]
    example=ver["prefixes"][0]
    # The local visual intervention is a sensitivity of the motion center before generative readout.
    png=figure(test,v)
    css="body{font:16px/1.7 'Microsoft YaHei','Segoe UI',sans-serif;color:#172b37;margin:0}main{max-width:1110px;margin:35px auto;padding:0 32px}h1{font-size:32px;line-height:1.3}h2{font-size:22px;margin-top:32px}p{margin:10px 0}table{border-collapse:collapse;width:100%;margin:12px 0 19px;font-size:13.5px;font-variant-numeric:tabular-nums}td,th{padding:8px 9px;text-align:right;border-bottom:1px solid #d7e0e4}th{background:#e8f0f2}td:first-child,th:first-child{text-align:left}figure{margin:12px 0}img{max-width:100%}figcaption{font-size:14px;color:#425560}a{color:#155d77}code{padding:2px 4px;background:#edf2f3}@media(max-width:800px){main{padding:0 12px}section{overflow-x:auto}}@media print{main{margin:0;padding:0}h2{break-after:avoid}tr,figure{break-inside:avoid}}"
    out=f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第一问 · 转向预警、视觉消融与三方对比</title><style>{css}</style><main>
<h1>第一问　转向预警、视觉消融与三方对比</h1>
<p>改进版在同一批回顾测试起点上把 30 帧平均腕点位置误差从 <b>{v1.path_macro_cm:.2f} cm</b> 降到 <b>{v2.path_macro_cm:.2f} cm</b>，原方案为 <b>{base.path_macro_cm:.2f} cm</b>。对提前十帧的转向预警，99 个实际转向中命中 {int(tt.tp)} 个，误报 {int(tt.fp)} 个。单独比较三类校准概率的最大类，当前版命中 0 个，改进版命中 {int(phase.at['v2_calibrated_argmax','tp'])} 个。</p>
<h2>同一腕点、起点和环境的配对结果</h2><section>{summary}</section>
<p>整段误差为整条预测路径上的平均三维欧氏距离，终点误差取第 30 帧。先在各环境内算平均，再对四个环境等权。验证与测试的合格终点分别为 12,817 和 23,649 个，完整路径分别为 11,861 和 22,704 个。区间指标使用 401 和 760 个非重叠窗口，30 帧全部被球形范围覆盖才记为命中。原方案和当前版使用各自归档的自适应范围。能量分数评价整个轨迹分布，越低越好。</p>
<p>相对原方案，改进版测试终点误差只相差 {abs(base.endpoint_macro_cm-v2.endpoint_macro_cm):.3f} cm。与当前版相比，改进版在 {int((by.v2_spaced_ade<by.v1_spaced_ade).sum())}/4 个环境的非重叠整段误差下降；与原方案相比为 {int((by.v2_spaced_ade<by.original_spaced_ade).sum())}/4。改进版测试平均半径更窄，整段覆盖率 {v2.coverage:.2%} 低于预设的 95%，在转向子组为 {coverage_turn.coverage:.2%}。四个测试环境均在项目历史实验中出现过，本表为回顾测试。</p>
<h2>为什么原来的 99 个转向没有一个成为最高概率类</h2>
<p>标签定义为起点后第 10 帧的运动类别。利用第 0→5 帧和第 5→10 帧两段位移求夹角，后一段速度达到训练阈值 {turns.speed_cut_mps.iat[0]:.3f} m/s 且夹角大于 45° 才判转向。已从记录位置和实际时间戳逐窗重算，760 个测试标签全部吻合。99 个转向中，起点已在转向的有 {int((turns.current_phase==1).sum())} 个，其余 {len(onset)} 个在起点属于停留或移动。目标是预判腕点运动方向，标签本身不表示抓取、接触或操作成功。</p>
<p>当前版专家训练中有 {counts['turn']:,} 个转向，占 {counts['turn']/sum(counts.values()):.1%}。对测试真转向，模型给转向的平均概率为 {original.mean_true:.3f}，其余窗口为 {original.mean_other:.3f}。真转向的概率第 90 百分位为 {original.p90_true:.3f}，最大值为 {original.max_true:.3f}。当前版转向 AP 为 {original.ap:.3f}，高于测试发生率 {original.prevalence:.3f}，说明概率排序保留了部分信息，三类直接取最大时仍全部落入另外两类。</p>
<p>原架构让同一组三类分数同时参与阶段分类和运动专家加权，阶段使用普通交叉熵。保持网络结构、训练批次、训练历程和权重初始化配对，两种随机种子下分别改变类别损失和门控梯度。按训练频率的倒数重加权明显改变了转向检出，仅阻断门控梯度的改变很小。此处均按三类分数直接取最大，相同种子对应的逐项结果一并保存。</p><section>{loss_table}</section>
<p>改进版用独立监督头预测运动阶段，输入包括多尺度方向变化、加减速、双手与头部相对运动及近期转向史。概率在三个训练环境上重新校准。预警阈值候选由训练门控环境的分数确定，验证集在精确率至少 25%、报警占比不超过 35% 的候选中按环境平均转向 F1 选择。最终阈值为 {lock['phase']['rule'].get('threshold',0):.3f}，验证命中 {int(val_stage.tp)}/{int(val_stage.turns)}、误报 {int(val_stage.fp)}。发生率、最高概率类别与是否发出预警分别保留。原方案的归档只提供轨迹和误差，转向分类栏仅比较有阶段输出的模型。</p><section>{phase_table}</section>
<p>测试中的 {len(onset)} 个未来新转向命中 {int(onset.v2_turn.sum())} 个，另 {len(turns)-len(onset)} 个已转向起点命中 {int(turns[turns.current_phase.eq(1)].v2_turn.sum())} 个。误报中有 {int(mistaken.get(0,0))} 个实际停留、{int(mistaken.get(2,0))} 个实际移动。99 个真转向里有 {int((turns.angle_deg<55).sum())} 个夹角介于 45° 至 55°。这 99 窗的整段误差按窗口平均，当前版为 {turns.v1_path_cm.mean():.2f}、改进版为 {turns.v2_path_cm.mean():.2f}、原方案为 {turns.original_path_cm.mean():.2f} cm。</p>
<p>相同流程也给当前版试了单独调整报警阈值，它在验证集的候选均未同时满足上述两个限制。表内列出该情况下的候选作诊断。逐个真实转向的角度、位移、两版概率、预警与轨迹误差可供复核。<a href="99个转向窗口.csv">99 个真转向窗口</a> · <a href="turn_loss_interventions.csv">两种子配对消融</a></p>
<h2>为什么加入全图视觉后跨环境误差上升</h2>
<p>当前版把每只手共用的两帧 64×48 头部相机画面编码为 32 维特征，再拼入手部状态预测网络。相机缩略图采样约 {camera.fps.min():.1f} 至 {camera.fps.max():.1f} Hz。对全部 69 条记录检查了过去最近一帧的采集时间，每条记录第 95 百分位图像年龄的均值为 {camera.age_p95_s.mean():.3f} 秒。未出现使用预测起点之后图像的情况。</p>
<p>复算归档的非重叠窗口，全图特征使训练误差从 {raw.loc[('fit','B3'),'path_cm']:.2f} 降至 {raw.loc[('fit','B4'),'path_cm']:.2f} cm，测试却从 {raw.loc[('test','B3'),'path_cm']:.2f} 升至 {raw.loc[('test','B4'),'path_cm']:.2f} cm。又用同一带视觉槽的网络对照两种种子，全程输入真实全图与全程置零的测试误差分别为 {vv.loc[('aligned','aligned','test')]:.2f} 和 {vv.loc[('zero','zero','test')]:.2f} cm。在已经用全图训练的同一模型上，预测时置零也使测试误差降到 {vv.loc[('aligned','zero','test')]:.2f} cm。表中各值均为同一非重叠网格的环境等权误差。</p><section>{visual_table}</section>
<p>四个测试环境中有 {int((visual_gap>0).sum())} 个在当前版加入全图后误差上升，增加 {visual_gap.min():.2f} 至 {visual_gap.max():.2f} cm。在 {len(dpi)} 条具有同环境其他记录的样本上，用整条记录的平均视觉特征寻找最近的另一条记录，{int(dpi.hit.sum())} 次来自同环境，命中率 {dpi.hit.mean():.1%}，按数据组成随机匹配的期望为 {dpi.random_baseline.mean():.1%}。这支持全图保留了较强的环境信息，加上训练受益、留出环境退化的配对结果，视觉过拟合是合理解释。具体由哪些背景像素或物体驱动仍待定位。</p>
<figure><img src="data:image/png;base64,{png}" alt="转向精确率召回率曲线和带图、置零配对误差"><figcaption>左图以 760 个测试窗口比较排序与实际预警点，右图按相同网络结构的两个训练种子取均值。</figcaption></figure>
<h2>本次模型改动与部署选择</h2>
<p>动力学部分在手腕、另一只手、头部和手掌的多尺度运动向量上拟合位移。对右手的横向方向特征做镜像以对齐双手，共用跨坐标轴的回归系数，再由校准阶段概率调节有界速度残差。轨迹、质量模型、门控和条件扩散均从本题训练数据重新拟合。八条残差扩散样本与动力学中心组成九条有权轨迹。验证集选定的生成分支权重为{gate_text}，单条输出取{readout_label}。<a href="interval_sensitivity.csv">四种预设校准策略</a> · <a href="common_radius.csv">三套方案使用同一半径的覆盖率</a></p>
<p>另行测试了腕点投影附近两帧的亮度变化与相对背景光流，并限制视觉对每秒位移的修正上限。四分之一强度在四个验证环境均略有改善，整段平均误差只从 {local.at[('validation',0.),'ade_cm']:.4f} 降到 {local.at[('validation',.25),'ade_cm']:.4f} cm，低于事先设定的 0.02 cm 启用条件。冻结强度为 {lock['motion']['visual_weight']:.0f}，实际预测入口不读取视频。回顾测试同一运动中心的四分之一强度误差为 {local.at[('test',.25),'ade_cm']:.4f} cm，零强度为 {local.at[('test',0.),'ade_cm']:.4f} cm。<a href="local_visual_ablation.csv">全部视觉强度</a></p>
<section>{env_table}</section>
<h2>复算和核验</h2>
<p>三套方案共用原始数据、官方环境划分、腕点定义和逐起点评分口径。改进版与当前版保持相同的 10 个专家拟合、3 个门控、3 个外层校准训练环境。验证环境只用于选方案和决策规则。原方案使用其归档评测。最终锁定后才评测改进版的测试轨迹和转向预警。已对齐 {ver['matched_endpoints']:,} 个验证和测试终点、{ver['matched_windows']:,} 个非重叠窗口，并验证 {len(ver['prefixes'])} 个截断前缀在实际入口的阶段、位置和半径均与批量实验一致。</p>
<p>在项目 cws 环境中运行 <code>python run.py</code> 可从头复算。单起点推断示例为 <code>python predict.py {example['sample']} {example['hand']} {example['origin']}</code>。输出含三类校准概率、预警判定、30 步世界坐标、半径及九条候选轨迹的权重。<a href="../../c_restocking_toolkit/qsm_moe/q1_v2/run.py">从头复算入口</a> · <a href="three_way_metrics.csv">三方数值表</a> · <a href="paired_windows.csv.gz">逐窗口配对</a> · <a href="phase_validation_candidates.csv">阈值验证表</a> · <a href="frozen_before_test.json">冻结配置</a> · <a href="verification.json">核验记录</a></p></main></html>"""
    (OUT/"第一问优化与三方对比.html").write_text(out,encoding="utf8")
    print("Q1 report ready",flush=True)


if __name__=="__main__":
    report()
