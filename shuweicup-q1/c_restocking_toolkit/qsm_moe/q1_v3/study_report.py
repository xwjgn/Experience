"""Generate a self-contained Chinese report for the two requested Q1 quantities."""
import base64
import io
import os
import tempfile

os.environ.setdefault("MPLCONFIGDIR",os.path.join(tempfile.gettempdir(),"qsm_mpl_cache"))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import OUT,read_json

NAMES={"original":"原方案","v1":"QSM v1 · 全图融合","v2":"QSM v2 · 上一版","v3":"QSM v3 · 转向与覆盖优化"}


def read(name):
    return pd.read_csv(OUT/name)


def table(columns,rows):
    return pd.DataFrame(rows,columns=columns).to_html(index=False,border=0,justify="left",escape=True)


def chart(test,summary):
    fig,ax=plt.subplots(1,2,figsize=(9.4,3.55),constrained_layout=True)
    turns=test[test.turn]
    delta=np.sort(turns.v3_path_cm-turns.v2_path_cm)
    ax[0].bar(np.arange(1,len(delta)+1),delta,width=.85,color=np.where(delta<=0,"#247287","#bc7150"))
    ax[0].axhline(0,color="#444444",linewidth=.7)
    ax[0].set(xlabel="True-turn window, ordered by paired difference",ylabel="V3 minus V2 error (cm)",
              title="Paired path errors on the 99 turns")
    ax[0].grid(axis="y",alpha=.17)
    labels={"original":"Original","v1":"V1","v2":"V2","v3":"V3"}
    for key,color in (("original","#565a82"),("v1","#9a714f"),("v2","#6d838d"),("v3","#16708a")):
        r=summary.loc[key]
        ax[1].scatter(r.mean_radius_cm,100*r.coverage,s=55,color=color)
        ax[1].annotate(labels[key],(r.mean_radius_cm,100*r.coverage),xytext=(5,5),textcoords="offset points",fontsize=9)
    ax[1].axhline(95,color="#666666",linestyle="--",linewidth=1)
    ax[1].set(xlabel="Mean issued radius (cm)",ylabel="Complete paths covered (%)",
              title="Whole-path coverage and radius",xlim=(24,35),ylim=(90.5,99.5))
    ax[1].grid(alpha=.17)
    buf=io.BytesIO();fig.savefig(buf,format="png",dpi=145);plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def report():
    rows=read("four_way_metrics.csv")
    test=rows[rows.split.eq("test")].set_index("method")
    val=rows[rows.split.eq("validation")].set_index("method")
    t=test.loc["v3"];prev=test.loc["v2"]; orig=test.loc["original"]
    locked=read_json(OUT/"frozen_before_test.json")
    q=locked["interval"]
    allwindows=read("paired_windows.csv.gz")
    g=allwindows[allwindows.split.eq("test")]
    turns=g[g.turn]
    dirs=read("paired_differences.csv").query("split=='test'").set_index("comparison")
    relative=dirs.loc["v3 minus v2"]; historical=dirs.loc["v3 minus original"]
    envs=read("by_environment.csv").query("split=='test'")
    candidates=read("point_validation.csv")
    bands=read("coverage_validation.csv")
    verification=read_json(OUT/"verification.json")
    main=table(["冻结方案","99 窗平均轨迹误差 cm","全部整段覆盖","仅转向整段覆盖","全部平均半径 cm","全部起点整段误差 cm","第 30 帧误差 cm"],
        [[NAMES[k],f"{r.turn_ade_cm:.3f}",f"{r.covered:.0f}/{r.windows:.0f} · {r.coverage:.2%}",
          f"{r.turn_covered:.0f}/{r.turns:.0f} · {r.turn_coverage:.2%}",f"{r.mean_radius_cm:.2f}",f"{r.overall_ade_cm:.3f}",
          f"{r.overall_fde_cm:.3f}"] for k,r in test.iterrows()])
    environments=table(["环境","实际转向数","原方案转向误差 cm","v2 转向误差 cm","v3 转向误差 cm","v3 全窗覆盖","v3 平均半径 cm"],
       [[r.env,int(r.turns),f"{r.original_turn_ade_cm:.2f}",f"{r.v2_turn_ade_cm:.2f}",f"{r.v3_turn_ade_cm:.2f}",
          f"{r.v3_coverage:.2%}",f"{r.v3_radius_cm:.2f}"] for r in envs.itertuples()])
    valtable=table(["方案","75 个验证转向误差 cm","验证整段覆盖","验证转向覆盖","验证平均半径 cm"],
       [[NAMES[k],f"{r.turn_ade_cm:.3f}",f"{r.covered:.0f}/401 · {r.coverage:.2%}",
         f"{r.turn_covered:.0f}/75 · {r.turn_coverage:.2%}",f"{r.mean_radius_cm:.2f}"] for k,r in val.iterrows()])
    sub=read("subgroups.csv")
    sub=sub[sub.split.eq("test")].set_index(["group","method"])
    labels={"onset":"起点尚未转向","ongoing":"起点已经转向","alerted_turn":"起点预警命中的转向","unalerted_turn":"起点未预警的转向"}
    groups=table(["转向子组","窗口数","原方案 cm","v2 cm","v3 cm","v3 覆盖"],
       [[name,int(sub.loc[(k,"v3"),"n"]),*[f"{sub.loc[(k,m),'ade_cm']:.2f}" for m in ("original","v2","v3")],
         f"{sub.loc[(k,'v3'),'coverage']:.2%}"] for k,name in labels.items()])
    point_names={"none":"保持重新拟合的基础中心","brake":"减弱原方向外推","body_brake":"减弱相对头部外推",
                 "broad_regression":"增加转向样本权重的回归","turn_regression":"按转向概率融合的专用回归",
                 "hard_alert":"只对起点预警启用专用回归","turn_analog":"同类历史运动检索",
                 "regression_analog":"转向回归与检索","brake_regression":"减弱前向外推并叠加转向专用回归"}
    chosen=locked["point"]["selected"]
    choices=candidates[candidates.eligible].sort_values("objective").drop_duplicates("family")
    family_table=table(["可部署候选","幅度","验证转向误差 cm","验证全部轨迹误差 cm","门控训练环境转向误差 cm"],
             [[point_names[r.family],f"{r.strength:.2f}",f"{r.turn_cm:.3f}",f"{r.overall_ade:.3f}",
               f"{r.gate_turn_cm:.3f}"] for r in choices.itertuples()])
    schemes={"static":"固定外层因子","endpoint":"已到达终点反馈","completed":"已完整到达轨迹反馈",
             "static_endpoint":"固定与终点反馈范围的并集","static_completed":"固定与整轨反馈范围的并集","envelope":"三类范围的并集"}
    scheme_table=table(["覆盖候选中通过门槛的最窄配置","验证覆盖","最差验证环境","验证转向覆盖","平均半径 cm"],
        [[schemes[r.scheme],f"{r.coverage:.2%}",f"{r.minimum_environment:.2%}",f"{r.turn_coverage:.2%}",f"{r.radius_cm:.2f}"]
          for r in bands[bands.eligible].sort_values("radius_cm").drop_duplicates("scheme").itertuples()])
    target=int(np.ceil(len(g)*.95))
    png=chart(g,test)
    html=f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第一问 · 真实转向误差与整段覆盖</title><style>
body{{font:16px/1.65 'Microsoft YaHei','Segoe UI',sans-serif;color:#20323b;background:white;margin:0}}main{{max-width:1110px;margin:36px auto;padding:0 30px}}h1{{font-size:31px;line-height:1.3}}h2{{font-size:22px;margin-top:31px}}table{{border-collapse:collapse;font-size:13.5px;width:100%;margin:12px 0 18px;font-variant-numeric:tabular-nums}}td,th{{padding:8px;border-bottom:1px solid #d4dee2;text-align:right}}td:first-child,th:first-child{{text-align:left}}th{{background:#e8f0f2}}img{{width:100%;height:auto}}figure{{margin:16px 0}}figcaption{{font-size:14px;color:#425a65}}p{{margin:10px 0}}a{{color:#146281}}code{{background:#edf2f4;padding:1px 3px}}section{{overflow-x:auto}}@media(max-width:700px){{main{{padding:0 12px}}}}@media print{{main{{margin:0;padding:0}}h2{{break-after:avoid}}tr,figure{{break-inside:avoid}}}}</style><main>
<h1>第一问　真实转向误差与 95% 整段覆盖</h1>
<p>四个回顾测试环境中，同一批 99 个真实转向的平均腕点轨迹误差从上一版的 <b>{prev.turn_ade_cm:.2f} cm</b> 降至 <b>{t.turn_ade_cm:.2f} cm</b>，原方案为 {orig.turn_ade_cm:.2f} cm。整段覆盖达到 <b>{int(t.covered)}/{len(g)}（{t.coverage:.2%}）</b>，95% 目标需要至少覆盖 {target} 个窗口。新范围的平均半径由 {prev.mean_radius_cm:.2f} 增至 {t.mean_radius_cm:.2f} cm。</p>
<h2>在同一批窗口比较四套方案</h2><section>{main}</section>
<p>转向窗口的指标先在每窗 30 帧上平均三维欧氏位置误差，再对 99 窗等权平均。转向指未来第 10 帧满足既定速度和 45° 转角规则，分母同时包含 {int(turns.issue_alert.sum())} 个已报警和 {int((~turns.issue_alert).sum())} 个未报警的真实转向。整段覆盖按 760 个非重叠窗口计数，30 帧的参考腕点均在相应三维球内才记一次覆盖。半径按窗口和时刻平均。全部起点整段误差与终点误差分别在 22,704 和 23,649 个合格起点上计算，四个环境等权。</p>
<p>对上一版逐窗配对，{int(relative.turn_improved)} 个转向窗口误差下降，{int(relative.turn_worse)} 个上升，配对差值中位数为 {relative.median_delta_cm:+.3f} cm。与原方案相比，均值差为 {historical.mean_delta_cm:+.3f} cm，{int(historical.turn_improved)} 个转向窗口误差下降。新模型对两项指定指标的改善伴随全部起点整段误差从 {prev.overall_ade_cm:.3f} 变为 {t.overall_ade_cm:.3f} cm。样本来自四个环境，这些均值为回顾测试结果。</p>
<figure><img src="data:image/png;base64,{png}" alt="99 个转向窗口逐窗误差差值与四套方案的半径覆盖对照"><figcaption>左图零线以下表示新模型在该转向窗口的误差更小。右图的虚线对应 95% 整段覆盖目标。</figcaption></figure>
<h2>转向时如何修正预测</h2>
<p>仅用官方训练环境重新拟合基础模型、阶段分类器与残差扩散模型。候选修正利用预测起点的转向概率、最近各时段腕点速度、与头部的相对运动及方向变化。专用回归在 10 个训练环境中用未来确实转向的轨迹学习位移残差。每次实际预测只读取起点及以前的观测，再按当时可计算的转向风险决定修正强度。九条轨迹支持点接受相同的位移修正。</p>
<p>冻结的点预测规则为「{point_names[chosen['family']]}」。选择时同时检查验证集非重叠转向均值、密集转向起点均值及逐环境均值，并限制全部验证起点的性能退化。验证转向误差从 {val.loc['v2','turn_ade_cm']:.3f} 降到 {val.loc['v3','turn_ade_cm']:.3f} cm。以下列出各方法满足总体约束的最优幅度。</p><section>{family_table}</section>
<section>{groups}</section>
<h2>整段范围为何能覆盖更多窗口</h2>
<p>在三个训练环境中学习运动状态与路径误差尺度的关系，另外三个训练环境只做外层校准。每个校准窗口的分数取其 30 帧中最大的「位置误差 / 当时预测尺度」。候选包括固定因子、过去终点反馈、整条旧轨迹完全到达后的反馈及其范围并集。最后按四个验证环境选择满足覆盖筛选条件且平均半径最小的规则。</p>
<p>冻结配置采用「{schemes[q['scheme']]}」，训练校准窗口数为 {locked['calibration_grid']}。采用 {q['level']:.1%} 经验分位后，外层半径倍数为 {q['factors']['static']:.3f}（固定因子）。验证时覆盖 {q['covered']}/{q['n']}，覆盖最少的环境为 {q['minimum_environment']:.2%}，验证转向覆盖为 {q['turn_coverage']:.2%}。验证筛选要求总体覆盖的一侧 90% Wilson 下界达到 95%，并要求各环境及验证转向子组的经验覆盖均达到 95%。同一记录的窗口具有关联，Wilson 值在这里用于留出余量，跨新环境的统计保证尚需更多独立环境。</p>
<p>固定上一版发出的半径，仅把中心换为新预测，测试只覆盖 {int(g.unchanged_band_on_new_center.sum())}/{len(g)}。使用新中心和新半径后覆盖 {int(t.covered)}/{len(g)}，相对上一版新增覆盖 {int(relative.coverage_gained)} 窗、丢失 {int(relative.coverage_lost)} 窗。转向子组由 {int(prev.turn_covered)}/99 提高到 {int(t.turn_covered)}/99。半径增加 {t.mean_radius_cm-prev.mean_radius_cm:.2f} cm，应与覆盖提升一起报告。</p>
<section>{valtable}</section><section>{scheme_table}</section>
<h2>逐环境实测与复算</h2><section>{environments}</section>
<p>依据官方 16 个训练、4 个验证、4 个测试环境划分。基础参数、转向修正、范围模型均在独立目录从头拟合，旧参数仅以哈希核验，旧预测表用于配对。冻结配置写出后才计算本版测试指标。已按环境、记录、左右手与起点核对全部 99 个转向的身份，并用 {len(verification['prefix_checks'])} 个独立截断前缀复算在线输出。先前项目已使用过这批测试环境。</p>
<p><a href="99个转向窗口逐窗对照.csv">99 个转向逐窗对照</a> · <a href="four_way_metrics.csv">四套方案实测表</a> · <a href="paired_windows.csv.gz">全部验证与测试窗口</a> · <a href="turn_by_horizon.csv">逐预测步长误差</a> · <a href="coverage_validation.csv">覆盖候选的验证结果</a> · <a href="frozen_before_test.json">冻结配置</a> · <a href="verification.json">核验记录</a> · <a href="../../c_restocking_toolkit/qsm_moe/q1_v3/run_study.py">从头复算入口</a> · <a href="../../c_restocking_toolkit/qsm_moe/q1_v3/forecast.py">单腕点预测入口</a></p></main></html>"""
    (OUT/"第一问转向误差与整段覆盖.html").write_text(html,encoding="utf8")
    print("Two-target report written.",flush=True)


if __name__=="__main__":
    report()
