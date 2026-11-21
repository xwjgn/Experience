"""Write the user-facing results from the frozen replays."""
import pandas as pd
from wrist_history import M4_DIR,OUT,read


def table(headers,rows):
    return '<div class="scroll">'+pd.DataFrame(rows,columns=headers).to_html(index=False,border=0)+'</div>'


def report():
    intervals=pd.read_csv(OUT/"intervals.csv").set_index("model")
    events=pd.read_csv(OUT/"turn_alerts.csv").set_index(["rule","model"])
    points=pd.read_csv(OUT/"point_errors.csv").set_index("model")
    by_env=pd.read_csv(OUT/"by_environment.csv")
    steps=pd.read_csv(OUT/"by_horizon.csv")
    causes=pd.read_csv(OUT/"miss_diagnosis.csv").set_index("group")
    previous=pd.read_csv(M4_DIR/"comparison.csv")
    locks=read(OUT/"frozen_before_test.json")
    point_lock=read(OUT/"point_locked_before_test.json")
    check=read(OUT/"online_verification.json")
    chosen=locks["detector"]
    old=intervals.loc["previous_low_turn"]
    paired=intervals.loc["mixed_center_same_window_budget"]
    tight=intervals.loc["mixed_center_compact"]
    diag=causes.loc["old14"]
    prefix="../../c_restocking_toolkit/q1_origin_refinement"
    def row(label,m):
        return (label,f"{m.ade_cm:.2f}",f"{m.turn_ade_cm:.2f}",f"{m.radius_cm:.2f}",
                f"{int(m.covered)}/760（{100*m.coverage:.2f}%）",f"{int(m.turn_covered)}/99")
    rows=[]
    for key,label in (("m4","原 M4"),("previous_low_turn","上一轮低分收紧"),("same_window_budget","仅重新分配 30 步半径"),
           ("mixed_center_previous_budget","仅调整轨迹中心，使用上一轮半径"),
           ("mixed_center_same_window_budget","调整中心和各步半径，逐窗保持上一轮平均半径"),
           ("mixed_center_compact","调整中心，允许进一步收紧低风险窗口")):
        rows.append(row(label,intervals.loc[key]))
    evrows=[]
    for key,label in (("m4","M4 原阶段头"),("previous_full","上一轮完整头手特征分类器"),
           ("tree","新增掌部运动与外推残差的树模型"),("single","只训练第 10 帧转向的神经网络"),
           ("wrist_aux","加入多步转向与未来腕部运动监督"),("joint_aux","再加入未来头部运动监督，验证选定"),
           ("ensemble","三个独立预测分数平均"),("late_fusion","结合已发出的 M4 阶段分数")):
        a=events.loc[("same_alarm_count",key)]
        reported=("online",key) in events.index
        b=events.loc[("online",key)] if reported else None
        evrows.append((label,f"{int(a.tp)}/99",f"{a.average_precision:.3f}",f"{int(b.tp)}/99" if reported else "—",
                       int(b.fp) if reported else "—",f"{100*b.precision:.1f}%" if reported else "—"))
    phases=[]
    for lower,upper in ((1,3),(4,10),(11,20),(21,30)):
        take=steps[steps.step.between(lower,upper)].groupby("model").radius_cm.mean()
        phases.append((f"{lower}–{upper}",int(diag[f"first_{lower}_{upper}"]),int(diag[f"worst_{lower}_{upper}"]),
                       f"{take['previous_low_turn']:.2f}",f"{take['mixed_center_same_window_budget']:.2f}"))
    env_rows=[]
    for env,block in by_env.groupby("env"):
        frame=block.set_index("model")
        base,a,b=(frame.loc[k] for k in ("previous_low_turn","mixed_center_same_window_budget","mixed_center_compact"))
        env_rows.append((env,int(base.n),int(base.turns),f"{100*base.coverage:.2f}%",f"{100*a.coverage:.2f}%",
                         f"{100*b.coverage:.2f}%",f"{base.turn_ade_cm:.2f} → {a.turn_ade_cm:.2f}"))
    hist=[]
    for name,band,label in (("published_original","archived","原方案"),("published_qsm_low_turn","archived","已有 QSM-MoE 起点收紧"),
                             ("published_quality_stage","cafht_cptc","上一版独立 LSTTN")):
        m=previous.query("split=='test' and model==@name and band==@band").iloc[0]
        hist.append(row(label,m))
    single=points.loc["fresh_expert"]
    train=read(OUT/"fit_protocol.json")
    mix=point_lock["share"]
    headers=["方案","总体轨迹误差（cm）","转向轨迹误差（cm）","平均半径（cm）","整段覆盖","转向整段覆盖"]
    html=f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第一问 · 转向预警和分步预测半径</title>
<style>
*{{box-sizing:border-box}}body{{margin:0 auto;max-width:1050px;padding:30px 24px 60px;color:#1d3440;font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif}}
h1,h2{{line-height:1.35}}h1{{font-size:clamp(27px,3.5vw,39px)}}h2{{font-size:22px;margin:32px 0 8px}}.scroll{{overflow:auto}}
table{{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}}td,th{{padding:8px;border-bottom:1px solid #d5e0e3;text-align:right;white-space:nowrap}}
th{{background:#edf3f2}}th:first-child,td:first-child{{text-align:left;white-space:normal}}a{{color:#075b77}}.lead{{font-size:18px}}code{{overflow-wrap:anywhere}}
</style><main><h1>第一问 · 转向预警和分步预测半径</h1>
<p class="lead">沿用上一轮每个窗口的平均半径，调整轨迹中心和 30 个预测步各自的半径。总体整段覆盖从 <b>{int(old.covered)}/760</b> 提高到 <b>{int(paired.covered)}/760</b>，99 个真实转向均被覆盖。转向轨迹误差从 <b>{old.turn_ade_cm:.2f}</b> 降到 <b>{paired.turn_ade_cm:.2f} 厘米</b>。</p>
<h2>14 个新增漏失发生在哪里</h2>
<p>这 14 个窗口均在上一轮收紧后才漏出。按当时可见的腕点运动，14 个起点都处于平移，其最近 5 帧腕速中位数为 {diag.median_wrist_speed:.2f} m/s，同样已收紧且成功覆盖的窗口为 {causes.loc['other_previously_routed_and_covered','median_wrist_speed']:.2f} m/s。单靠速度大小难以把它们筛出来。</p>
<p>以下用误差除以当步原半径定位最严重的越界。按该帧沿起点腕速方向与垂直方向的误差分量，{int(diag.mainly_undershot)} 个主要表现为沿原方向预测得偏短，{int(diag.mainly_lateral)} 个主要是侧向偏差。越界位置和误差方向均来自回顾诊断，预测器只用起点及此前的观测。</p>
{table(["步长（帧）","首次越界窗口","最严重越界窗口","上一轮平均半径（cm）","同宽新方案平均半径（cm）"],phases)}
<h2>点误差、整段覆盖和半径</h2>
{table(headers,rows)}
<p>每个窗口包含未来 30 个腕点。轨迹误差取 30 步三维欧氏误差的平均，整段覆盖要求全部 30 步均落在相应半径以内。同宽组合比上一轮多覆盖 {int(paired.covered-old.covered)} 个窗口，其中 {int(paired.old14_saved)} 个来自那 14 个新增漏失，{int(paired.other_previously_missed_now_hit)} 个是原 M4 已漏掉的窗口，另有 {int(paired.newly_missed_vs_previous)} 个从覆盖转为漏失。紧半径版将平均半径降至 {tight.radius_cm:.2f} 厘米，四个测试环境的最低覆盖为 {100*tight.least_env:.2f}%。</p>
<p>点预测专家以起点腕速方向为一条坐标轴，学习未来八个时间点的腕点位移，再插值得到 30 步轨迹。线性回归提供基本运动，随机树估计其余偏差。专家独立使用时，总体和转向误差分别为 {single.ade_cm:.2f} 和 {single.turn_ade_cm:.2f} 厘米。验证环境选定 {mix:.0%} 的新专家和 {1-mix:.0%} 的 M4 已发出中心。改变中心后，紧半径版用留一验证环境的误差复核校准。相同半径的点预测与区间交叉对照固定这两项选择直接回放。</p>
<p>同宽版在每个窗口内重新分配半径。验证选择给最初几个预测步稍多余量，随后平滑回落，同时严格保持 30 个半径之和。进一步收紧版另行评估过去真实腕点偏离短时匀速外推的程度，在预测转向风险和物理误差风险均较低且近期质量正常的窗口收紧。宽度和误差应各自比较。</p>
<h2>预测起点能提前认出多少转向</h2>
{table(["预警模型","同为 181 次预警时命中","排序平均精度 AP","在线命中","在线误报","在线精确率"],evrows)}
<p>标签沿用未来第 10 帧的转向定义，提前约 0.33 秒。网络按当前停留、平移或转向状态学习后续变化。头手联合版本在训练时同时预测未来 5、10、20 帧的腕部位移、头部平移及转动。在线短时预警阈值仅在已到达第 10 帧、腕部观测有效且实际未转向后，用这类过去的分数更新。训练与验证先选定 <code>{chosen}</code>。同额排序使用整批得分检验排序能力，实际在线误报和命中使用冻结的逐起点规则。</p>
<p>加入头部运动的训练监督在验证选型中占优，在测试集同额排序中，腕部辅助版本命中了更多转向。两者差异不能单独证明头部信号在新环境具有稳定额外收益。可选的联合预警只额外读取当前 M4 已发布的阶段概率。</p>
<h2>四个测试环境和更早的方案</h2>
{table(["环境","窗口","转向","上一轮覆盖","同宽新方案覆盖","紧半径版覆盖","转向误差（cm）"],env_rows)}
{table(headers,hist)}
<p>较早方案来自同一批窗口的结果档案。此处的环境和记录均已在前期工作中使用，所有测试对照属于回顾验证。未来新环境的覆盖率仍需另行检验。</p>
<h2>训练、冻结与在线复现</h2>
<p>全部新参数从头训练。算法共享原始观测读取、物理特征和评分口径。10 个训练环境提供 {train['fit_windows']:,} 个学习窗口，3 个训练环境选训练轮次，另 3 个训练环境用于预警参考分布。4 个官方验证环境选模型和半径，随后冻结，再回放 4 个测试环境。评估和组合使用已有 M4 在相应起点发出的预测结果。程序禁止读取其他方案的权重文件。</p>
<p>逐窗同宽组合需要该起点上一轮已经发出的宽度倍率。独立转向预警和单独腕点专家可以只用原始观测运行。在线入口已在三种数据来源、双手、全部测试环境的 {check['prefixes']} 个真实前缀上核对，其中包括 14 个旧漏失。 <a href="{prefix}/run.py">完整复现</a> · <a href="{prefix}/refine.py">在线预测入口</a> · <a href="online_verification.json">回放核验</a></p>
<p><a href="intervals.csv">覆盖和半径</a> · <a href="point_errors.csv">点预测</a> · <a href="turn_alerts.csv">转向预警</a> · <a href="99个真实转向逐窗对照.csv">99 个真实转向</a> · <a href="14个新增漏失逐窗复核.csv">14 个漏失</a> · <a href="by_environment.csv">逐环境结果</a> · <a href="by_horizon.csv">逐步结果</a></p></main></html>"""
    path=OUT/"第一问预警与分步半径改进实测.html"
    path.write_text(html,encoding="utf8")
    print(path,flush=True)


if __name__=="__main__":
    report()
