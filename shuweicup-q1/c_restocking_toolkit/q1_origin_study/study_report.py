"""Publish the comparison with each metric tied to its own task and evaluation cohort."""
import json
import pandas as pd
from origin_signals import OUT


def report():
    event = pd.read_csv(OUT/"event_recognition.csv").set_index(["split","rule","model"])
    coverage = pd.read_csv(OUT/"intervals.csv").set_index("policy")
    stages = pd.read_csv(OUT/"origin_phase_and_quality.csv").set_index(["group","origin_stage"])
    prior = json.loads((OUT/"frozen_before_replay.json").read_text(encoding="utf8"))
    checked = json.loads((OUT/"prefix_replay.json").read_text(encoding="utf8"))
    ablations = []
    for key,label in (("m4_score","已有 M4 阶段头"),("wrist","只用腕部"),("wrist_hand","腕部＋本手手指和另一手"),
                      ("wrist_head","腕部＋头部运动"),("wrist_hand_head","腕部＋手指、另一手和头部")):
        val = event.loc[("validation","same_alarm_count_ranking",key)]
        test = event.loc[("test","same_alarm_count_ranking",key)]
        online = event.loc[("test","online_frozen",key)]
        ablations.append((label,f"{int(val.true_alert)}/75",f"{int(test.true_alert)}/99",f"{test.average_precision:.3f}",
                          f"{int(online.true_alert)}/99",int(online.false_alert)))
    def table(rows, headings):
        return '<div class="scroll">'+pd.DataFrame(rows,columns=headings).to_html(index=False,border=0)+'</div>'
    rules = []
    for key,label in (("m4","原 M4"),("low_turn","低转向分数时减去 20%"),("low_turn_same_width_diagnostic","所有窗口等比例缩至相同平均半径"),
                      ("tail","整段误差风险较低时减去 20%"),("tail_same_width_diagnostic","所有窗口等比例缩至误差规则的平均半径")):
        m = coverage.loc[key]
        rules.append((label,f"{m.radius_cm:.2f}",f"{int(m.covered)}/760（{100*m.coverage:.2f}%）",
                      f"{int(m.turn_covered)}/99",int(m.additional_misses)))
    low = coverage.loc["low_turn"]
    original = coverage.loc["m4"]
    cor = prior["routing"]["low_turn"]
    code = "../../c_restocking_toolkit/q1_origin_study"
    html = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>第一问 · 起点转向预警与半径收紧</title><style>
body{{max-width:1000px;margin:3em auto;padding:0 1.2em;font:16px/1.65 "Microsoft YaHei",system-ui,sans-serif;color:#1a303b}}
h1,h2{{line-height:1.35}}h2{{font-size:1.32em;margin-top:1.7em}}a{{color:#075871}}.scroll{{overflow:auto}}table{{border-collapse:collapse;width:100%;font-size:.92em}}th,td{{padding:.48em;text-align:right;border-bottom:1px solid #cfdcdd}}th{{background:#edf3f5}}td:first-child,th:first-child{{text-align:left}}code{{background:#f1f5f6}}
</style><main><h1>第一问 · 起点转向预警与半径收紧</h1>
<p>在已有 M4 的同一批 760 个测试窗口上，起点分数最低且近期腕点观测正常的 {int(low.routed)} 个窗口采用原半径的 80%。平均半径从 {original.radius_cm:.2f} 降到 <b>{low.radius_cm:.2f} 厘米</b>，整段覆盖 <b>{int(low.covered)}/760</b>，真实转向覆盖 <b>{int(low.turn_covered)}/99</b>。新漏掉的 {int(low.additional_misses)} 个均为非转向窗口。M4 的预测中心保持相同，99 个转向的平均轨迹误差为 {low.turn_ade_cm:.2f} 厘米。</p>
<h2>起点看得见什么</h2>
<p>真实转向按未来第 10 帧的速度和方向变化划分，约提前 0.33 秒判定。从起点回看，99 个中有 {int(stages.loc[('all','dwell'),'windows'])} 个当时为停留，{int(stages.loc[('all','translate'),'windows'])} 个为平移，{int(stages.loc[('all','turn'),'windows'])} 个已经转向。漏报的 47 个中，对应数量为 19、16、12。全部 99 个在起点及此前 9 帧的目标腕点坐标完整，赛题质量标记均正常。</p>
<p>独立的转向二分类器使用相同容量的浅层梯度提升树，从 10 个训练环境的 {prior['fit_rows']:,} 个窗口重新拟合。腕部特征包括最近 3、5、10、20 帧的速度变化、转角与轨迹直线度。头部特征包含角速度、相对速度和较早头部运动与当前腕速的夹角。手部附加特征包含手指开合与双腕协同。轨迹真值和标签共用原评分口径，训练不读取旧模型权重。</p>
{table(ablations,["输入","验证命中（同为124次预警）","测试命中（同为181次预警）","测试 AP","冻结阈值测试命中","冻结阈值测试误报"])}
<p>固定数量取该批窗口得分最高的若干个，只检验排序。在线阈值由另外三个训练环境的 {prior['threshold_rows']} 个不重叠窗口确定。头部历史在验证集改善固定预算的命中数，在测试集上与腕部单独使用持平。</p>
<h2>收紧规则的代价</h2>
{table(rules,["规则","平均半径（cm）","整段覆盖","转向覆盖","新漏掉窗口"])}
<p>低分规则使用完整头手特征模型的分数。目标腕点近 10 帧质量正常且分数不超过 {cor['cut']:.4f} 时，每一步半径乘 {prior['shrink_factor']:.2f}。高分保留原模型范围。另一个对照用 7 个起点量估计窗口内最大归一化误差的第 85 百分位数，据此决定能否收紧。在四个验证环境内轮流留出一个环境，选择满足总体和转向各 95% 且不增加转向漏失的收紧比例，随后冻结并回放测试。相同平均半径的均匀收紧属于事后对照。</p>
<p>低分规则在留一验证中收紧 {cor['validation']['routed']}/401 个窗口，在测试中收紧 {int(low.routed)}/760 个，后者有 {int(low.routed_turns)} 个真实转向。该差异提示分数分布随环境变化。测试环境此前已用于其他研究，是回顾对照。 <a href="environment_coverage.csv">逐环境覆盖</a> · <a href="validation_routing.csv">验证规则选择</a></p>
<h2>三篇论文在本题的位置</h2>
<p><b>Uni-Hand</b> 联合预测手和头，使用视觉语言融合与双分支扩散。本题优先检验起点前头部是否给腕部提供额外信号，直接用已测得的头部转动和相对腕速就能消融。既有实验在另一组 23,649 个起点预测 1 秒终点，仅轨迹扩散和加入视频、文字后的平均误差依次为 23.21 和 25.27 厘米。 <a href="https://irmvlab.github.io/unihand.github.io/">作者项目页</a> · <a href="../三种文献方法实测/visual_point.csv">既有终点结果</a></p>
<p><b>Bayesian KalmanNet</b> 用随机神经网络计算滤波增益，由多次采样估计状态误差协方差。对本题可把截至起点的滤波创新与不确定性用于风险评分，再用独立环境校准整段误差。现有 1 秒终点实验在测试环境等权下，学习增益为 17.52 厘米，冻结增益为 17.55 厘米。 <a href="https://doi.org/10.1109/TSP.2025.3581703">论文</a> · <a href="../三种文献方法实测/gain_sensitivity.csv">已有增益消融</a></p>
<p><b>BayOTIDE</b> 用低秩时间因子表示多个通道，以高斯过程作为先验，再用等价随机微分方程进行在线推断。腕点被遮挡但指根仍可见时，可向下游传入补点、方差及缺失掩码。已有第二问固定观测实验在部分通道缺失时，在线低秩模型为 1.25 厘米，指根位移法为 1.30 厘米。全部通道缺失时，两种可用模型中速度衰减为 4.46 厘米，在线低秩模型为 8.06 厘米。 <a href="https://proceedings.mlr.press/v235/fang24d.html">论文</a> · <a href="../三种文献方法实测/imputation_summary.csv">已有缺测结果</a></p>
<h2>复算与回放</h2><p>在新目录运行 <code>python evaluate_origin.py</code> 生成本实验。单个起点可运行 <code>python issue_origin.py E018 right 70</code>，获得风险分数、预警与半径倍率。在线核验使用三种数据来源、两只手、四个测试环境的 {checked['raw_prefixes']} 个原始前缀。 <a href="{code}/evaluate_origin.py">独立实验</a> · <a href="{code}/issue_origin.py">在线入口</a> · <a href="prefix_replay.json">回放核验</a></p>
<p><a href="event_recognition.csv">转向识别</a> · <a href="intervals.csv">半径与覆盖</a> · <a href="paired_origins.csv">760 个起点逐窗明细</a> · <a href="frozen_before_replay.json">冻结规则</a></p></main></html>"""
    (OUT/"第一问起点转向与收紧实测.html").write_text(html,encoding="utf8")


if __name__=="__main__":
    report()
