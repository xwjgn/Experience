"""Chinese comparison for the first-question wrist forecasting experiment."""
from html import escape
import os
from urllib.parse import quote
import pandas as pd

from paths_q1 import HERE,OUT,PREVIOUS,read_json


def table(cols,rows):
    head = "".join("<th scope='col'>"+escape(str(c))+"</th>" for c in cols)
    body = "".join("<tr>"+"".join("<td>"+escape(str(c))+"</td>" for c in r)+"</tr>" for r in rows)
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def count(i,n):
    return f"{int(i)}/{int(n)}（{i/n:.2%}）"


def local_link(path,label):
    target = quote(os.path.relpath(path,OUT).replace(os.sep,"/"),safe="/")
    return f'<a href="{target}">{escape(label)}</a>'


def report():
    df = pd.read_csv(OUT/"comparison.csv")
    test = df[df.split=="test"].set_index(["model","calibration"])
    val = df[df.split=="validation"].set_index(["model","calibration"])
    current = test.loc[("qsm_low_turn","archived")]
    main = test.loc[("quality_stage","cafht_cptc")]
    mval = val.loc[("quality_stage","cafht_cptc")]
    raw = test.loc[("quality_stage","marginal_95")]
    anchors = pd.read_csv(OUT/"current_kalman_anchor.csv").set_index("split")
    matched = pd.read_csv(OUT/"matched_radius_diagnostics.csv").set_index(["split","diagnostic"])
    same = matched.loc[("test","equal_split_mean_diagnostic")]
    vmatch = matched.loc[("validation","equal_split_mean_diagnostic")]
    model = read_json(OUT/"frozen_before_test.json")["experiments"]["quality_stage"]
    spec = model["intervals"]["cafht_cptc"]["recipe"]
    audit = read_json(OUT/"verification.json")
    stages = pd.read_csv(OUT/"stage_metrics.csv")
    alarms = stages[(stages.model=="quality_stage")&(stages.split=="test")].set_index("rule")
    pairs = pd.read_csv(OUT/"99个转向窗口对照.csv")
    def metric(label,r):
        return [label,f"{r.ade_cm:.2f}",f"{r.turn_ade_cm:.2f}",count(r.covered,r.windows),
                count(r.turn_covered,r.turns),f"{r.mean_radius_cm:.2f}"]
    main_table = table(["方案","全部 ADE（厘米）","转向 ADE（厘米）","总体整段覆盖","转向整段覆盖","平均半径（厘米）"],[
        metric("原方案",test.loc[("original","archived")]),metric("当前 QSM v3",test.loc[("qsm_v3","archived")]),
        metric("当前 QSM 低转向规则",current),metric("LSTTN＋CAFHT/CPTC",main),
        metric("LSTTN＋仅按总体和转向 95% 选择的校准",test.loc[("quality_stage","cafht_cptc_pooled_target")]),
        metric("LSTTN 等宽对照（事后）",same)])
    bands = table(["同一 LSTTN 中心","逐帧覆盖","整段覆盖","真实转向整段覆盖","平均半径（厘米）"],[
        [label,f"{m.frame_coverage:.2%}",count(m.covered,m.windows),count(m.turn_covered,m.turns),f"{m.mean_radius_cm:.2f}"]
        for name,label in (("marginal_95","原始混合分布的逐步 95% 球"),("static_joint","固定整段校准"),
                          ("cafht","CAFHT 式全局反馈"),("cafht_cptc","加入阶段条件反馈"))
        for m in (test.loc[("quality_stage",name)],)])
    pr = pd.read_csv(OUT/"probability_metrics.csv").query("split=='test'").set_index("model")
    abl = table(["从头训练的配置","全部 ADE","转向 ADE","概率轨迹能量分数（厘米，越小越好）"],[
        [label,f"{m.ade_cm:.2f}",f"{m.turn_ade_cm:.2f}",f"{pr.at[name,'energy_score_cm']:.2f}"]
        for name,label in (("quality_stage_one","单种子完整模型"),("short_only","仅保留短历史"),
                           ("no_quality","关闭质量标签与可靠性门控"),("single_gaussian","单个高斯轨迹头"),
                           ("quality_stage","双种子完整模型"))
        for m in (test.loc[(name,"cafht_cptc")],)])
    env = pd.read_csv(OUT/"by_environment_and_quality.csv")
    env = env[(env.split=="test")&(env.model=="quality_stage")&(env.calibration=="cafht_cptc")&(env.group=="environment")]
    env_table = table(["环境","窗口","整段覆盖","转向平均误差（厘米）","平均半径（厘米）"],[
        [e.value,int(e.windows),count(e.covered,e.windows),f"{e.turn_ade_cm:.2f}",f"{e.mean_radius_cm:.2f}"] for e in env.itertuples()])
    refs = ' · '.join([
        '<a href="https://arxiv.org/abs/2403.16495">LSTTN 论文</a>','<a href="https://github.com/GeoX-Lab/LSTTN">LSTTN 作者代码</a>',
        '<a href="https://proceedings.mlr.press/v235/zhou24l.html">CAFHT 论文</a>','<a href="https://github.com/FionaZ3696/CAFHT">CAFHT 作者代码</a>',
        '<a href="https://proceedings.neurips.cc/paper_files/paper/2025/hash/12271b64c483ad8f6192eb6aaa102044-Abstract-Conference.html">CPTC 论文</a>',
        '<a href="https://github.com/Rose-STL-Lab/CPTC">CPTC 作者代码</a>'])
    files = ' · '.join([local_link(HERE/"run.py","独立重训入口"),local_link(HERE/"issue_lsttn.py","LSTTN 在线预测入口"),
        local_link(HERE.parent/"q1_stable_windows"/"issue_stable.py","第一问模型切换入口"),
        local_link(HERE.parent/"q1_v3"/"forecast.py","现有 QSM 点预测实现"),local_link(OUT/"comparison.csv","完整比较"),
        local_link(OUT/"99个转向窗口对照.csv","99 个转向逐窗结果"),local_link(OUT/"matched_radius_diagnostics.csv","等宽对照"),
        local_link(OUT/"stage_metrics.csv","阶段识别"),local_link(OUT/"probability_metrics.csv","概率分布评分"),
        local_link(OUT/"frozen_before_test.json","冻结参数"),local_link(OUT/"verification.json","因果与签发核验")])
    html = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>第一问 · 用质量感知 LSTTN 替代 Kalman 动力学预测</title><style>
body{{max-width:1030px;margin:40px auto;padding:0 22px 54px;font:16px/1.7 'Microsoft YaHei',sans-serif;color:#20333b}}
h1,h2{{line-height:1.35}}h2{{margin:1.8em 0 .5em;font-size:21px}}a{{color:#09586a}}.scroll{{overflow:auto}}
table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{border-bottom:1px solid #d5dfe1;padding:8px 10px;text-align:right}}
td:first-child,th:first-child{{text-align:left}}th{{background:#eaf1f2}}li{{margin:.45em 0}}
</style><h1>第一问 · 质量感知 LSTTN 独立替换试验</h1>
<p>已完成从头训练的 30 步三维腕点预测器。新模型在全部测试窗口上的平均轨迹误差为 <strong>{main.ade_cm:.2f} 厘米</strong>，当前 QSM 为 {current.ade_cm:.2f} 厘米。在优先考察的 99 个转向窗口上，新模型为 <strong>{main.turn_ade_cm:.2f} 厘米</strong>，当前 QSM 为 {current.turn_ade_cm:.2f} 厘米。</p>
<h2>现有预测链路与替换范围</h2>
<p>现有 QSM 用观测质量调整 Kalman 测量噪声，得到位置和速度，按 0.4 秒时间常数做衰减外推。岭回归和阶段条件神经速度残差修正运动预测，随后用扩散残差生成八条候选，与运动预测作加权几何中位数读出，最后应用转向修正。上一轮的低转向规则在此中心周围收紧半径。单独以这条 Kalman 物理外推作预测，测试总体 ADE 为 {anchors.at['test','ade']:.2f} 厘米，转向 ADE 为 {anchors.at['test','turn']:.2f} 厘米。</p>
<p>替换模型直接读取双手与头部的历史几何、实测时间戳和观测质量。所有可训练参数从随机初始化开始。现有方案的点预测、协方差、阶段分数与权重都未进入新模型的训练或在线签发。</p>
<h2>新模型如何产生一条轨迹和一个范围</h2><ol>
<li>腕点与五个指尖作空间注意力汇聚。报告质量、有效性、距上次有效观测的时长等信号调节历史特征，缺失节点使用掩码。实际时差用于估计历史速度。</li>
<li>短分支用膨胀因果卷积读取最近 15 帧。长分支把最多 96 帧按 6 帧分块，使用作者仓库中的时间分块嵌入和可学习位置编码，再用 Transformer 和膨胀卷积提取趋势。门控融合两支。原交通模型的道路图和日、周周期由腕部关节和短时运动建模取代。</li>
<li>阶段头输出停留、转向、平移的概率，监督标签取未来第 10 帧的运动模式。三个专家直接学习 30 步位移均值和正的逐步误差尺度。每个轨迹样本整段共享所选专家，标准化残差采用相关系数 0.85 的 AR(1) 模型。双种子等权集成得到六个高斯轨迹分量。发布的单条点轨迹由概率和尺度共同读出。</li>
<li>CAFHT 式外层用整条轨迹最严重的相对越界量校准三维球半径。CPTC 式内层用起点的阶段概率组合状态残差分位数，已发布的前一窗口全部到达且腕点观测有效时更新各状态的反馈水平。主配置训练得到的外层系数为 {spec['outer_factor']:.3f}。转向状态在训练参照的 {audit['gate_count']} 个窗口中仅有 {audit['warm_state_counts'][1]} 个达到候选状态概率阈值 0.30，因此按预设的 25 个样本下限使用合并残差库。阶段概率变化与分状态反馈仍参与签发。</li></ol>
<p>上述做法针对本题的 30 步一次性发布和三维欧氏距离作了适配。本文报告的是环境留出实测覆盖率。{refs}</p>
<h2>冻结后，在相同的 760 个测试窗口比较</h2>{main_table}
<p>ADE 是先对每个窗口的 30 步三维欧氏误差取平均，再对窗口等权平均。整段命中要求 30 步全部进入各自的球形范围。平均半径也是窗口和时间步等权。新主配置相对当前 QSM 在转向组新覆盖 {int(pairs.newly_covered.sum())} 个窗口，同时少覆盖 {int(pairs.newly_lost.sum())} 个。</p>
<p>等宽行用已发布半径作比例缩放，使两者的总体平均半径相等，只用于效率比较。相同操作在验证集的转向覆盖是 {count(vmatch.turn_covered,vmatch.turns)}。应使用验证时冻结的半径签发正式预测。主配置在验证集上的转向 ADE 为 {mval.turn_ade_cm:.2f} 厘米，整段覆盖为 {count(mval.covered,mval.windows)}，测试时转向误差优势没有延续。</p>
<h2>校准与阶段识别</h2>{bands}
<p>原始神经分布的逐步名义 95% 球在测试中的逐帧实测覆盖只有 {raw.frame_coverage:.2%}，存在明显低估。最终的整段覆盖依赖留出残差校准。用训练环境选定的转向报警阈值 {model['alert_threshold']:.3f}，测试识别出 {int(alarms.at['frozen_alarm','tp'])}/99 个转向，误报 {int(alarms.at['frozen_alarm','fp'])} 个。直接取概率最大的类别会识别出 {int(alarms.at['argmax','tp'])}/99 个。转向分类与转向组的整段覆盖分别计算。</p>
<h2>相同种子下的模块对照</h2>{abl}
<p>前四行使用相同随机种子和训练划分。短历史对照仅保留短分支。关闭质量门控的对照仍标记物理缺失节点。单高斯对照保留辅助阶段分类头，位移分布只有一个分量。能量分数用 48 对独立的联合轨迹样本估计，分布评分使用校准球扩张前的神经概率输出。</p>
<h2>主模型分环境成绩</h2>{env_table}
<p>数据共 69 条记录、24 个环境。10 个训练环境拟合模型，3 个训练环境选择训练轮次、概率温度并建立残差参照，另 3 个训练环境估计外层校准系数。4 个官方验证环境参与锁定点读出和校准方案。测试沿用此前已经报告的 4 个官方测试环境，最终 760 个参考窗口及 99 个转向标签均与原比较逐键核对。已用 {len(audit['raw_prefixes'])} 个原始观测前缀核对在线签发，最大半径差为 {audit['largest_online_radius_delta_m']:.3g} 米。</p>
<p><strong>版本选择。</strong>若第一问优先考虑转向误差与范围紧凑度，现有 QSM 低转向规则更符合这两个指标。若侧重总体点误差和偏向高覆盖的概率预测，LSTTN 可独立使用。在原第一问入口的记录、左右手和起点参数后加 <code>--predictor lsttn</code> 即可切换至联合校准模型。该入口默认选择 QSM，LSTTN 独立入口还支持 <code>--draws</code> 抽取联合轨迹。</p><p>{files}</p></html>"""
    (OUT/"第一问LSTTN替换实测.html").write_text(html,encoding="utf8")


if __name__=="__main__":
    report()
