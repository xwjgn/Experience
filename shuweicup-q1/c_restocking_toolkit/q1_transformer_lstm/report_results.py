"""Build the Chinese study report from the frozen comparison and online checks."""
from html import escape

import pandas as pd

from study_paths import OUT, read


def table(headers, rows):
    head = "".join(f"<th>{escape(x)}</th>" for x in headers)
    body = "".join("<tr>"+"".join(f"<td>{escape(str(x))}</td>" for x in row)+"</tr>" for row in rows)
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table></div>"


def pct(x):
    return f"{100*x:.2f}%"


def report():
    metrics = pd.read_csv(OUT/"comparison.csv").set_index(["split","model","band"])
    quality = pd.read_csv(OUT/"quality_all_handframes.csv").set_index(["model","split","environment"])
    alarms = pd.read_csv(OUT/"stage_scores.csv").set_index(["model","split","rule"])
    counts = pd.read_csv(OUT/"paired_issues.csv.gz")
    extra = pd.read_csv(OUT/"natural_bad_observations.csv").set_index(["model","split","subset","horizon"])
    proper = pd.read_csv(OUT/"distribution_scores.csv").set_index(["model","split"])
    horizons = pd.read_csv(OUT/"by_horizon.csv").set_index(["model","split","horizon"])
    audit = read(OUT/"online_parity.json")
    runtime = read(OUT/"runtime.json")
    provenance = read(OUT/"run_file_reads.json")
    protocol = read(OUT/"release_protocol.json")
    lock = read(OUT/"frozen_before_test.json")
    models = [
        ("已有质量感知 Kalman＋速度衰减外推（M0）","M0","archived_quality_kalman_damped"),
        ("原方案的腕点轨迹与整段校准","published_original","archived"),
        ("已有 QSM-MoE 起点收紧版","published_qsm_low_turn","archived"),
        ("上一版独立质量感知 LSTTN","published_quality_stage","cafht_cptc"),
        ("M1　普通 LSTM＋整段校准","M1","joint_stage"),
        ("M2　质量感知 Transformer-LSTM＋整段校准","M2","joint_stage"),
        ("M3　加入多尺度运动历史，逐步 95% 原始半径","M3","raw_step95"),
        ("M4　M3＋阶段自适应整段校准","M3","joint_stage"),
    ]
    def main_table(split):
        rows = []
        for label, model, band in models:
            m = metrics.loc[(split,model,band)]
            hit = "—" if pd.isna(m.covered) else f"{int(m.covered)}/{int(m.n)}（{pct(m.coverage)}）"
            turn = "—" if pd.isna(m.turn_covered) else f"{int(m.turn_covered)}/{int(m.turns)}"
            radius = "—" if pd.isna(m.radius_cm) else f"{m.radius_cm:.2f}"
            rows.append((label,f"{m.ade_cm:.2f}",f"{m.turn_ade_cm:.2f}",hit,turn,radius))
        return table(("模型","总体 ADE（cm）","真实转向 ADE（cm）","整段覆盖","转向整段覆盖","平均半径（cm）"),rows)
    qrows, arows, ablate = [], [], []
    for kind in ("M1","M2","M2_uniform","M3"):
        q = quality.loc[(kind,"test","all")]
        a = alarms.loc[(kind,"test","gate_threshold")]
        choice = alarms.loc[(kind,"test","argmax")]
        model_name = "等权关节的 M2" if kind == "M2_uniform" else kind
        qrows.append((model_name,f"{q.average_precision:.4f}",f"{q.auc:.3f}",f"{int(q.tp)}/{int(q.positive)}",
                      f"{int(q.fp):,}",pct(q.precision)))
        arows.append((model_name,f"{int(a.tp)}/{int(a.turns)}",int(a.fp),pct(a.precision),f"{int(choice.tp)}/{int(choice.turns)}"))
        m = metrics.loc[("test",kind,"joint_stage")]
        ps = proper.loc[(kind,"test")]
        ablate.append((model_name,f"{m.ade_cm:.2f}",f"{m.turn_ade_cm:.2f}",f"{ps.energy_score_cm:.2f}",f"{ps.turn_energy_score_cm:.2f}"))
    group_rows = []
    m4 = counts[(counts.split=="test") & (counts.model=="M3")]
    for enabled,group in m4.groupby("turn_alert"):
        turns = group[group.turn]
        group_rows.append(("发出转向预警" if enabled else "未触发转向预警",len(group),len(turns),
                           f"{group.radius_cm.mean():.2f}",f"{int(group.hit.sum())}/{len(group)}",f"{int(turns.hit.sum())}/{len(turns)}"))
    step_rows = []
    for h in (1,3,10,30):
        row = horizons.loc[("M3","test",h)]
        step_rows.append((h,f"{row.median_seconds:.3f}",f"{row.mean_error_cm:.2f}",f"{row.turn_mean_error_cm:.2f}",
                          f"{int(row.covered)}/{int(row.n)}",f"{row.mean_radius_cm:.2f}"))
    observed = [extra.loc[(m,"test","at_origin",30)] for m in ("M1","M2","M3")]
    assert all(int(r.n)==int(observed[0].n) for r in observed)
    reads = provenance["random_fit"]
    assert reads["new_checkpoint_reads"] == reads["archived_result_reads"] == []
    assert len(provenance["score"]["archived_result_reads"]) == 3
    family = metrics.loc[("test","M3","cafht")]
    same = metrics.loc[("test","M3","equal_QSM_width_diagnostic")]
    direct = metrics.loc[("test","M3","joint_stage")]
    old = metrics.loc[("test","published_qsm_low_turn","archived")]
    plain = quality.loc[("M3","test","all")]
    ng = lock["partitions"]
    html = f"""<!DOCTYPE html><html lang="zh-CN"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>第一问 · 独立质量感知 Transformer-LSTM 实测</title>
<style>
*{{box-sizing:border-box}}body{{font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif;color:#172a35;background:#f3f6f7;margin:0}}
main{{max-width:1080px;margin:32px auto;background:white;padding:34px 46px 44px;border:1px solid #d7e3e5;border-radius:10px}}
h1{{font-size:clamp(25px,4vw,38px);line-height:1.3;margin:0 0 14px}}h2{{font-size:22px;line-height:1.45;margin:32px 0 8px}}p{{margin:8px 0 14px}}
a{{color:#075e78;text-underline-offset:3px}}.lead{{font-size:18px}}.scroll{{overflow-x:auto}}table{{width:100%;font-size:14px;border-collapse:collapse;margin:8px 0 12px;font-variant-numeric:tabular-nums}}
th,td{{padding:8px;border-bottom:1px solid #dbe5e7;text-align:right;vertical-align:top;white-space:nowrap}}th{{color:#294853;background:#eaf2f3}}
th:first-child,td:first-child{{text-align:left;white-space:normal;min-width:160px}}.note{{color:#405965;font-size:14px}}code{{font-family:Consolas,monospace;overflow-wrap:anywhere}}
@media(max-width:700px){{main{{margin:0;padding:18px 12px;min-height:100vh;border:0}}}}@media print{{body{{background:white}}main{{border:0;margin:0;padding:0}}table{{font-size:11px}}}}
</style><main><h1>第一问 · 独立质量感知 Transformer-LSTM</h1>
<p class="lead">在相同的 760 个测试窗口上，多尺度模型 M4 的平均轨迹误差为 <b>{direct.ade_cm:.2f} 厘米</b>，其中 99 个真实转向窗口为 <b>{direct.turn_ade_cm:.2f} 厘米</b>。整段覆盖 {int(direct.covered)}/760，转向覆盖 {int(direct.turn_covered)}/99，平均半径 {direct.radius_cm:.2f} 厘米。</p>
<p>已有质量感知 Kalman 按观测可靠性调整测量噪声，本次直接引用其速度衰减外推成绩。独立模型从赛题发布的三份原始压缩包起步，每个模型结构使用相同的两个随机种子，各自优化全部参数。模型输入为双手关节、头部姿态、时间和起点可用的观测质量标记。本实验检验关节时序和质量信息的作用，视觉视频未参与训练与评分。 <a href="../../c_restocking_toolkit/qsm_moe/qsm_data.py">已有 Kalman 实现</a> · <a href="../第一问质量感知LSTTN与阶段概率预测/current_kalman_anchor.csv">M0 对照原表</a></p>
<h2>相同起点上的测试结果</h2>
{main_table("test")}
<p class="note">每个窗口从同一只手的预测起点一次发出未来 30 帧。ADE 先计算每一步的三维欧氏误差，再平均 30 步及各窗口。整段覆盖要求 30 个真实腕点全部落入各自球形范围。转向由起点后第 10 帧的回顾性运动方向及速度定义，预测时只能读取起点及之前的观测。M0 档案保存了匹配的点误差，区间栏留空。</p>
<h2>同一预测中心，半径改变了多少</h2>
<p>M3 原始逐步 95% 半径只覆盖 474/760 个完整窗口。只加入 CAFHT 式整段校准，平均半径为 {family.radius_cm:.2f} 厘米，整段覆盖 {int(family.covered)}/760，转向覆盖 {int(family.turn_covered)}/99。M4 又按当前阶段概率调整误差尺度，并用已经完整到达且质量正常的历史窗口反馈，整段覆盖达到 {pct(direct.coverage)}。M3 与 M4 使用同一份点预测。</p>
<p>把 M4 的平均半径事后统一到已有 QSM 的 {old.radius_cm:.2f} 厘米，M4 的覆盖为 <b>{int(same.covered)}/760（{pct(same.coverage)}）</b>，QSM 为 <b>{int(old.covered)}/760（{pct(old.coverage)}）</b>，两者都覆盖 95/99 个转向。这个对照使用整批测试窗口的平均宽度，正式在线输出使用验证阶段冻结的规则。</p>
<h2>起点能否提前识别转向</h2>
{table(("模型","提前识别转向","误报窗口","预警精确率","直接选最高概率类别时识别"),arows)}
<p>模型输出停留、转向和移动三个概率。直接选概率最高的类别时，M1 漏掉全部真实转向，M2 和 M3 各识别了 1 个。用训练环境上选定的转向概率门槛，M3 提前识别 52 个，同时误报 129 个。有 47 个转向落在未预警组。即使没有提前报警，误差范围仍可能覆盖完整转向。</p>
{table(("M4 起点判定","窗口数","其中真实转向","平均半径（cm）","整段覆盖","真实转向覆盖"),group_rows)}
<p>未触发预警组中仍有 47 个真实转向，其平均半径为 37.88 厘米。发出预警组的平均半径为 37.20 厘米。这个版本尚不能单凭起点的转向预警，把平稳窗口可靠地筛出来并缩小半径。</p>
<h2>未来 1、3、10、30 帧的腕点误差</h2>
{table(("预测步数","实测时间中位数（秒）","该步误差（cm）","真实转向该步误差（cm）","该步覆盖","该步平均半径（cm）"), step_rows)}
<p class="note">这组数仅取指定步的腕点。每个预测起点的时间输入使用原始时间间隔，测试窗口从起点到第 30 帧约为 1 秒。 <a href="by_horizon.csv">所有新模型按步长的实测结果</a></p>
<h2>关节注意力、质量标记与运动历史</h2>
{table(("模型","总体 ADE（cm）","转向 ADE（cm）","轨迹能量评分（cm）","转向能量评分（cm）"),ablate)}
<p>能量评分用固定随机种子从预测分布采样，分值越低越好。M2 把可见关节由等权汇总改成腕点查询的三头注意力，总体 ADE 改善，转向 ADE 略升。M3 在 M2 上增加短历史和稀疏长历史融合，给出了本组实验中最低的总体与转向 ADE。</p>
{table(("模型","质量标记平均精度 AP","ROC AUC","检出质量异常","误报手帧","质量预警精确率"),qrows)}
<p>质量目标是赛题记录中的异常标记。全部 {int(plain.n):,} 个测试手帧中有 {int(plain.positive)} 个标为异常，占 {pct(plain.prevalence)}。质量分类头仅看运动、时间与可见性，当前观测质量标记只供轨迹预测头使用。M3 的质量预警精确率为 {pct(plain.precision)}，单凭这个报警做观测决策会产生大量误报。</p>
<p>在另一个补充集合中，起点已有质量异常且第 30 帧有合格参考腕点，共 {int(observed[0].n)} 个可重叠起点。M1、M2、M3 的第 30 帧误差依次为 {observed[0].error_cm:.2f}、{observed[1].error_cm:.2f}、{observed[2].error_cm:.2f} 厘米。该处统计终点误差，主表统计不重叠窗口的 30 步平均误差。</p>
<h2>方案与论文的对应关系</h2>
<p>M2 借鉴卫星观测质量论文的时间记忆、跨观测通道注意力和质量输出。对本题，每只手按腕部与五指构造六条几何序列，由腕点查询在当前时刻可见的其他通道。针对赛题新增三阶段、30 步混合轨迹预测头。作者材料快照含说明与定位结果图，网络在本目录独立实现。 <a href="https://doi.org/10.1109/ITSC57777.2023.10422672">GNSS 论文</a> · <a href="https://github.com/rwth-irt/DeepNLOSDetection">作者材料</a></p>
<p>M3 借鉴 LSTTN 的长短历史与膨胀卷积。短分支看最近 18 帧，长分支按 4 帧间隔抽取最多 20 个时刻并用实际时间差编码。每个种子产生三条阶段条件均值轨迹及正值尺度，两个种子共同形成六成分分布。每个窗口共享同一采样阶段，轨迹内的时间相关性用训练环境中的误差估计。 <a href="https://doi.org/10.1016/j.knosys.2024.111637">LSTTN 论文</a> · <a href="https://github.com/GeoX-Lab/LSTTN/blob/68185aa8ffd266d4d1a3acffeb6875a144335554/models/lsttn.py">作者实现</a></p>
<p>M4 借鉴 CAFHT 的整条路径校准，通过窗口内最大归一化距离做外层校准。借鉴 CPTC 的阶段变化适应，以预测的运动阶段概率调整参考分布。两者均针对三维腕点、质量标记和延迟反馈改写。这些覆盖率是四个留出测试环境的实测频率，新环境上的可靠性仍需独立数据检验。 <a href="https://proceedings.mlr.press/v235/zhou24l.html">CAFHT</a> · <a href="https://proceedings.neurips.cc/paper_files/paper/2025/hash/12271b64c483ad8f6192eb6aaa102044-Abstract-Conference.html">CPTC</a></p>
<h2>环境划分与独立性核验</h2>
<p>{protocol['records']} 条记录来自 {protocol['environments']} 个环境。官方训练环境内分出 10 个环境拟合网络、3 个选训练轮次与预警阈值、3 个做外层校准（{ng['cal']} 个完整窗口）。随后在 4 个验证环境（{ng['validation']} 个完整窗口）确定点预测读出和校准设置，冻结后在另外 4 个测试环境评分。既往方案也使用过这些测试环境，因此模型之间的差异属于同一测试集上的回顾比较。</p>
{main_table("validation")}
<p>两次从随机种子独立初始化的训练分别保留参数指纹。文件读取审计在训练阶段记录的模型权重读取数为 0。历史对照仅在评分时读取三张已发布成绩表。在线回放覆盖三种数据来源、左右手及 8 个验证和测试环境，共 {audit['prefix_comparisons']} 次。在线坐标转换、30 步预测分布和半径与存档最大绝对差低于 10⁻⁵。仅改写当前质量标记时，几何质量分类结果在全部核验窗口一致。</p>
<p>完整复现使用新目录的 <code>python run.py</code>。本次运行环境使用 Python {runtime['python']} 与 PyTorch {runtime['torch']}。预测示例为 <code>python forecast.py E018 right 70 --model M4 --draws 3</code>，输出世界坐标米、逐步半径、阶段概率及可复现轨迹样本。 <a href="../../c_restocking_toolkit/q1_transformer_lstm/run.py">复现入口</a> · <a href="../../c_restocking_toolkit/q1_transformer_lstm/forecast.py">在线预测入口</a> · <a href="runtime.json">运行环境</a></p>
<p><a href="comparison.csv">各组完整结果</a> · <a href="99个真实转向逐窗对照.csv">99 个转向逐窗结果</a> · <a href="by_environment.csv">按环境分解</a> · <a href="quality_all_handframes.csv">各环境质量分类</a> · <a href="frozen_before_test.json">冻结设置</a> · <a href="run_file_reads.json">各阶段数据来源</a> · <a href="online_parity.json">在线回放核验</a></p>
</main></html>"""
    path = OUT/"第一问Transformer-LSTM独立实测.html"
    path.write_text(html, encoding="utf-8")
    print(path)


if __name__ == "__main__":
    report()
