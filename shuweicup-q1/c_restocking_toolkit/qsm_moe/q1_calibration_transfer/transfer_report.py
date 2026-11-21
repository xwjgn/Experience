"""Chinese report of the original Q1 calibration and the fixed-center transfer results."""
from html import escape
import os
from urllib.parse import quote
import pandas as pd

from transfer_config import OUT, HERE, KIT, ORIGINAL
from config import read_json

def link(path, label):
    return '<a href="'+quote(os.path.relpath(path,OUT).replace(os.sep,"/"),safe="/")+'">'+escape(label)+'</a>'

def table(headers, rows):
    heading = ''.join('<th scope="col">'+escape(str(s))+'</th>' for s in headers)
    body = ''.join('<tr>'+''.join('<td>'+escape(str(v))+'</td>' for v in row)+'</tr>' for row in rows)
    return '<table><thead><tr>'+heading+'</tr></thead><tbody>'+body+'</tbody></table>'

def count(hit, total):
    return f"{int(hit)}/{int(total)}（{hit/total:.2%}）"

def report():
    out = pd.read_csv(OUT/"comparison.csv")
    test = out[out.split=="test"].set_index("model")
    old = pd.read_csv(OUT/"original_band_audit.csv").query("split=='test'").set_index("band")
    tail = pd.read_csv(OUT/"paired_tail_diagnosis.csv").query("split=='test'").set_index(["group","model"])
    feedback = pd.read_csv(OUT/"feedback_diagnosis.csv").set_index(["split","group"])
    val = out[out.split=="validation"].set_index("model")
    lock = read_json(OUT/"frozen_before_test.json")
    fold = read_json(OUT/"verification.json")
    chk = read_json(OUT/"paired_audit.json")
    def point_row(code,name):
        r = test.loc[code]
        return [name,f"{r.turn_ade_cm:.2f}",count(r.covered,r.windows),
                count(r.turn_covered,r.turns),f"{r.mean_radius_cm:.2f}",f"{r.turn_radius_cm:.2f}"]
    main = table(["方案","99 个转向窗口 ADE（厘米）","760 个窗口整段覆盖","转向组整段覆盖","总体平均半径（厘米）","转向组平均半径（厘米）"],
          [point_row("original","原方案"),point_row("current","当前 QSM v3"),
           point_row("gru_endpoint_95","重新训练 GRU 尺度并采用原反馈规则"),
           point_row("selected","按验证集选出的迁移版")])
    original = table(["原方案的半径设置","整段覆盖","逐帧覆盖","平均半径（厘米）"],[
              [name,count(old.at[key,"covered"],old.at[key,"n"]),f"{old.at[key,'frame_coverage']:.2%}",f"{old.at[key,'mean_radius_cm']:.2f}"]
              for key,name in (("flat","整段用一个半径"),("pointwise","各步分别校准"),("joint_static","按最大误差整段校准"),
                               ("joint_adaptive","整段校准和延迟终点反馈"))])
    tails = table(["窗口","原方案平均半径","QSM v3 平均半径","原方案窗内最大误差的第 95 百分位","QSM v3 同一统计量"],[
           [label+f"（{int(tail.loc[(group,'current'),'windows'])}）",
            *[f"{tail.loc[(group,model),col]:.2f}" for col in ("mean_radius_cm","p95_max_error_cm") for model in ("original","current")]]
           for group,label in (("all","全部"),("turn","真实转向"),("nonturn","其余窗口"))])
    audit = pd.read_csv(OUT/"paired_changes.csv").query("split=='test' and model=='selected'").iloc[0]
    sensitivity = test.loc["gru_val_calibration"]
    original_sources = ' · '.join([link(KIT/"literature_experiments"/"q1_gru_path_calibration.py","原方案轨迹、半径和校准源码"),
                       link(ORIGINAL/"protocol.json","原方案训练与校准协议"),link(ORIGINAL/"band_coverage.csv","原方案逐项实验")])
    resources = ' · '.join([link(OUT/"comparison.csv","完整方案对照"),link(OUT/"by_environment.csv","分环境结果"),
                link(OUT/"feedback_diagnosis.csv","关闭反馈的配对诊断"),link(OUT/"paired_tail_diagnosis.csv","误差尾部与半径"),
                link(OUT/"frozen_before_test.json","测试前冻结的选择"),link(OUT/"verification.json","逐前缀核验"),
                link(HERE/"run.py","独立重训与复现实验入口"),link(HERE/"issue_band.py","按原始观测签发预测")])
    feedback_details = ""
    if ("test","all") in feedback.index:
        v,t,tr = (feedback.loc[key] for key in (("validation","all"),("test","all"),("test","turn")))
        feedback_details = f"""<h2>同一迁移模型，终点反馈付出了多少半径</h2>
<p>事后只把反馈分位数固定在初值，保留相同的中心、分步尺度、整段系数、时间步权重和半径下限。测试平均半径由 {t.with_feedback_cm:.2f} 降至 {t.without_feedback_cm:.2f} 厘米。整体覆盖由 {count(t.with_feedback_covered,t.windows)} 变为 {count(t.without_feedback_covered,t.windows)}，转向组由 {count(tr.with_feedback_covered,tr.windows)} 变为 {count(tr.without_feedback_covered,tr.windows)}。两次签发的点轨迹相同。</p>
<p>转向组覆盖是单列的诊断指标。如果考核只要求总体覆盖达标，这个设置在当前测试集恰好达到 95%，可用于后续独立评估。测试中的反馈参数低于初值的窗口占 {t.alpha_below_initial:.1%}，验证时占 {v.alpha_below_initial:.1%}。参数越低，模型取的误差分位数越高，同样的轨迹就会收到更宽的范围。</p>"""
    html = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>第一问 · 原方案校准方法迁移对照</title><style>
body{{max-width:1020px;margin:42px auto;padding:0 24px 56px;font:16px/1.7 'Microsoft YaHei',sans-serif;color:#192e36}}
h1,h2{{line-height:1.35}}h2{{margin-top:2em;font-size:21px}}a{{color:#07566a}}table{{border-collapse:collapse;width:100%;font-size:14px;margin:15px 0 12px}}td,th{{padding:8px 10px;border-bottom:1px solid #cbd5d8;text-align:right}}th:first-child,td:first-child{{text-align:left}}th{{background:#eaf1f2}}li{{margin-bottom:6px}}
@media(max-width:680px){{body{{padding:0 12px}}td,th{{padding:5px 3px}}table{{font-size:12px}}}}</style>
<h1>第一问 · 原方案校准方法迁移对照</h1>
<p>对于已经超过 95% 整段覆盖的方案，半径越小，给出的腕点范围越精确。原方案与当前 QSM 的全体平均半径相差 {test.at['current','mean_radius_cm']-test.at['original','mean_radius_cm']:.2f} 厘米。这次独立重训 QSM，逐点固定其预测中心，单独比较半径机制。</p>
<p><strong>本次建议保留当前 QSM v3 作为主结果。</strong>按验证集选出的迁移版在测试中多覆盖了部分转向窗口，代价是总体半径增大。若只考核全体 95% 覆盖和转向 ADE，固定反馈分位数的配置提供了更窄的候选，仍需在独立环境检验。</p>
<h2>原方案采用的三个措施</h2><ol>
<li>保留已有的一秒腕点预测，用时间常数 0.45 秒的速度衰减外推生成中间轨迹，再按预测步数逐渐校正，使第 30 步落在该终点。</li>
<li>三个独立初始化的轨迹 GRU 同时学习误差尺度。每个模型根据约两秒的历史运动和最近 11 帧的密集腕点数据，输出四段正增长率，形成随预测时间变化的基础半径。训练集再为各预测步分别估计标准化误差分位数。</li>
<li>按记录和左右手分别更新一个反馈参数。每隔 30 帧，上一预测的终点到达且腕点观测质量合格时才更新。上次越界会放宽半径，连续命中则逐步收紧。外层校准取 30 步标准化误差中的最大值。预测发布后，后续观测只影响下一次发布的半径。</li></ol>
{original}<p>{original_sources}</p>
<h2>同一批 760 个完整测试窗口</h2>{main}
<p>ADE 为窗口中 30 个三维欧氏位置误差的平均值，并在真实转向的 99 个窗口上取平均。真实转向沿用 QSM 的分组标准，取预测起点后第 10 帧的运动模式标签。整段覆盖要求全部 30 步均落在各自半径内。平均半径对所有纳入窗口和各自的 30 步等权。窗口按记录和左右手分别从第 10 帧开始，每隔 30 帧取一次。</p>
<p>当前版在验证集覆盖 {count(val.at['current','covered'],val.at['current','windows'])}，平均半径 {val.at['current','mean_radius_cm']:.2f} 厘米。迁移版覆盖 {count(val.at['selected','covered'],val.at['selected','windows'])}，平均半径 {val.at['selected','mean_radius_cm']:.2f} 厘米。原有验证门槛兼顾总体、每个环境和转向组。若额外要求总体及转向覆盖数均追平当前版，验证集仍选中了当前版。测试中迁移版相对当前版新覆盖 {int(audit.gained)} 个窗口，同时少覆盖 {int(audit.lost)} 个，转向组分别为 {int(audit.turn_gained)} 个和 {int(audit.turn_lost)} 个。</p>
{feedback_details}
<h2>总体半径差在哪里</h2>{tails}
<p>表中长度均为厘米。每个窗口先取 30 步三维误差的最大值，再对同组窗口取第 95 百分位。转向窗口的平均半径十分接近。其余 661 个窗口的平均半径相差 {tail.loc[('nonturn','current'),'mean_radius_cm']-tail.loc[('nonturn','original'),'mean_radius_cm']:.2f} 厘米。从窗口最远偏差的分布看，需要优先检查半径如何分配给这些窗口。</p>
<h2>训练与可比性</h2>
<p>所有 QSM 权重在新的目录从随机初始化开始训练。预先固定的 16 个官方训练环境分为 10 个拟合、3 个估计误差分布、3 个拟合整段系数。三个 GRU 尺度头只使用 10 个拟合环境，以当前 QSM 点误差作监督。4 个官方验证环境用于选择方案。冻结后的主选取 QSM 的运动误差尺度、分预测步的权重、终点反馈和当前半径的 90% 下限。对每种半径方案，转向标签只用于分组评价。</p>
<p>原方案另外使用 4 个验证环境拟合最终校准系数，报告验证结果时逐环境留一。迁移的敏感性对照也使用这一口径，测试总体覆盖 {count(sensitivity.covered,sensitivity.windows)}，转向组 {count(sensitivity.turn_covered,sensitivity.turns)}，平均半径 {sensitivity.mean_radius_cm:.2f} 厘米。主比较保持 QSM 原有的数据划分。</p>
<p>重新训练的中心与存档在所有验证和测试窗口逐点一致，最大坐标差 {max(z['max_center_difference_m'] for z in chk['center_checks']):.1f} 米。在 {len(fold['prefix_issues'])} 个环境核验了只读取已到达帧的在线签发。原函数与迁移实现的反馈半径在一条连续记录的 {fold['original_feedback_match']['num_issues']} 次签发中最大差异为 {fold['original_feedback_match']['original_function_max_difference_m']:.2g} 米。本次候选先冻结再评分。测试沿用此前已报告的 4 个环境，新环境表现需要另外验证。</p>
<p>在线入口默认采用验证覆盖数追平当前版的设置。参数 <code>--decision selected</code> 签发迁移版，<code>--decision fixed_feedback_control</code> 签发固定反馈分位数的诊断版本。</p>
<p>{resources}</p></html>"""
    (OUT/"第一问原方案校准迁移对照.html").write_text(html,encoding="utf8")

if __name__=="__main__":
    report()
