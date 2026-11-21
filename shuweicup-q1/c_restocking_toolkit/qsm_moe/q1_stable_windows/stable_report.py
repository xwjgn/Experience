"""Chinese readout for the fixed-center stability experiment."""
from html import escape
from math import ceil
import os
from urllib.parse import quote

import pandas as pd

from stable_config import HERE, OUT
from config import read_json


def table(header,body):
    head = "".join("<th scope='col'>"+escape(str(v))+"</th>" for v in header)
    rows = "".join("<tr>"+"".join("<td>"+escape(str(v))+"</td>" for v in r)+"</tr>" for r in body)
    return f"<div class='scroll'><table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table></div>"


def count(x,n):
    return f"{int(x)}/{int(n)}（{x/n:.2%}）"


def link(path,label):
    url = quote(os.path.relpath(path,OUT).replace(os.sep,"/"),safe="/")
    return f'<a href="{url}">{escape(label)}</a>'


def report():
    results = pd.read_csv(OUT/"comparison.csv")
    test = results[results.split=="test"].set_index("model")
    val = results[results.split=="validation"].set_index("model")
    low,base = test.loc["phase_only"],test.loc["v3"]
    env = pd.read_csv(OUT/"by_environment.csv")
    env = env[(env.split=="test") & (env.model=="phase_only")]
    lowest = env.loc[env.coverage.idxmin()]
    need = max(0,ceil(.95*lowest.n)-int(lowest.covered))
    environment_note = f"低转向概率规则覆盖率最低的环境是 {escape(lowest.env)}，覆盖 {count(lowest.covered,lowest.n)}。"
    if need:
        environment_note += f"该环境还差 {need} 个窗口达到 95%。"
    log = pd.read_csv(OUT/"paired_windows.csv.gz")
    small = log[(log.split=="test") & (log.model=="phase_only") & log.routed]
    main_miss = log[(log.split=="test") & (log.model=="conservative") & log.newly_lost & log.turn]
    lock = read_json(OUT/"frozen_before_test.json")
    simple = read_json(OUT/"phase_rule.json")
    verified = read_json(OUT/"verification.json")
    cutoff = simple["maximum_turn_probability"]
    retain = simple["policy"]["factor"]
    checks = pd.read_csv(OUT/"held_out_validation.csv")
    folds = checks[checks.comparison=="phase_only"]
    methods = [("original","原方案"),("v3","当前 QSM v3"),("previous_transfer","上轮验证选中的反馈版"),
               ("previous_fixed_control","上轮固定反馈对照"),("conservative","本轮预设主模型"),
               ("phase_only","低转向概率规则"),("uniform_phase_same_mean","与低转向规则同宽，统一缩小"),
               ("recognition","运动学规则")]
    comparisons = table(["方案","转向 ADE（厘米）","总体整段覆盖","真实转向整段覆盖","平均半径（厘米）"],
              [[label,f"{m.turn_ade_cm:.2f}",count(m.covered,m.n),count(m.turn_covered,m.turns),f"{m.radius_cm:.2f}"]
               for key,label in methods for m in (test.loc[key],)])
    environments = table(["测试环境","收紧窗口","整段覆盖","新增漏失","新增漏失的转向","平均节省（厘米）"],
            [[e.env,int(e.routed),count(e.covered,e.n),int(e.lost),int(e.lost_turn),f"{e.mean_saving_cm:.2f}"]
             for e in env.itertuples()])
    missed = ""
    if len(main_miss):
        m = main_miss.iloc[0]
        missed = f"<p>预设主模型额外漏掉的真实转向出现在记录 <code>{escape(m['sample'])}</code> 的 {escape(m.hand)} 腕、第 {int(m.origin)} 帧。起点的 QSM 转向概率为 {m.turn_probability:.2%}。窗口最远偏差已占对应原半径的 {m.v3_worst_ratio:.2%}，收紧后占 {m.worst_ratio:.2%}。低转向概率规则在此起点保留了原半径。</p>"
    sources = " · ".join([link(OUT/"comparison.csv","完整对照"),link(OUT/"99个转向窗口逐窗核对.csv","99 个转向逐窗核对"),
            link(OUT/"收紧后新增漏失.csv","新增漏失明细"),link(OUT/"by_environment.csv","各环境结果"),
            link(OUT/"held_out_validation.csv","验证环境留一复查"),link(OUT/"frozen_before_test.json","测试前冻结参数"),
            link(OUT/"phase_rule.json","低转向规则阈值"),link(OUT/"verification.json","在线与离线一致性"),
            link(HERE/"run.py","独立重训入口"),link(HERE/"issue_stable.py","按原始观测签发预测")])
    html = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>第一问 · 预测起点的平稳判断与半径收紧</title><style>
body{{max-width:1060px;margin:42px auto;padding:0 22px 55px;font:16px/1.7 'Microsoft YaHei',sans-serif;color:#1d303b}}
h1,h2{{line-height:1.35}}h2{{margin:1.8em 0 .5em;font-size:21px}}a{{color:#07556a}}.scroll{{overflow:auto}}
table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:8px 10px;border-bottom:1px solid #cfdbdf;text-align:right}}
th{{background:#eaf1f3}}th:first-child,td:first-child{{text-align:left}}li{{margin:.4em 0}}code{{overflow-wrap:anywhere}}
</style><h1>第一问 · 预测起点的平稳判断与半径收紧</h1>
<p><strong>低转向概率规则把平均半径从 {base.radius_cm:.2f} 厘米降到 {low.radius_cm:.2f} 厘米，原先覆盖的 {int(base.turn_covered)} 个真实转向全部保住。</strong>
总体整段覆盖为 {count(low.covered,low.n)}，新增的 {int(low.lost)} 个漏失均发生在非转向窗口。每个预测中心和当前 QSM v3 逐点一致，99 个真实转向的平均轨迹误差为 {low.turn_ade_cm:.2f} 厘米。</p>
<h2>预测起点怎样作决定</h2>
<ol><li>读取 QSM 对未来第 10 帧转向的概率。阈值约为 {cutoff:.2%}，由 3 个训练环境的起点评分分布确定。</li>
<li>当前腕点必须有效，最近 10 帧至少 80% 的腕点观测有效。若最近 5 帧平均速度超过 0.12 米/秒，且与再往前 5 帧的速度方向夹角大于 60°，则保留原半径。</li>
<li>只查看已经到达的历史窗口。如果最近一次得到完整有效观测的预测在过去 60 帧内结束，且有一步越界，也保留原半径。尚无完整历史时，使用当前概率和已观测运动判定。</li>
<li>满足收紧条件时，把该窗口每一步的原半径乘以 {retain:.2f}。其余窗口使用原半径。新观测到达后，可调整下一次签发的半径。</li></ol>
<p>760 个窗口中有 {int(low.routed)} 个触发收紧。该组原平均半径 {small.v3_radius_cm.mean():.2f} 厘米，收紧后为 {small.radius_cm.mean():.2f} 厘米，{count(small.hit.sum(),len(small))} 的窗口仍被整段覆盖。触发组中有 {int(low.routed_turns)} 个真实转向，完整覆盖 {int(low.routed_turn_covered)} 个。</p>
<p>按真实运动分组，{int(low.n-low.turns)} 个非转向窗口的平均半径从 {base.nonturn_radius_cm:.2f} 降到 {low.nonturn_radius_cm:.2f} 厘米。{int(low.turns)} 个真实转向窗口的平均半径从 {base.turn_radius_cm:.2f} 降到 {low.turn_radius_cm:.2f} 厘米。</p>
<h2>同一批测试窗口的对照</h2>{comparisons}
<p>同宽对照把全部 QSM 半径统一按比例调整，使测试集平均半径与低转向规则相同，只用于回顾性比较。两者平均半径均为 {low.radius_cm:.2f} 厘米，低转向规则多完整覆盖 {int(low.covered-test.at['uniform_phase_same_mean','covered'])} 个窗口，其中 {int(low.turn_covered-test.at['uniform_phase_same_mean','turn_covered'])} 个是真实转向。</p>
<h2>识别平稳的能力</h2>
<p>用实际发生的 30 步运动单独评价“平稳”。腕点始终处于起点 3 厘米内，或三个连续 10 帧速度段同时满足相邻夹角不超过 30°、速度变异系数不超过 0.35、净位移与分段路程之比至少 0.93，即判为平稳。速度用原始实测时间间隔计算。这是本次实验的运动学定义。99 个真实转向仍采用历次实验统一的第 10 帧方向变化标签。</p>
<p>全体有 {int(low.actual_stable)}/{int(low.n)}（{low.actual_stable/low.n:.2%}）个平稳窗口。在低转向概率规则筛出的窗口中有 {int(low.routed_stable)}/{int(low.routed)}（{low.stable_precision:.2%}）个实际平稳，覆盖了所有平稳窗口的 {low.stable_recall:.2%}。另有 {int(low.routed-low.routed_stable)} 个被收紧的窗口未满足上述严格定义，这一区别影响对“平稳识别率”的解读。</p>
{missed}
<h2>各环境的代价</h2>{environments}
<p>如果要求四个测试环境各自达到 95%，运动学规则可作为另一选择。它将平均半径降到 {test.at['recognition','radius_cm']:.2f} 厘米，{int(test.at['recognition','covered'])} 个原有整段命中全部保留。{environment_note}</p>
<h2>训练、选择与复查</h2>
<p>从原始数据和随机初始化重新训练 QSM，以它的点轨迹和原半径作配对参照。10 个训练环境共 {lock['training_rows']} 个有效起点用于训练平稳、未来转向、接近区间边缘的转向风险以及轨迹误差上分位数。三个训练环境用于确定起点评分阈值，另三个训练环境及四个官方验证环境用于筛选规则。真实未来标签只在训练与评分时使用。缩半径的在线入口默认使用低转向概率规则，运行时直接读取 QSM 的转向概率，无需加载新增的平稳分类器。<code>--policy recognition</code> 签发运动学规则，<code>--policy conservative</code> 重现预设主模型。</p>
<p>主模型、低转向概率对照、运动学对照及其阈值均在本次测试评分前冻结。低转向规则在验证集覆盖 {count(val.at['phase_only','covered'],val.at['phase_only','n'])}，转向覆盖 {count(val.at['phase_only','turn_covered'],val.at['phase_only','turns'])}。对该对照逐一拿掉一个验证环境再选规则，在被拿掉的四个环境合计覆盖 {count(folds.covered.sum(),folds.n.sum())}，转向覆盖 {count(folds.turn_covered.sum(),folds.turns.sum())}。此留一复查只改变半径规则的选择，沿用同一 QSM 预测中心。</p>
<p>低转向概率规则成为本轮更窄的候选，是比较上述冻结版本后作出的决定。官方测试仍是此前研究过的 4 个环境。精简规则在 {verified['compact_rule_matches_full_router_on']} 个验证及测试窗口与完整实现逐个一致，并在 {len(verified['raw_prefixes'])} 个原始数据前缀核对了实际签发结果。</p>
<p>{sources}</p></html>"""
    (OUT/"第一问起点平稳与半径收紧.html").write_text(html,encoding="utf8")


if __name__=="__main__":
    report()
