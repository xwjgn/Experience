"""Chinese comparison of the four established options and the independently fitted Bayesian module."""
import numpy as np
import pandas as pd
from gain_records import OUT,PREVIOUS,read


def table(labels,rows):
    return '<div class="scroll">'+pd.DataFrame(rows,columns=labels).to_html(index=False,border=0)+'</div>'


def report():
    a=pd.read_csv(OUT/"sphere_comparison.csv").set_index(["model","rule"])
    historic=pd.read_csv(PREVIOUS/"intervals.csv").set_index("model")
    diagnostics=pd.read_csv(OUT/"covariance_diagnostics.csv").set_index("model")
    ellipses=pd.read_csv(OUT/"ellipsoid_coverage.csv").set_index(["model","center"])
    paired=pd.read_csv(OUT/"760个起点配对.csv")
    locked=read(OUT/"frozen_before_test.json")
    check=read(OUT/"raw_prefix_checks.json")
    result=a.loc[("bayesian","matched")]
    baseline=a.loc[("third","matched")]
    four=a.loc[("fourth","archived")]
    transfer=a.loc[("bayesian","transfer_fourth_width")]
    stand=a.loc[("bayesian","standalone_sphere")]
    bayes=locked["models"]["bayesian"]["hybrid"]
    def row(label,x):
        return (label,f"{x.ade_cm:.2f}",f"{x.turn_ade_cm:.2f}",f"{x.radius_cm:.2f}",
                f"{int(x.covered)}/760（{100*x.coverage:.2f}%）",f"{int(x.turn_covered)}/99",f"{100*x.least_env:.2f}%")
    main=[row(label,historic.loc[k]) for k,label in (("m4","第一行 · 原 M4"),("previous_low_turn","第二行 · 低风险窗口收紧"),
             ("mixed_center_same_window_budget","第三行 · 调整中心，逐窗保持平均半径"),("mixed_center_compact","第四行 · 调整中心，进一步收紧"))]
    main.extend((row("第三行＋贝叶斯协方差分配半径",result),row("第四行＋相同的分配规则",transfer)))
    headers=["方案","总体误差（cm）","转向误差（cm）","平均半径（cm）","整段覆盖","转向整段覆盖","最低环境覆盖"]
    tests=[]
    for k,label in (("constant","全局常数增益"),("deterministic","随观测变化的确定性神经增益"),("bayesian","贝叶斯神经增益"),
                    ("posterior_mean_weights","贝叶斯模型只用后验均值权重")):
        m=a.loc[(k,"standalone_sphere")]
        chosen=a.loc[(k,"matched")]
        same=a.loc[(k,"same_profile_power_cov_only")]
        tests.append((label,f"{m.ade_cm:.2f} / {m.turn_ade_cm:.2f}",f"{int(m.covered)}/760",f"{m.radius_cm:.2f}",
                      f"{int(chosen.covered)}/760",f"{int(same.covered)}/760"))
    e=ellipses.loc[("bayesian","hybrid")]
    base_old=historic.loc["previous_low_turn"]
    cases=int((paired.hybrid_hit & ~paired.third_hit).sum())
    lost=int((~paired.hybrid_hit & paired.third_hit).sum())
    plus4=int((paired.fourth_covariance_hit & ~paired.fourth_hit).sum())
    minus4=int((~paired.fourth_covariance_hit & paired.fourth_hit).sum())
    with np.load(OUT/"bayesian_issued.npz") as archive:
        r,adjusted=archive["third_radius"],archive["hybrid_radius"]
    curve=[(h,f"{100*r[:,h-1].mean():.2f}",f"{100*adjusted[:,h-1].mean():.2f}") for h in (1,3,10,20,30)]
    script="../../c_restocking_toolkit/q1_bayesian_gain_covariance"
    html=f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>第一问 · 贝叶斯增益与三维协方差对照</title>
<style>body{{max-width:1060px;margin:30px auto;padding:0 22px 50px;color:#20333f;font:16px/1.65 system-ui,"Microsoft YaHei",sans-serif}}h1,h2{{line-height:1.35}}h1{{font-size:clamp(26px,4vw,38px)}}h2{{font-size:21px;margin-top:1.6em}}.scroll{{overflow-x:auto}}table{{width:100%;border-collapse:collapse;font-size:14px;font-variant-numeric:tabular-nums}}th,td{{padding:8px;text-align:right;white-space:nowrap;border-bottom:1px solid #d2dfe2}}th{{background:#eaf2f4}}th:first-child,td:first-child{{white-space:normal;text-align:left}}a{{color:#145b7b}}code{{overflow-wrap:anywhere}}</style>
<main><h1>第一问 · 贝叶斯神经增益和三维协方差能带来多少改善</h1>
<p><b>截图中优先选第三行作为主方案。</b>第三、四行的轨迹中心相同。第三行在四个测试环境的最低覆盖为 {100*baseline.least_env:.2f}%，第四行为 {100*four.least_env:.2f}%。若优先追求达到 95% 覆盖后的最小半径，可以选第四行。整段覆盖要求未来 30 个三维腕点全部落入各自的预测球。</p>
{table(headers,main)}
<p>第一个改进是把 M4 已发出轨迹与独立训练的腕部位移专家按 70% 和 30% 组合，腕部专家沿预测起点的运动方向估计向前和侧向的位移。第二个改进是在每个窗口内重新分配 30 步半径。上一轮从 {int(base_old.covered)}/760 起，只改中心为 746/760，再加入分步分配为 747/760。第三行相较第二行，平均半径相等，轨迹误差与整段覆盖同时改善。 <a href="../第一问转向预警与分步半径改进/第一问预警与分步半径改进实测.html">上轮逐项消融</a></p>
<h2>贝叶斯模块的接法</h2>
<p>新模型只读取起点及此前的观测，最近最多 16 帧进入可学习滤波器。增益网络根据位置创新、实际时间间隔、质量和滤波状态估计三轴的位置与速度更新量。增益输出层的权重和偏置采用可学习的高斯后验，训练用重参数采样和 KL 正则。两次随机初始化独立训练，每个种子在推断时抽取 16 组权重，每组在同一个历史窗口内共享。完全缺失的腕点跳过测量更新。</p>
<p>滤波器传播位置和速度的误差协方差。未来每一步由正对角 Cholesky 因子预测三维残差协方差，再加入传播到对应步的状态不确定性和不同预测均值之间的协方差。由此得到每步一个正定的 3×3 矩阵。原文的随机增益思想在这里扩展为 30 步腕点预测，已有论文实验权重没有参与训练。 <a href="https://arxiv.org/html/2309.03058">Bayesian KalmanNet 原文</a> · <a href="https://github.com/yonatandn/Uncertainty-Quantification-in-Model-Based-DL/tree/Bayesian-KalmanNet">作者实现</a></p>
<p>验证集给贝叶斯新轨迹的融合权重为 <b>{bayes['share']:.0%}</b>。正式同宽输出保留第三行的轨迹中心，用每一步协方差迹的平方根表示径向误差尺度，调整已有半径。规则限制单步倍率，并精确约束每个窗口的 30 个半径之和。第三行新增覆盖 {cases} 个窗口、新增漏失 {lost} 个，平均半径为 {result.radius_cm:.2f} 厘米，转向整段覆盖 {int(result.turn_covered)}/99。将已冻结规则应用到第四行已经发出的宽度，新增覆盖 {plus4} 个窗口、新增漏失 {minus4} 个。这一附加回放没有为第四行单独选参数。</p>
{table(["预测步","第三行球半径（cm）","加入协方差后的球半径（cm）"],curve)}
<h2>学习增益、参数采样与半径分配分别贡献什么</h2>
{table(["增益处理","单独预测总体 / 转向误差（cm）","单独预测整段覆盖","其独立校准半径（cm）","各自验证选型后同宽覆盖","统一采用贝叶斯分配强度的同宽覆盖"],tests)}
<p>各模型分别在验证环境选融合比例与分配强度，再在同一批测试窗口评分。最后一列固定第三行的中心和同一个分配强度，专门检验后验采样带来的独立作用。去掉采样均值的散布项，其余计算完全相同时覆盖为 {int(a.loc[('bayesian','without_gain_sampling_variance'),'covered'])}/760。用训练环境按预测步估计的固定误差尺度分配为 {int(a.loc[('bayesian','gate_time_profile_cov_only'),'covered'])}/760。本次贝叶斯增益参数采样对总预测协方差迹的平均占比为 {100*diagnostics.loc['bayesian','learned_weight_trace_share']:.3f}%。</p>
<p>新增模块单独提供轨迹时，总体误差 {stand.ade_cm:.2f} 厘米，转向误差 {stand.turn_ade_cm:.2f} 厘米。它在独立校准环境得到的球半径在测试集平均为 {stand.radius_cm:.2f} 厘米，整段覆盖 {int(stand.covered)}/760，低于 95% 目标。因此主对照始终把球半径、覆盖和点误差一起报告。</p>
<h2>椭球与环境变化</h2>
<p>协方差矩阵还可表示不同方向的误差。椭球用马氏距离定义，在校准时对同一窗口的 30 个距离取最大值。沿用融合中心的椭球在独立测试环境中覆盖 {int(e.covered)}/760，转向覆盖 {int(e.turn_covered)}/99，未达到目标。它的平均体积等效半径为 {e.volume_equivalent_cm:.2f} 厘米，最长半轴为 {e.largest_semiaxis_cm:.2f} 厘米，表首的半径均为三维欧氏球半径。 <a href="ellipsoid_coverage.csv">椭球指标</a> · <a href="environment_spheres.csv">分环境的球覆盖</a></p>
<h2>比较与复算</h2>
<p>共用原始观测、环境划分与合格窗口。10 个训练环境拟合网络，3 个训练环境选择轮次，另 3 个用于独立分支的 95% 整段校准。第三行的融合权重和半径规则在 4 个官方验证环境冻结，再对另外 4 个测试环境的 760 个窗口评分。复算基线时单独从头拟合原腕点专家和低风险门槛分类器，读取 M4 已发出的预测。逐窗核对第三行原存档的点误差、半径和覆盖，旧权重读取数为 0。由于同一测试集在前期使用过，这些是回顾结果。</p>
<p>在线入口在 {check['prefixes']} 个原始前缀上核对，包含第三行、第四行所有覆盖结果发生变化的窗口，覆盖三种数据来源与双手。所有预测步的协方差都通过正定性检查。 <a href="{script}/run_bayesian.py">独立实验入口</a> · <a href="{script}/forecast_bayesian.py">在线预测入口</a> · <a href="raw_prefix_checks.json">核验记录</a></p>
<p><a href="sphere_comparison.csv">完整球形对照</a> · <a href="covariance_diagnostics.csv">协方差与采样诊断</a> · <a href="760个起点配对.csv">760 个起点</a> · <a href="99个转向配对.csv">99 个转向窗口</a> · <a href="frozen_before_test.json">冻结设置</a></p></main></html>"""
    path=OUT/"第三行与贝叶斯增益协方差实测.html"
    path.write_text(html,encoding="utf-8")
    print(path,flush=True)


if __name__=="__main__":
    report()
