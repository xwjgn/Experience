# 第一问　腕部轨迹融合预测与贝叶斯协方差估计

数维杯 C 题第一问研究补货操作中的手部运动和观测可靠性。在一个预测起点，模型根据截至该帧的历史观测，为每只手给出未来约 1 秒的 30 个腕部三维坐标和 30 个预测半径。

本目录提供模型源码、同窗口的比较结果及复算程序。逐窗口表采用匿名窗口编号和环境分组，保留转向标签与评测指标。三个原始数据压缩包另附文件大小和 SHA-256，见[原始数据发布页](https://github.com/xwjgn/Experience/releases/tag/shuweicup-q1-data-v1)。

## 同一批窗口上的比较

按照官方环境划分，测试集来自 4 个独立于训练和验证划分的环境。每条记录、每只手从第 10 帧起每隔 30 帧取一个起点，保留 30 步参考腕点均合格的 760 个窗口，其中 99 个为转向窗口。轨迹误差为每个窗口 30 步三维欧氏误差的平均值。一个窗口全部 30 步都落在对应的预测球内，才计为整段覆盖。

| 方法 | 平均轨迹误差 | 转向轨迹误差 | 平均半径 | 整段覆盖（目标 95%） | 转向整段覆盖 |
|---|---:|---:|---:|---:|---:|
| 保持起点位置 | 17.31 cm | 9.95 cm | — | — | — |
| 归档原方案 | 10.04 cm | 8.95 cm | 30.09 cm | 740/760（97.37%） | 98/99 |
| M4 | 9.87 cm | 8.90 cm | 37.71 cm | 751/760（98.82%） | 99/99 |
| 融合预测＋参考半径 | 9.35 cm | 8.58 cm | 34.84 cm | 747/760（98.29%） | 99/99 |
| **融合预测＋协方差分配** | **9.35 cm** | **8.58 cm** | **34.84 cm** | **754/760（99.21%）** | **99/99** |

融合预测使用 M4 的 70% 和腕部位移模型的 30% 作为中心权重。M4 使用短时关节注意力与 LSTM、LSTTN 式长时特征、阶段概率和整段校准。若近期观测质量合格且起点转向风险低，参考半径整体乘 0.80。贝叶斯分支学习观测更新增益，给出各步三维位置的协方差。根据协方差迹的平方根分配各步半径，并保持每个窗口的平均半径。

两种融合方法采用同一条中心轨迹。按协方差分配半径后，额外覆盖 7 个窗口。相同分配强度下，固定增益和确定性增益对照也达到 754/760。提前识别转向另行评估，M4 的冻结报警阈值命中 52/99，有 129 个误报。

## 复算公开结果

在本目录运行以下命令。复算仅使用 Python 标准库。

```text
python summarize_results.py
python summarize_results.py --check
```

第一条命令从逐窗口表重新计算各项指标，第二条检查结果是否与已发布的汇总一致。逐窗的长度单位均为厘米，`whole_path_covered` 为 1 表示整段覆盖。保持起点位置仅提供点预测，区间指标留空。两种融合方法在表中分别记为 `fusion_reference` 和 `fusion_covariance`。[逐窗口结果](results/paired_windows.csv) · [可下载的汇总](results/summary.csv)

附加实验包含原方案的轨迹和区间对照、LSTTN 与 QSM 系列、M1–M4 比较、起点转向预警、观测质量以及固定、确定性和贝叶斯增益消融。[归档轨迹对比](results/archived_trajectory_comparison.csv) · [归档区间对比](results/archived_band_comparison.csv) · [LSTTN 与 QSM](results/lsttn_qsm_comparison.csv) · [Transformer-LSTM 消融](results/transformer_lstm_ablation.csv) · [转向预警](results/turn_alerts.csv) · [观测质量](results/observation_quality.csv) · [增益与协方差消融](results/gain_ablation.csv)

## 模型实现

| 环节 | 源码入口 |
|---|---|
| 归档原方案，既有终点配合阻尼运动轨迹与整段校准 | [原方案对比实现](c_restocking_toolkit/literature_experiments/q1_gru_path_calibration.py)，对应 `anchored_damped` 与 `joint_adaptive` |
| 质量感知 Transformer-LSTM 与 M4 | [原始观测和网络训练](c_restocking_toolkit/q1_transformer_lstm/run.py) · [M4 在线预测](c_restocking_toolkit/q1_transformer_lstm/forecast.py) |
| 起点质量和低转向风险 | [用当前及历史观测估计收紧系数](c_restocking_toolkit/q1_origin_study/issue_origin.py) |
| 腕部位移专家与参考半径 | [位移模型](c_restocking_toolkit/q1_origin_refinement/path_expert.py) · [轨迹融合](c_restocking_toolkit/q1_origin_refinement/refine.py) |
| 贝叶斯增益和预测协方差 | [增益网络](c_restocking_toolkit/q1_bayesian_gain_covariance/neural_gain.py) · [半径分配](c_restocking_toolkit/q1_bayesian_gain_covariance/gain_calibration.py) · [训练和对照](c_restocking_toolkit/q1_bayesian_gain_covariance/run_bayesian.py) |
| 早期 QSM 和质量感知 LSTTN 对照 | [QSM 实现](c_restocking_toolkit/qsm_moe) · [LSTTN 实验](c_restocking_toolkit/qsm_moe/q1_lsttn/run.py) |

## 使用本地数据预测

实际训练环境为 Python 3.12、PyTorch 2.8.0（CUDA 12.8）。核心数值库版本见 [requirements.txt](requirements.txt)。各阶段按相对目录读取本地材料。

- 下载三个附件，保留原文件名，放入 `Restocking_Problem_and_Data_EN/Raw_Data/`。[EgoDemo.zip](https://github.com/xwjgn/Experience/releases/download/shuweicup-q1-data-v1/EgoDemo.zip) · [EgoPro.zip](https://github.com/xwjgn/Experience/releases/download/shuweicup-q1-data-v1/EgoPro.zip) · [EgoStandard.zip](https://github.com/xwjgn/Experience/releases/download/shuweicup-q1-data-v1/EgoStandard.zip)
- `Restocking_Problem_and_Data_EN/Restocking_Tasks_and_Constraints/` 包含本地的 `record_split.csv` 和 `source_segment_index.csv`。
- `分析报告/` 保存本地训练产物和冻结配置。训练依次采用 M4、起点风险、腕部位移融合、贝叶斯增益各阶段的实验入口。完整历史对照的评分代码还会读取先前实验的本地输出，部分早期 QSM/LSTTN 实验另需其源码指定的作者仓库快照。
- 主方案实验读取独立 M4 已给出的预测值，重新拟合腕部位移与低风险模型，并从随机初始化训练贝叶斯增益网络。

已有上述本地数据及各阶段冻结产物时，可运行 [predict_wrist.py](predict_wrist.py)。将 `RECORD_ID` 换为本地记录编号，将 `LOCAL_PROJECT` 换为包含原始数据、`c_restocking_toolkit` 和 `分析报告` 的目录。

```text
python predict_wrist.py RECORD_ID right 70 --project LOCAL_PROJECT
```

输出 JSON 包含 30 个预计时刻 `time_s`、世界坐标 `center_m` 和预测半径 `radius_m`。位置和半径的单位为米。
