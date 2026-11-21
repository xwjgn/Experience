"""Place every fitted parameter for this experiment in its own directory."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
KIT = HERE.parents[1]
ROOT = KIT.parent
OUT = ROOT / "分析报告" / "QSM-MoE第一问原方案校准迁移"
REPLICA = OUT / "qsm_retrained"
BASE = REPLICA / "backbone"
WORK = BASE / "fresh"
sys.path.insert(0, str(HERE.parent / "q1_v3"))
import config

config.OUT, config.BASE, config.WORK = REPLICA, BASE, WORK
config.backbone_settings.OUT, config.backbone_settings.WORK = BASE, WORK
PREVIOUS = ROOT / "分析报告" / "QSM-MoE第一问转向与覆盖优化"
ORIGINAL = config.ORIGINAL
