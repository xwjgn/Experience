import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
os.environ.setdefault("MKL_THREADING_LAYER","SEQUENTIAL")
HERE = Path(__file__).resolve().parent
KIT = HERE.parents[1]
ROOT = KIT.parent
OUT = ROOT/"分析报告"/"第一问质量感知LSTTN与阶段概率预测"
DATA = OUT/"data"
FRAMES = KIT/"cache"/"canonical"/"frames.parquet"
SNAP = ROOT/"参考文献"/"论文与代码仓库"/"仓库快照"
PREVIOUS = ROOT/"分析报告"/"QSM-MoE第一问起点平稳识别与半径收紧"
KEY = ["split","env","sample","hand","origin"]
STAGES = ("dwell","turn","translate")
DEVICES = (61717,61729)


def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def save_json(path, value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8")


def digest(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle,"sha256").hexdigest()
