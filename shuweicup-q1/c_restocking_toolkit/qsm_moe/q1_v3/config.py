"""Independent v3 workspace. Configure shared v2 methods before any of them resolve their data paths."""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
STUDY_CODE = HERE.parent / "q1_v2"
KIT = HERE.parents[1]
ROOT = KIT.parent
OUT = ROOT / "分析报告" / "QSM-MoE第一问转向与覆盖优化"
BASE = OUT / "backbone"
WORK = BASE / "fresh"
V2 = ROOT / "分析报告" / "QSM-MoE第一问优化"
V1 = ROOT / "分析报告" / "QSM-MoE融合方案对比"
ORIGINAL = ROOT / "分析报告" / "第一问GRU轨迹与整段校准"
sys.path.insert(0, str(STUDY_CODE))
import settings as backbone_settings

backbone_settings.OUT = BASE
backbone_settings.WORK = WORK
from qsm_data import write_json

KEY = ["split", "env", "sample", "hand", "origin"]
SEED = 56731


def read_json(path):
    return json.loads(path.read_text(encoding="utf8"))


def save(name, data):
    write_json(OUT / name, data)


def make_dirs():
    assert OUT.resolve() not in (V2.resolve(), V1.resolve(), ORIGINAL.resolve())
    WORK.mkdir(parents=True, exist_ok=True)


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 19), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def joined(a, b):
    merged = a.merge(b, on=KEY, how="inner", sort=False, validate="one_to_one")
    if len(merged) != len(a) or len(merged) != len(b):
        raise ValueError("Forecasts do not share identical wrist origins")
    return merged


def macro(v, env):
    return float(pd.Series(np.asarray(v)).groupby(np.asarray(env)).mean().mean())
