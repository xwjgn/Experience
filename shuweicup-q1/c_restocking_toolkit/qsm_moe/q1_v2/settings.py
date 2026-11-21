"""Paths and protocol shared only by the second Q1 experiment."""
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
QSM = HERE.parent
KIT = QSM.parent
ROOT = KIT.parent
OUT = ROOT / "分析报告" / "QSM-MoE第一问优化"
WORK = OUT / "fresh"
PREVIOUS = ROOT / "分析报告" / "QSM-MoE融合方案对比"
ORIGINAL = ROOT / "分析报告" / "第一问GRU轨迹与整段校准"
sys.path.insert(0, str(QSM))
from qsm_data import write_json

KEY = ["split", "env", "sample", "hand", "origin"]
SEED = 41171
MODES = ("dwell", "turn", "translate")


def array(name):
    return np.load(WORK / "cache" / (name + ".npy"), mmap_mode="r")


def equal_env(values, env):
    return float(pd.Series(np.asarray(values)).groupby(np.asarray(env)).mean().mean())


def make_dirs():
    WORK.mkdir(parents=True, exist_ok=True)
    assert PREVIOUS.resolve() not in (OUT.resolve(), *OUT.resolve().parents)
