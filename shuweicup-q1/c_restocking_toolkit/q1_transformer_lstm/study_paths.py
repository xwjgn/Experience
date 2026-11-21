"""Paths for the standalone hand-observation LSTM experiment."""
import hashlib
import json
import os
from pathlib import Path

os.environ.setdefault("MKL_THREADING_LAYER","SEQUENTIAL")
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
RELEASE = ROOT/"Restocking_Problem_and_Data_EN"
TABLES = RELEASE/"Restocking_Tasks_and_Constraints"
RAW = RELEASE/"Raw_Data"
OUT = ROOT/"分析报告"/"第一问独立质量感知Transformer-LSTM"
DATA = OUT/"derived_from_release"
KEY = ["split","env","sample","hand","origin"]
MODES = ("dwell","turn","translate")
SEEDS = (78241,78259)


def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False),encoding="utf8")


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream,"sha256").hexdigest()
