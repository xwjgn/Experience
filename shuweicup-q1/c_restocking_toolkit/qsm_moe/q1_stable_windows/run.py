"""Refit the reference and origin-time decisions, freeze, score, and check live issuance."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
os.environ.setdefault("OMP_NUM_THREADS","3")
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")

from fresh_qsm import fit_qsm
from freeze_gate import freeze
from phase_rule import export_rule
from compare_windows import score
from verify_stability import verify
from stable_report import report


def run():
    fit_qsm()
    freeze()
    export_rule()
    score()
    verify()
    report()


if __name__=="__main__":
    run()
