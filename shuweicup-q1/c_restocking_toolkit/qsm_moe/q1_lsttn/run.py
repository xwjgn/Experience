"""Rebuild the standalone first-question LSTTN study using the local paper-code snapshot and canonical measurements."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
os.environ.setdefault("MKL_THREADING_LAYER","SEQUENTIAL")
os.environ.setdefault("OMP_NUM_THREADS","3")
os.environ.setdefault("OPENBLAS_NUM_THREADS","2")

from prepare_lsttn import prepare
from train_lsttn import fit_all
from freeze_lsttn import freeze
from evaluate_lsttn import score
from matched_diagnostics import diagnose
from verify_lsttn import verify
from lsttn_report import report


def run():
    prepare()
    fit_all()
    freeze()
    score()
    diagnose()
    verify()
    report()


if __name__=="__main__":
    run()
