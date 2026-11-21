"""Run the independent fixed-center transfer experiment end to end."""
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
os.environ.setdefault("OMP_NUM_THREADS", "3")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "2")

from transfer_config import OUT
from config import write_json
from refit import refit
from motion import Inputs
from error_scale import fit_scales
from select_bands import fit_and_lock
from evaluate_transfer import score
from feedback_diagnosis import diagnose
from verify_transfer import verify
from transfer_report import report


def run():
    refit()
    write_json(OUT/"scale_fit.json",fit_scales(Inputs()))
    fit_and_lock()
    score()
    diagnose()
    verify()
    report()


if __name__=="__main__":
    run()
