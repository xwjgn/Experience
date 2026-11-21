"""Rebuild this two-target Q1 experiment from raw records and verify its locked scores."""
import os
import subprocess
import sys
from filelock import FileLock

from config import OUT,HERE,make_dirs,save,digest
from protocol import register

CHECKS=("point_validation.csv","coverage_validation.csv","frozen_before_test.json","four_way_metrics.csv",
        "paired_differences.csv","by_environment.csv","subgroups.csv")


def hashes():
    return {n:digest(OUT/n) for n in CHECKS if (OUT/n).is_file()}


def run():
    before=hashes()
    register()
    env=os.environ.copy()
    env.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    env.setdefault("OMP_NUM_THREADS","3")
    env.setdefault("OPENBLAS_NUM_THREADS","2")
    for entry in ("retrain.py","backbone.py","point.py","intervals.py","score.py","audit.py","study_report.py"):
        print("Running",entry,flush=True)
        subprocess.run([sys.executable,"-u",str(HERE/entry)],cwd=HERE,env=env,check=True)
    after=hashes()
    save("reproducibility.json",dict(previous=before,current=after,
         identical_to_previous=(before==after) if len(before)==len(CHECKS) else None))


if __name__=="__main__":
    make_dirs()
    with FileLock(str(OUT/"run.lock"),timeout=0):
        save("run_state.json",dict(status="running"))
        try:
            run()
        except BaseException:
            save("run_state.json",dict(status="incomplete"))
            raise
        save("run_state.json",dict(status="complete"))
