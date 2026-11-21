"""Rebuild the Q1 revision and its comparisons without modifying either earlier experiment."""
import os
import hashlib
import subprocess
import sys
from pathlib import Path

from filelock import FileLock
from settings import OUT,make_dirs,write_json


TABLES=("three_way_metrics.csv","turn_metrics.csv","visual_interventions.csv","turn_loss_interventions.csv",
        "phase_selection.json","motion_selection.json","frozen_before_test.json")


def fingerprints():
    return {name:hashlib.sha256((OUT/name).read_bytes()).hexdigest() for name in TABLES if (OUT/name).exists()}


def rebuild():
    here=Path(__file__).resolve().parent
    before=fingerprints()
    env=os.environ.copy()
    env.setdefault("CUBLAS_WORKSPACE_CONFIG",":4096:8")
    env.setdefault("OMP_NUM_THREADS","3")
    env.setdefault("OPENBLAS_NUM_THREADS","2")
    write_json(OUT/"protocol.json",dict(scope="Q1 wrist forecasting only",comparison=["original","QSM v1","QSM v2"],
        horizon_frames=30,stage_forecast_horizon=10,previous_model_parameters_loaded_for_training=0,
        independent_unit="official environment split; common non-overlap grid and identical wrist origins",
        fit_validation_test_order=["fit experts on 10 training environments","fit probability and mixed-distribution components on 3 training environments",
              "fit interval factors on 3 further training environments","select on 4 validation environments","freeze","score 4 retrospective test environments"],
        turn_selection="environment-mean turn F1, validation precision >=.25 and alarm share <=.35",
        visual_selection="global validation ADE gain >=.02 cm, >=3/4 environments improving, macro FDE increase <=.05 cm",
        interval_selection="narrowest validation band attaining .95 whole-path coverage; otherwise greatest validation coverage",
        control_design="2 seeds with common batches for masked/aligned visual training and class weighting × stage gate-gradient detachment"))
    for script in ("diagnose.py","refit_inputs.py","features.py","visual_controls.py","stage_controls.py",
                   "phase.py","motion.py","distribution.py","evaluate.py","verify.py","report.py"):
        print(f"Running {script}",flush=True)
        subprocess.run([sys.executable,"-u",str(here/script)],cwd=here,env=env,check=True)
    after=fingerprints()
    write_json(OUT/"reproducibility.json",dict(current_sha256=after,prior_sha256=before,
               identical_to_preceding_run=(before==after) if len(before)==len(TABLES) else None))


if __name__=="__main__":
    make_dirs()
    with FileLock(str(OUT/"run.lock"), timeout=0):
        write_json(OUT/"run_state.json",{"status":"running"})
        try:
            rebuild()
        except BaseException:
            write_json(OUT/"run_state.json",{"status":"incomplete"})
            raise
        write_json(OUT/"run_state.json",{"status":"complete"})
