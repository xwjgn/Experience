"""One-command rebuild. All learned files are written by this run; baselines are read only during scoring."""
from pathlib import Path
import subprocess
import sys


if __name__ == "__main__":
    folder=Path(__file__).resolve().parent
    for entry in ("qsm_data.py","qsm_train.py","qsm_q1.py","qsm_q2.py","qsm_report.py"):
        print(f"Running {entry}",flush=True)
        subprocess.run([sys.executable,"-u",str(folder/entry)],cwd=folder,check=True)
