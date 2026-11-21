"""Rerun the shared raw-data recipe into this experiment's private output directory."""
from settings import WORK, make_dirs
import qsm_data


if __name__=="__main__":
    make_dirs()
    # The original recipe resolves all learned outputs through these two module paths.
    # This entry runs in its own process; no QSM-v1 checkpoint or embedding is an input.
    qsm_data.OUT = WORK
    qsm_data.CACHE = WORK / "cache"
    qsm_data.prepare()
