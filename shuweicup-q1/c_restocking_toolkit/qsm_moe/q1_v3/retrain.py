"""Fit the reference architecture into v3 using raw observations and random initialization."""
from config import BASE, WORK, OUT, make_dirs, save
import qsm_data
from features import make_features
from phase import fit_stage
from motion import fit_motion
from distribution import select


def train():
    make_dirs()
    qsm_data.OUT, qsm_data.CACHE = WORK, WORK / "cache"
    qsm_data.prepare()
    make_features()
    fit_stage()
    fit_motion()
    select()
    save("backbone_provenance.json", dict(source="canonical positions, timestamps, quality flags and decoded pixels",
        previous_parameters_loaded=0, fresh_directory=str(BASE.relative_to(OUT)),
        phase_comparator="archived v1 probabilities used only for the v1 rule comparison"))


if __name__ == "__main__":
    train()
