"""Rebuild QSM and its published v3 comparison band from raw inputs."""
from transfer_config import OUT
from retrain import train
from backbone import materialize
from point import select as freeze_centers
from intervals import select as freeze_v3_band

def refit():
    OUT.mkdir(parents=True, exist_ok=True)
    train()
    materialize()
    freeze_centers()
    freeze_v3_band()

if __name__ == "__main__":
    refit()
