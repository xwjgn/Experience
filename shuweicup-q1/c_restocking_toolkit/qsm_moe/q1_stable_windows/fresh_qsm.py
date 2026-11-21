from stable_config import OUT
from retrain import train
from backbone import materialize
from point import select as select_points
from intervals import select as select_interval


def fit_qsm():
    OUT.mkdir(parents=True, exist_ok=True)
    train()
    materialize()
    select_points()
    select_interval()


if __name__ == "__main__":
    fit_qsm()
