"""Export the already-frozen low-turn rule; applying it needs only QSM's own issue-time outputs."""
import joblib
import numpy as np

from stable_config import OUT
from config import digest,read_json,write_json
from stability_router import Policy, origin_checks


def export_rule():
    path = OUT/"frozen_before_test.json"
    lock = read_json(path)
    assert digest(OUT/"origin_router.joblib")==lock["router_sha256"]
    trained = joblib.load(OUT/"origin_router.joblib")
    spec = lock["policies"]["phase_only"]
    simple = spec["family"]=="phase_only" and spec["protection"]==1 and spec["core_factor"]>=spec["factor"]
    result = dict(derived_from_sha256=digest(path),policy=spec,simple=simple,veto=trained.veto)
    if simple:
        threshold = trained.thresholds["phase_only"][str(spec["fraction"])]
        result.update(score_threshold=threshold,maximum_turn_probability=1-threshold)
    write_json(OUT/"phase_rule.json",result)
    return result


def apply_rule(spec,x,radius):
    admissible,_ = origin_checks(x,spec["veto"])
    score = 1-x.turn_probability.to_numpy(dtype=np.float32)
    chosen = admissible & (score>=spec["score_threshold"])
    high = np.zeros(len(x),bool)
    return Policy(**spec["policy"]).apply(radius,chosen,high),chosen,high


if __name__=="__main__":
    export_rule()
