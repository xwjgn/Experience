"""Fit references and calibration multipliers within official training; select on validation only."""
import numpy as np
import pandas as pd

from transfer_config import OUT, REPLICA, WORK
from config import digest, read_json, write_json
from calibration import (Panel, LEVELS, SHAPES, SCHEMES, fit_prior, replay, incumbent, outer_factors)
from intervals import evaluate_candidate, order_stat
from motion import Inputs

REFERENCE = {"kind":"incumbent"}
TARGET = .95


def describe(p, radius):
    mask = p.complete
    return evaluate_candidate(p.error[mask], radius[mask], p.env[mask], p.turn[mask])


def get_radius(recipe, p, previous, replays, priors):
    kind = recipe["kind"]
    if kind=="incumbent":
        return previous
    if kind=="flat":
        return np.full_like(previous, recipe["radius_m"])
    base = replays[recipe["shape"]][recipe["scheme"]]
    if kind=="marginal":
        return base*np.asarray(recipe["factors"])
    multiplier = priors[recipe["shape"]]["weights"] if recipe["weighted"] else 1.
    radius = base*(multiplier*recipe["factor"])
    if recipe.get("floor",0.)>0:
        radius = np.maximum(radius, recipe["floor"]*previous)
    return radius


def fit_and_lock():
    d = Inputs()
    if np.any(d.test & (d.gate | d.cal | d.val)):
        raise ValueError("Overlapping environment roles")
    with np.load(OUT/"issued_scales.npz") as widths:
        gate = Panel(d, d.gate, widths)
        cal = Panel(d, d.cal, widths)
        val = Panel(d, d.val, widths)
    priors = {s:fit_prior(gate,s) for s in SHAPES}
    for s, prior in priors.items():
        np.savez(OUT/(s+"_training_prior.npz"), **prior)
    replay_cal = {s:replay(cal,s,priors[s]) for s in SHAPES}
    replay_val = {s:replay(val,s,priors[s]) for s in SHAPES}
    ref_val = incumbent(val)
    incumbent_stats = describe(val, ref_val)
    original_controls = {}
    trials = []

    def evaluate(recipe):
        radius = get_radius(recipe,val,ref_val,replay_val,priors)
        stats = describe(val, radius)
        trials.append((recipe, stats))
        return stats

    evaluate(REFERENCE)
    for shape in SHAPES:
        for scheme in SCHEMES:
            for weighted in (False, True):
                weight = priors[shape]["weights"] if weighted else 1.
                base = replay_cal[shape][scheme][cal.complete]*weight
                for level in LEVELS:
                    for pooling in ("pooled","max_env"):
                        scalar = outer_factors(cal.error[cal.complete],base,cal.env[cal.complete],level,pooling)
                        recipe = dict(kind="joint",shape=shape,scheme=scheme,weighted=weighted,
                                      level=level,pooling=pooling,factor=float(scalar),floor=0.)
                        evaluate(recipe)
                        # Additional predefined floor for the comparison at matched validation coverage.
                        if scheme in ("endpoint","endpoint_recent"):
                            evaluate({**recipe,"floor":.90})
                        if not weighted and level==TARGET and pooling=="pooled" and scheme in ("static","endpoint"):
                            original_controls[shape+"_"+scheme+"_95"] = recipe

    val_goal = int(incumbent_stats["covered"])
    turn_goal = round(incumbent_stats["turn_coverage"]*incumbent_stats["turns"])
    def rank(trial):
        recipe, stats = trial
        return (stats["shortfall"], stats["radius_cm"], int(recipe!=REFERENCE))
    chosen, stats = min(trials,key=rank)
    matched_trials = [(r,m) for r,m in trials if m["eligible"] and m["covered"]>=val_goal
                      and round(m["turn_coverage"]*m["turns"])>=turn_goal]
    matched, matched_stats = min(matched_trials,key=rank) if matched_trials else (REFERENCE,incumbent_stats)
    grid = cal.complete
    marginal_scores = cal.error[grid]/replay_cal["gru"]["static"][grid]
    original_controls["marginal"] = dict(kind="marginal",shape="gru",scheme="static",
        factors=[order_stat(marginal_scores[:,h],TARGET) for h in range(30)])
    original_controls["flat"] = dict(kind="flat",radius_m=order_stat(cal.error[grid].max(1),TARGET))

    # Secondary experiment reproducing the original validation calibration convention for one fixed recipe.
    # Each reported validation fold excludes that entire environment. Test receives one fixed pooled scalar.
    basic = original_controls["gru_endpoint_95"]
    env = val.env[val.complete]
    error = val.error[val.complete]
    base = replay_val["gru"]["endpoint"][val.complete]
    folds = {e:outer_factors(error[env!=e],base[env!=e],env[env!=e],TARGET,"pooled") for e in np.unique(env)}
    test_scalar = outer_factors(error,base,env,TARGET,"pooled")
    alternate = dict(recipe={**basic,"factor":test_scalar}, validation_folds=folds)

    lock = dict(target=TARGET, roles=d.roles, primary_uses_validation_for_calibration=False,
        counts=dict(gate=int(gate.complete.sum()),calibration=int(grid.sum()),validation=int(val.complete.sum())),
        center_sha256=digest(WORK/"corrected_center.npy"), qsm_lock_sha256=digest(REPLICA/"frozen_before_test.json"),
        scale_files={str(p.name):digest(p) for p in sorted(OUT.glob("error_scale_*.pt"))},
        priors_sha256={s:digest(OUT/(s+"_training_prior.npz")) for s in SHAPES},
        previous_checkpoints_loaded=0, controls=original_controls, incumbent=incumbent_stats,
        selected=dict(recipe=chosen,validation=stats),
        matched=dict(recipe=matched,validation=matched_stats,validation_target_all=val_goal,
                     validation_target_turn=turn_goal),
        alternate_validation_calibration=alternate, n_validation_candidates=len(trials))
    pd.DataFrame([{**r,**m} for r,m in trials]).to_csv(OUT/"validation_candidates.csv",index=False)
    write_json(OUT/"frozen_before_test.json",lock)
    print("FROZEN",lock["counts"],chosen,stats,matched,matched_stats,flush=True)
    return lock


if __name__=="__main__":
    fit_and_lock()
