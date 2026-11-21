"""Freeze the stability rule using only official training and validation environments."""
from dataclasses import asdict
import joblib
import numpy as np
import pandas as pd

from stable_config import BASE, OUT, REPLICA, WORK
from config import digest, read_json, write_json
from motion import Inputs
from stability_router import OriginRouter, Policy, candidates, STRESS, TURN_CAP, FAMILIES
from stable_data import FixedReference


def assess(data, radius, selected, guarded):
    error, base, turn, stable = data.error,data.radius,data.turn,data.stable
    hit = (error<=radius).all(1)
    old = (error<=base).all(1)
    shrunk = (radius<base-1e-9)
    stressed = radius-np.where(shrunk,STRESS*base,0.)
    stress_missed = old & turn & ~(error<=stressed).all(1)
    env = data.meta.env.to_numpy()
    gains, cover_margin = [], []
    for e in np.unique(env):
        use = env==e
        gains.append(1-radius[use].mean()/base[use].mean())
        cover_margin.append(hit[use].mean()-min(.95,old[use].mean()))
    price = 1-radius.mean()/base.mean()
    stable_precision = stable[selected].mean() if selected.any() else stable.mean()
    return dict(n=len(hit),covered=int(hit.sum()),coverage=float(hit.mean()),base_covered=int(old.sum()),
                turns=int(turn.sum()),turn_covered=int(hit[turn].sum()),base_turn_covered=int(old[turn].sum()),
                lost_turns=int((old & ~hit & turn).sum()),gained_turns=int((~old & hit & turn).sum()),
                stress_lost_turns=int(stress_missed.sum()),minimum_env_margin=float(min(cover_margin)),
                routed=int(selected.sum()),routed_turns=int((selected & turn).sum()),
                turn_exposure=float(selected[turn].mean()),protected=int(guarded.sum()),
                actual_stable=int(stable.sum()),routed_stable=int((stable & selected).sum()),
                stable_precision=float(stable_precision), stable_rate=float(stable.mean()),
                mean_radius_cm=float(radius.mean()*100),mean_gain=float(price),env_gain=float(np.mean(gains)),
                min_env_gain=float(min(gains)), pooled_margin=float(hit.mean()-min(.95,old.mean())))


def acceptable(m, mode):
    nominal = m["minimum_env_margin"]>=-1e-10 and m["pooled_margin"]>=-1e-10
    turns_held = m["turn_covered"]>=m["base_turn_covered"]
    enrichment = m["stable_precision"]+.015>=m["stable_rate"]
    if not (nominal and turns_held and enrichment):
        return False
    if mode=="efficient":
        return m["lost_turns"]<=1 and m["turn_exposure"]<=.4 and m["min_env_gain"]>=-.035
    return (m["lost_turns"]==0 and m["stress_lost_turns"]==0 and m["turn_exposure"]<=TURN_CAP
            and m["covered"]>=m["base_covered"]-.012*m["n"] and m["min_env_gain"]>=-.025)


def pick(trials, mode, only=None):
    eligible = []
    for policy,metrics in trials:
        if only and policy.family not in ("current",only):
            continue
        if not all(acceptable(m,mode) for m in metrics):
            continue
        # Compare radius reductions relative to each environment's own original width.
        gains = [.5*(m["env_gain"]+m["mean_gain"]) for m in metrics]
        utility = .55*min(gains)+.45*np.mean(gains)
        if mode=="recognition":
            if any(m["routed"]<max(12,.04*m["n"]) for m in metrics):
                continue
            # Reward enrichment of independently defined physically stable trajectories
            # with a one-standard-error penalty for a small selected subset.
            lifts = []
            for m in metrics:
                p = m["stable_precision"]
                conservative_precision = p-np.sqrt(p*(1-p)/m["routed"])
                lifts.append(conservative_precision/max(m["stable_rate"],1e-5))
            utility += .7*min(lifts)+.3*np.mean(lifts)
        eligible.append((float(utility),-max(m["turn_exposure"] for m in metrics),-policy.protection,policy))
    if not eligible:
        return Policy()
    return max(eligible,key=lambda row:row[:3])[-1]


def protocol():
    return dict(goal="Reduce average radius of the fixed 30-step QSM v3 point forecast while preserving whole-path turn coverage.",
        labels="Established future-turn label at frame +10; motion stability independently scored from the realized 3D path.",
        stable_truth="Within 3 cm throughout OR three 10-frame velocities with adjacent angles <=30 degrees, CV <=0.35 and straightness >=0.93; actual timestamps.",
        joint_risk="Boosted origin-time future-stability and turn scores, vulnerability of a turn at 72% reference radius and the conditional 90th percentile of the full-path error-to-radius ratio.",
        roles="10 training environments fit risk models with stride 5; 3 training gate environments set score ranks; 3 other training environments and 4 official validation environments check the decision. Full 30-step valid windows on the fixed grid are used for selection and scoring.",
        primary_rule="Preserve each incumbent-covered turn in gate/calibration/validation, including an additional five percentage points of stress on tightened steps. Expose at most 25% of true turns in each partition. Preserve local 95% coverage or the incumbent local coverage where lower. Retain the incumbent as a candidate.",
        recognition_rule="Additional validation refinement before test scoring: compare persistence, a stability-first score and a two-level contraction. Separately freeze the rule that enriches physically stable futures under the same turn safeguards with at least 4% or 12 selected windows per partition.",
        efficient_rule="Secondary predefined rule holds aggregate turn hits per partition. Up to one paired loss per partition and 40% turn exposure are allowed.",
        objective="Maximum environment-balanced width gain with 55% weight on the worst partition. All actual labels are restricted to training and retrospective evaluation.",
        extra_checks="Phase-only and hand-crafted rules; uniform scaling at the frozen mean validation gain and at the same retrospective mean radius per split; paired losses, validation leave-one-environment-out for the router, raw-prefix parity, delayed-feedback timing.",
        history="Only a preceding scheduled forecast whose endpoint has arrived can contribute. Full path ratios require all 30 wrist observations to be valid.",
        test_status="Previously reported official test environments. New gate and thresholds are locked before their test predictions are scored.")


def freeze():
    OUT.mkdir(parents=True,exist_ok=True)
    write_json(OUT/"protocol.json",protocol())
    d = Inputs()
    partitions = [d.fit,d.gate,d.cal,d.val,d.test]
    np.testing.assert_array_equal(np.sum(partitions,axis=0),np.ones(len(d.meta)))
    ref = FixedReference(d)
    fit = ref.windows(d.fit,stride=5)
    gate,cal,val = (ref.windows(m) for m in (d.gate,d.cal,d.val))
    phase_rule = read_json(BASE/"phase_selection.json")["rule"]
    veto = phase_rule.get("threshold",.30)
    router = OriginRouter().fit(fit,gate,veto)
    joblib.dump(router,OUT/"origin_router.joblib")
    panels = [("gate",gate),("calibration",cal),("validation",val)]
    environments = sorted(val.meta.env.unique())
    panels.extend(("val_without_"+e,val.take(val.meta.env.to_numpy()!=e)) for e in environments)
    scores = {name:router.scores(data.x) for name,data in panels}
    candidates_to_score = list(candidates((*FAMILIES,"phase_only")))
    trials,rows = [],[]
    for policy in candidates_to_score:
        metrics = {}
        for name,data in panels:
            radius,selected,guarded = policy.issue(router,data.x,data.radius,scores[name])
            metrics[name] = assess(data,radius,selected,guarded)
            if name in ("gate","calibration","validation"):
                rows.append(dict(**asdict(policy),partition=name,**metrics[name]))
        trials.append((policy,metrics))

    def chosen(mode, val_panel="validation",only=None):
        runs = [(p,[m["gate"],m["calibration"],m[val_panel]]) for p,m in trials
                if p.family in (*FAMILIES,"current") or p.family==only]
        return pick(runs,mode,only)
    locked = {name:asdict(chosen(name)) for name in ("conservative","efficient","recognition")}
    locked["phase_only"] = asdict(chosen("conservative",only="phase_only"))
    locked["motion_only"] = asdict(chosen("conservative",only="motion"))
    main = Policy(**locked["conservative"])
    val_radius,selected,padded = main.issue(router,val.x,val.radius,scores["validation"])
    held = []
    for e in environments:
        subset = val.take(val.meta.env.to_numpy()==e)
        for family,mode in ((None,"conservative"),("phase_only","conservative"),(None,"recognition")):
            policy = chosen(mode,"val_without_"+e,only=family)
            r,ch,pr = policy.issue(router,subset.x,subset.radius)
            held.append(dict(held_out=e,comparison=family or mode,**asdict(policy),**assess(subset,r,ch,pr)))
    pd.DataFrame(held).to_csv(OUT/"held_out_validation.csv",index=False)
    pd.DataFrame(rows).to_csv(OUT/"validation_candidates.csv",index=False)
    lock = dict(policies=locked,training_rows=len(fit.x),roles=d.roles,selection_counts={n:len(p.x) for n,p in panels[:3]},
                primary_validation=assess(val,val_radius,selected,padded),veto_probability=router.veto,
                uniform_validation_factor=float(val_radius.mean()/val.radius.mean()),
                candidates_per_partition=len(trials),prior_checkpoints_loaded=0,
                qsm_center_sha256=digest(WORK/"corrected_center.npy"),qsm_reference_sha256=digest(REPLICA/"frozen_before_test.json"),
                router_sha256=digest(OUT/"origin_router.joblib"),protocol_sha256=digest(OUT/"protocol.json"))
    write_json(OUT/"frozen_before_test.json",lock)
    print("Locked",locked,"validation",lock["primary_validation"],flush=True)


if __name__=="__main__":
    freeze()
