"""Lock detectors, thresholds and interval rules with the four validation environments."""
import hashlib
import joblib
import numpy as np
import pandas as pd

from wrist_history import DATA,KEY,M4_DIR,OUT,load_split,protect_weights,read,save
from event_models import Scorer
from horizon_budget import Windows,select_on_validation
from alarm_rules import anchors,combine,model_utility,select_online,stream


def fingerprint(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def reference(data,score,proxy):
    with np.load(M4_DIR/"M3_issued.npz") as archive:
        loc = pd.Index(archive["row"]).get_indexer(data.meta.row)
        if (loc < 0).any() or data.meta.duplicated(KEY).any():
            raise ValueError("Saved M4 forecasts differ from the benchmark window identifiers")
        center = archive["center"][loc]
        radius = archive["radius"][loc]
        phase = archive["phase"][loc,1]
    y = np.load(DATA/"y.npy",mmap_mode="r")[data.meta.row.to_numpy()]
    error=np.linalg.norm(y-center,axis=2)
    return Windows(data.meta,data.features,score,proxy,radius,error),phase


def freeze():
    scorer = Scorer(joblib.load(OUT/"fresh_tree_and_scales.joblib"))
    val = load_split("validation")
    assert len(val.meta)==401 and int(val.meta.turn.sum())==75
    y,env = val.meta.turn.to_numpy(),val.meta.env.to_numpy()
    probs,proxy = scorer.outputs(val.seq,val.features,val.meta.stage.to_numpy())
    rows=[]
    with np.load(OUT/"training_env_scores.npz") as cal:
        for name,p in probs.items():
            cg=model_utility(cal["turn"],cal[name],cal["env"])
            vg=model_utility(y,p,env)
            rows.append(dict(name=name,training_calibration=cg,validation=vg,objective=.35*cg+.65*vg))
        preferred=max(rows,key=lambda r:r["objective"])["name"]
        negatives = cal[preferred][~cal["turn"]].tolist()
    data, m4 = reference(val,probs[preferred],proxy)
    turns = val.meta.turn.to_numpy()
    m4_cut=read(M4_DIR/"frozen_before_test.json")["models"]["M3"]["turn_cut"]
    baseline = m4>=m4_cut
    streams=stream(scorer,"validation",preferred,negatives)
    grid=val.meta[KEY].merge(streams,on=KEY,validate="one_to_one")
    np.testing.assert_allclose(grid.stream_score,probs[preferred],atol=1.e-6,rtol=0)
    alarm,alarm_trials=select_online(y,probs[preferred],grid,baseline)
    # Optional late fusion consumes M4's already-issued stage score only. The independent detector has its own rule.
    mix_rows=[]
    for share in (.5,.75):
        out=np.empty(len(y))
        for e in np.unique(env):
            mask=env==e
            anchor=anchors(m4[~mask],probs[preferred][~mask])
            out[mask]=combine(m4[mask],probs[preferred][mask],anchor,share)
        mix_rows.append(dict(share=share,utility=model_utility(y,out,env)))
    mixture=max(mix_rows,key=lambda r:r["utility"])
    anchor=anchors(m4,probs[preferred])
    joined=combine(m4,probs[preferred],anchor,mixture["share"])
    fpr=float(baseline[~turns].mean())
    mix_cut=float(np.quantile(joined[~turns],1-fpr,method="higher"))
    radius_spec,shapes,trials,calibrator=select_on_validation(data)
    joblib.dump(calibrator,OUT/"validation_horizon_calibration.joblib",compress=3)
    pd.DataFrame(shapes).to_csv(OUT/"validation_shapes.csv",index=False)
    pd.DataFrame(trials).to_csv(OUT/"validation_widths.csv",index=False)
    pd.DataFrame(rows).to_csv(OUT/"validation_detectors.csv",index=False)
    pd.DataFrame(alarm_trials).to_csv(OUT/"validation_alarm_rules.csv",index=False)
    artifacts=[OUT/"fresh_tree_and_scales.joblib",OUT/"validation_horizon_calibration.joblib",*sorted(OUT.glob("*_fresh.pt"))]
    save(OUT/"frozen_before_test.json",dict(detector=preferred,selection=rows,negative_prior=negatives,online_alarm=alarm,
           optional_mixture=dict(**mixture,anchors=anchor,threshold=mix_cut),m4_threshold=m4_cut,radii=radius_spec,
           weights={p.name:fingerprint(p) for p in artifacts},test_used_to_fit=False,existing_weights_loaded=0,
           selection_protocol="Fixed-radius profile must preserve validation turns previously covered at factors .8, .925 and 1. Extra-conservative width may lose at most one net validation window. Compact width targets 95% overall and by environment while preserving baseline-covered validation turns."))
    print("Independent warning:",preferred,alarm,flush=True)
    print("Selected radius profile:",radius_spec,flush=True)


if __name__=="__main__":
    protect_weights()
    freeze()
