"""Check old-rule equivalence, raw-prefix inference, and paired tail diagnostics."""
import sys
from types import SimpleNamespace
import numpy as np
import pandas as pd

from transfer_config import KIT, OUT, ORIGINAL, PREVIOUS
from calibration import Panel, STEPS, freeze_endpoint_feedback, replay, load_priors
from config import KEY, joined, read_json, write_json
from issue_band import BandForecaster
from motion import Inputs
from select_bands import get_radius
from settings import array


def check_source(val, d, prior):
    # Import the original pure calibration function only. Its training entry point is never called.
    sys.path.insert(0,str(KIT/"literature_experiments"))
    from q1_gru_path_calibration import adaptive_bases
    keys = val.meta.groupby(["sample","hand"],sort=False).indices
    key = max(keys,key=lambda k:len(keys[k]))
    slots = keys[key]
    track = (d.meta["sample"]==key[0]) & (d.meta["hand"]==key[1])
    meta = d.meta[track].reset_index(drop=True)
    positions = val.meta.origin.to_numpy()[slots]
    dense_error = np.zeros((len(meta),30),float)
    dense_scale = np.ones_like(dense_error)
    dense_error[positions] = val.error[slots]
    dense_scale[positions] = val.shape["gru"][slots]
    original = adaptive_bases(meta,dense_error,dense_scale,np.asarray(array("ok"))[track],prior["reference"])
    sample = SimpleNamespace(shape={"gru":val.shape["gru"][slots]}, error=val.error[slots].copy(),
                             visible=val.visible[slots].copy(),groups=[np.arange(len(slots))])
    before = replay(sample,"gru",prior)
    for name,expected in zip(("static","endpoint"),original):
        np.testing.assert_allclose(before[name],expected[positions],atol=3e-7,rtol=0)
    cutoff = int(positions[len(slots)//2]+12)
    future = positions[:,None]+STEPS>cutoff
    sample.error[future] = 1000.
    sample.visible[future] = False
    after = replay(sample,"gru",prior)
    for scheme in before:
        np.testing.assert_array_equal(before[scheme][positions<=cutoff],after[scheme][positions<=cutoff])
    return dict(sample=key[0],hand=key[1],num_issues=len(slots),prefix_through_frame=cutoff,
                original_function_max_difference_m=float(np.abs(before["endpoint"]-original[1][positions]).max()),
                future_edit_changed_earlier_radii=False)


def check_prefix(panel, split, priors, online, online_gru, online_fixed, lock):
    from calibration import incumbent
    from features import rows,wrist,xyz
    from predict import filled
    from scipy.spatial.transform import Rotation
    prev = incumbent(panel)
    replayed = {s:replay(panel,s,prior) for s,prior in priors.items()}
    transfer = get_radius(lock["selected"]["recipe"],panel,prev,replayed,priors)
    fixed = get_radius(freeze_endpoint_feedback(lock["selected"]["recipe"]),panel,prev,replayed,priors)
    gru = get_radius(lock["controls"]["gru_endpoint_95"],panel,prev,replayed,priors)
    audit = []
    for j,e in enumerate(sorted(set(panel.env))):
        origin = panel.meta.origin.to_numpy()
        candidates = np.flatnonzero((panel.env==e)&panel.complete&(origin>=70)&(origin<=160))
        i = candidates[len(candidates)//2]
        record = panel.meta.iloc[i]
        sample,hand,t = record["sample"],record["hand"],int(record["origin"])
        result = online.issue(sample,hand,t)
        assert result["last_read_frame"]==t
        diff = np.abs(result["radius_m"]-transfer[i]).max()
        np.testing.assert_allclose(result["radius_m"],transfer[i],atol=1e-5,rtol=0)
        g = rows(sample,t)
        z = filled(wrist(g,hand),xyz(g,"head")[0])[-1]
        rot = Rotation.from_quat(g[["head_qx","head_qy","head_qz","head_qw"]].iloc[-1]).as_matrix()
        local = (np.array(result["center_world"])-z)@rot
        np.testing.assert_allclose(local,panel.center[i],atol=1e-5,rtol=0)
        gru_max = None
        fixed_max = None
        if j==0:
            alt = online_gru.issue(sample,hand,t)
            np.testing.assert_allclose(alt["radius_m"],gru[i],atol=1e-5,rtol=0)
            gru_max = float(np.max(np.abs(alt["radius_m"]-gru[i])))
            fixed_result = online_fixed.issue(sample,hand,t)
            np.testing.assert_allclose(fixed_result["radius_m"],fixed[i],atol=1e-5,rtol=0)
            fixed_max = float(np.max(np.abs(fixed_result["radius_m"]-fixed[i])))
            off_grid = online.issue(sample,hand,t+7)
            assert off_grid["last_read_frame"]==t+7
            assert off_grid.get("last_feedback_end") in (None,t)
        audit.append(dict(split=split,env=e,sample=sample,hand=hand,origin=t,max_center_difference_m=float(np.abs(local-panel.center[i]).max()),
                          max_selected_radius_difference_m=float(diff),max_literal_gru_difference_m=gru_max,
                          max_fixed_control_difference_m=fixed_max))
    return audit


def tail_analysis():
    old = pd.read_csv(ORIGINAL/"nonoverlap_windows.csv.gz")
    old = old[old.method=="anchored_damped"][KEY+["max_error_cm","joint_adaptive_radius_cm"]]
    qsm = pd.read_csv(PREVIOUS/"paired_windows.csv.gz")
    d = pd.read_parquet(OUT/"qsm_retrained"/"backbone"/"fresh"/"cache"/"meta.parquet")
    with np.load(PREVIOUS/"issued_grid.npz") as f:
        grid = d.iloc[f["row"]][KEY].copy()
        grid["qsm_max_cm"] = 100*np.linalg.norm(f["center"]-f["truth"],axis=2).max(1)
    log = joined(joined(grid,old),qsm[KEY+["turn","v3_radius_cm"]])
    rows = []
    for sp,part in log.groupby("split"):
        for tag,group in (("all",part),("turn",part[part.turn]),("nonturn",part[~part.turn])):
            for model,error,radius in (("original","max_error_cm","joint_adaptive_radius_cm"),
                                       ("current","qsm_max_cm","v3_radius_cm")):
                rows.append(dict(split=sp,group=tag,model=model,windows=len(group),
                                 median_max_error_cm=group[error].median(),p95_max_error_cm=group[error].quantile(.95),
                                 mean_radius_cm=group[radius].mean()))
    pd.DataFrame(rows).to_csv(OUT/"paired_tail_diagnosis.csv",index=False)


def verify():
    d = Inputs()
    priors = load_priors()
    online = BandForecaster("selected")
    online_gru = BandForecaster("gru_endpoint_95")
    online_fixed = BandForecaster("fixed_feedback_control")
    lock = read_json(OUT/"frozen_before_test.json")
    prefix = []
    with np.load(OUT/"issued_scales.npz") as widths:
        val = Panel(d,d.val,widths)
        source = check_source(val,d,priors["gru"])
        for split,panel in (("validation",val),("test",Panel(d,d.test,widths))):
            prefix.extend(check_prefix(panel,split,priors,online,online_gru,online_fixed,lock))
    tail_analysis()
    write_json(OUT/"verification.json",dict(original_feedback_match=source,prefix_issues=prefix))
    print("Verified raw prefixes in",len(prefix),"environments and original source feedback on",source["num_issues"],"issues.")


if __name__=="__main__":
    verify()
