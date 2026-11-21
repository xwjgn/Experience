"""Raw-prefix parity, delayed updates, missing observations, and the stage-bank sample counts."""
import numpy as np
import pandas as pd

from paths_q1 import DATA,OUT,read_json,save_json
from wrist_observations import measured,read_frames
from issue_lsttn import WristForecaster,sample_paths
from freeze_lsttn import panel,readout
from train_lsttn import Study
from path_calibration import replay


def verify():
    d = Study()
    front = WristForecaster()
    audit = []
    meta = d.meta
    with np.load(OUT/"issued_distribution.npz") as issued:
        ids = issued["row"]
        for split in ("validation","test"):
            selected = meta.iloc[ids].reset_index(drop=True)
            for env in sorted(selected.loc[selected.split==split,"env"].unique()):
                subset = selected[(selected.env==env)&(selected.origin>=40)]
                for hand in ("left","right"):
                    group = subset[subset.hand==hand]
                    if not len(group):
                        continue
                    pos = (group.origin-(100 if hand=="left" else 220)).abs().idxmin()
                    row = selected.iloc[pos]
                    sample,o = row["sample"],int(row.origin)
                    data = read_frames(sample,o)
                    out = front.issue_prefix(data,hand)
                    _,(_,z,rotation,mirror) = measured(data,hand)
                    rot = rotation[-1]
                    def local(values):
                        return np.einsum("...j,ji->...i",np.asarray(values)-z[-1],rot)*mirror
                    np.testing.assert_allclose(local(out["center_world"]),issued["center"][pos],rtol=0,atol=1e-5)
                    np.testing.assert_allclose(local(out["component_means_world"]),issued["means"][pos],rtol=0,atol=1e-5)
                    np.testing.assert_allclose(out["component_axis_sigma_m"],issued["sigma"][pos],rtol=0,atol=1e-5)
                    np.testing.assert_allclose(out["radius_m"],issued["radius"][pos],rtol=0,atol=1e-5)
                    np.testing.assert_allclose(out["mixture_weights"],issued["weight"][pos],rtol=0,atol=1e-5)
                    assert abs(sum(out["mixture_weights"])-1)<1e-5 and out["last_read_frame"]==o
                    assert out["last_valid_path_receipt"] is None or out["last_valid_path_receipt"]<=o
                    audit.append(dict(split=split,env=env,sample=sample,hand=hand,origin=o,
                        max_radius_delta_m=float(np.max(np.abs(out["radius_m"]-issued["radius"][pos])))))
    example = audit[0]
    s,hand,cut = example["sample"],example["hand"],example["origin"]
    longer = read_frames(s,cut+32)
    smaller = longer.iloc[:cut+1]
    before,_ = measured(smaller,hand)
    changed = longer.copy()
    suffix = changed.frame_index>cut
    for axis in "xyz":
        changed.loc[suffix,hand+"_j00_"+axis] += 3.
    changed.loc[suffix,hand+"_quality"] = "occlusion"
    for alter in (longer,changed):
        features,_ = measured(alter,hand)
        for k in before:
            np.testing.assert_array_equal(before[k],features[k][:cut+1])
    p = panel(d,"validation",front.nets,front.spec["temperature"])
    center,radial = readout(p,front.spec["point"])
    group = (p.meta["sample"]==s)&(p.meta.hand==hand)
    record = p.meta[group].reset_index(drop=True)
    err = np.linalg.norm(p.y[group]-center[group],axis=-1)
    prob = p.prob[group]
    valids = p.visible[group]
    future = record.origin.to_numpy()[:,None]+np.arange(1,31)>cut
    full = replay(record,err,radial[group],prob,valids,front.recipe,front.shape,front.banks)
    damaged = replay(record,np.where(future,999.,err),radial[group],prob,valids&~future,front.recipe,front.shape,front.banks)
    np.testing.assert_array_equal(full[record.origin<=cut],damaged[record.origin<=cut])
    missing = smaller.copy()
    missing.loc[missing.frame_index>cut-8,[f"{hand}_j00_{a}" for a in "xyz"]] = np.nan
    missing.loc[missing.frame_index>cut-8,hand+"_quality"] = "occlusion"
    result = front.issue_prefix(missing,hand)
    draws = sample_paths(result,16,13487)
    np.testing.assert_array_equal(draws,sample_paths(result,16,13487))
    assert draws.shape==(16,30,3) and np.isfinite(draws).all() and np.all(np.array(result["radius_m"])>0)
    assert front.issue(s,hand,10)["last_valid_path_receipt"] is None
    between = front.issue(s,hand,47)
    assert between["last_read_frame"]==47 and between["last_valid_path_receipt"] in (None,40)
    gate = panel(d,"gate",front.nets,front.spec["temperature"])
    support = (gate.prob[gate.full]>.30).sum(0).tolist()
    save_json(OUT/"verification.json",dict(raw_prefixes=audit,quality_after_eight_missing_ticks="finite",
        marker="suffix wrist/quality and future score changes leave earlier features and intervals identical",
        source_body="GeoX-Lab LSTTN commit 68185aa8, freshly instantiated patch embedding and temporal positional encoding",
        gate_count=int(gate.full.sum()),warm_state_counts=support,fallback_below=25,raw_mixture_samples_checked=16,
        largest_online_radius_delta_m=max(v["max_radius_delta_m"] for v in audit)))
    print("Verified",len(audit),"raw prefixes in",len({v["env"] for v in audit}),"environments; warm state counts",support,flush=True)


if __name__=="__main__":
    verify()
