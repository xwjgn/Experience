"""Select a point correction without consulting any new test trajectory or test label."""
import numpy as np
import pandas as pd

from config import BASE, OUT, WORK, read_json, save, macro
from motion import Inputs
from phase import decide
from qsm_data import reproducible_torch
from settings import array
from turn_models import FAMILIES, STRENGTHS, TurnExperts, fit_regressions, build_library


def alarms(d):
    cfg=read_json(BASE/"phase_selection.json")
    with np.load(WORK/(cfg["model"]+"_phase.npz")) as f:
        return decide(f["raw"],f["p"],cfg["rule"])==1


def apply_experts(experts,names,d,ids,ref,alert):
    return experts.corrections(names,ref,d.v[ids],d.x[ids],d.phase[ids],d.state[ids],d.cad[ids],
             d.meta.hand.to_numpy()[ids]=="right",d.meta["sample"].to_numpy()[ids],alert[ids])


def metrics(d,ids,pred):
    err=100*np.linalg.norm(pred-d.y[ids],axis=2)
    ade=err.mean(1)
    val=d.val[ids]
    turns=array("next_mode")[ids]==1
    grid=d.grid[ids]
    full=d.mask[ids].all(1)
    te=val & grid & turns
    dense=val & full & turns
    gate=~val & grid & turns
    tail=val & d.mask[ids,-1]
    return dict(turn_cm=float(ade[te].mean()),turn_macro=macro(ade[te],d.env[ids][te]),
                dense_turn_cm=float(ade[dense].mean()),gate_turn_cm=float(ade[gate].mean()),
                overall_ade=macro(ade[val&full],d.env[ids][val&full]),fde=macro(err[tail,-1],d.env[ids][tail]))


def select():
    reproducible_torch()
    d=Inputs()
    ref=np.load(WORK/"reference_center.npy",mmap_mode="r")
    fit_regressions(d,ref)
    lib=build_library(d,ref)
    experts=TurnExperts()
    alert=alarms(d)
    # Gate and validation only. Evaluate every true turn in these partitions, including unalerted turns.
    ids=np.flatnonzero(d.gate | d.val)
    raw=ref[ids]
    correction={k:[] for k in FAMILIES}
    for ix in np.array_split(ids,int(np.ceil(len(ids)/512))):
        pieces=apply_experts(experts,FAMILIES,d,ix,ref[ix],alert)
        for k,v in pieces.items():
            correction[k].append(v.astype("float32"))
    original=metrics(d,ids,raw)
    def row(name,strength,score):
        eligible=(score["overall_ade"]<=original["overall_ade"]+.05 and score["fde"]<=original["fde"]+.20
                  and score["gate_turn_cm"]<=original["gate_turn_cm"]+.10)
        objective=.5*score["turn_cm"]+.25*score["dense_turn_cm"]+.25*score["turn_macro"]
        return dict(family=name,strength=strength,eligible=eligible,objective=objective,**score)
    rows=[row("none",0.,original)]
    for k in FAMILIES:
        delta=np.concatenate(correction[k])
        for strength in STRENGTHS:
            rows.append(row(k,strength,metrics(d,ids,raw+strength*delta)))
    selected=min((r for r in rows if r["eligible"]),key=lambda r:(r["objective"],r["overall_ade"]))
    pd.DataFrame(rows).to_csv(OUT/"point_validation.csv",index=False)
    save("point_lock.json",dict(selected=selected,reference=original,turn_library_windows=len(lib["x"]),
                training_envs=d.roles["fit"],validation_envs=sorted(set(d.env[d.val])),old_checkpoints_loaded=0,
                inference="use only issued turn risk and past kinematics for every wrist origin"))
    # No future state or quality enters correction issuance. Full cache supports unselected origins in causal replay.
    pred=np.empty_like(ref)
    for start in range(0,len(ref),512):
        ix=np.arange(start,min(start+512,len(ref)))
        if selected["family"]=="none":
            pred[ix]=ref[ix]
        else:
            ds=apply_experts(experts,[selected["family"]],d,ix,ref[ix],alert)
            pred[ix]=ref[ix]+selected["strength"]*ds[selected["family"]]
    np.save(WORK/"corrected_center.npy",pred)
    print("Validation reference:",original,flush=True)
    print("Locked point policy:",selected,flush=True)


if __name__=="__main__":
    select()
