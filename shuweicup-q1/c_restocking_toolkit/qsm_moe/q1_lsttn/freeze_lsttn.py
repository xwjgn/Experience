from dataclasses import dataclass
import numpy as np
import pandas as pd

from paths_q1 import DATA,OUT,DEVICES,digest,save_json
from lsttn_model import deterministic
from train_lsttn import Study,checkpoint,criterion,load_nets,point_stats,predict,summarize
from path_calibration import band_stats,grid,outer,replay,save_bank,training_bank

EXPERIMENTS = {"quality_stage":("quality_stage",DEVICES),"quality_stage_one":("quality_stage",DEVICES[:1]),
               **{v:(v,DEVICES[:1]) for v in ("short_only","no_quality","single_gaussian")}}
LEVELS = (.95,.96,.97,.975,.98,.9825,.985,.99)


@dataclass
class Panel:
    ids: np.ndarray
    meta: pd.DataFrame
    full: np.ndarray
    y: np.ndarray
    mode: np.ndarray
    visible: np.ndarray
    prob: np.ndarray
    weight: np.ndarray
    means: np.ndarray
    sigma: np.ndarray


def panel(d,role,nets,temp):
    ids = d.rows(role,complete=False)
    meta = d.meta.iloc[ids].reset_index(drop=True)
    ahead = ids[:,None]+np.arange(1,31)
    safe = np.minimum(ahead,len(d.meta)-1)
    starts = np.load(DATA/"start.npy",mmap_mode="r")
    visible = (ahead<len(d.meta)) & (starts[safe]==starts[ids,None]) & d.ok[safe]
    return Panel(ids,meta,d.valid[ids].all(1),d.y[ids],d.mode[ids],visible,*predict(nets,d.inputs,ids,temp))


def readout(p,kind):
    if kind!="dominant_stage":
        return summarize(p.weight,p.means,p.sigma,kind)
    if p.weight.shape[1]%3!=0:
        return summarize(p.weight,p.means,p.sigma,"mean")
    stages = np.tile(np.arange(3),p.weight.shape[1]//3)
    inside = stages[None]==p.prob.argmax(1)[:,None]
    w = p.weight*inside
    center = ((w/w.sum(1,keepdims=True))[:,:,None,None]*p.means).sum(1)
    spread = (p.weight[:,:,None]*(3*p.sigma**2+np.square(p.means-center[:,None]).sum(-1))).sum(1)
    return center,np.sqrt(np.maximum(spread,1e-8))


def validate_point(p,kind):
    center,_ = readout(p,kind)
    use = p.full
    return point_stats(p.y[use],center[use],p.mode[use],p.meta.env.to_numpy()[use])


def fit_stage_readout(d,nets):
    ids = d.rows("gate")
    target = d.mode[ids]
    records = []
    for temp in (.70,.85,1.,1.20,1.45):
        prob,*_ = predict(nets,d.inputs,ids,temp)
        nll = -np.log(np.maximum(prob[np.arange(len(ids)),target],1e-9)).mean()
        records.append((float(nll),temp,prob))
    _,temp,prob = min(records,key=lambda r:(r[0],abs(1-r[1])))
    t = target==1
    options = []
    for frac in (.10,.15,.20,.25,.30,.35):
        cut = float(np.quantile(prob[:,1],1-frac))
        alarm = prob[:,1]>=cut
        tp = int((alarm&t).sum())
        f1 = 2*tp/max(1,int(alarm.sum()+t.sum()))
        options.append((f1,-frac,cut))
    return temp,max(options)[-1]


def calibrate(panels,name,kind):
    summaries = {key:readout(p,kind) for key,p in panels.items()}
    errs = {key:np.linalg.norm(p.y-summaries[key][0],axis=-1) for key,p in panels.items()}
    gate,cal,val = (panels[key] for key in ("gate","cal","validation"))
    priors = {form:training_bank(errs["gate"][gate.full],summaries["gate"][1][gate.full],gate.prob[gate.full],form)
              for form in ("global","stage")}
    for form,prior in priors.items():
        save_bank(OUT/(name+"_"+form+"_prior.npz"),*prior)
    results = {}
    rows = []
    for family in ("static_joint","cafht","cafht_cptc"):
        trials = []
        forms = ("global","stage") if family=="cafht_cptc" else ("global",)
        for recipe in [dict(**r,profile=f) for r in grid(family) for f in forms]:
            shape,banks = priors[recipe["profile"]]
            base = {key:replay(p.meta,errs[key],summaries[key][1],p.prob,p.visible,recipe,shape,banks)
                    for key,p in (("cal",cal),("validation",val))}
            for level in LEVELS:
                for pooling in ("pooled","max_env"):
                    factor = outer(errs["cal"][cal.full],base["cal"][cal.full],cal.meta.env.to_numpy()[cal.full],level,pooling)
                    if not np.isfinite(factor):
                        continue
                    rad = base["validation"][val.full]*factor
                    stats = band_stats(errs["validation"][val.full],rad,val.meta.env.to_numpy()[val.full],val.mode[val.full]==1)
                    fitted = dict(**recipe,outer_factor=factor,level=level,pooling=pooling)
                    trials.append((fitted,stats))
                    rows.append(dict(variant=name,**fitted,**stats))
        best = min(trials,key=lambda row:(row[1]["shortfall"],row[1]["radius_cm"],row[0]["gamma"],row[0]["local"]))
        results[family] = dict(recipe=best[0],validation=best[1])
        if family=="cafht_cptc":
            pooled = min(trials,key=lambda t:(max(0.,.95-t[1]["coverage"],.95-t[1]["turn_coverage"]),t[1]["radius_cm"]))
            results["cafht_cptc_pooled_target"] = dict(recipe=pooled[0],validation=pooled[1])
            def buffer_rank(trial):
                m = trial[1]
                deficit = max(m["shortfall"],.98-m["coverage"],.96-m["min_environment"],.97-m["turn_coverage"])
                return deficit,m["radius_cm"]
            guarded = min(trials,key=buffer_rank)
            results["cafht_cptc_buffer"] = dict(recipe=guarded[0],validation=guarded[1])
    return results,rows


def freeze():
    deterministic()
    d = Study()
    choice = {}
    all_bands,all_points = [],[]
    for name,(variant,seeds) in EXPERIMENTS.items():
        nets = load_nets(variant,seeds,d.dims,d.device)
        temp,threshold = fit_stage_readout(d,nets)
        panels = {role:panel(d,role,nets,temp) for role in ("gate","cal","validation")}
        kinds = ("mean","smooth_median") if variant=="single_gaussian" else ("mean","smooth_median","dominant_stage")
        points = []
        for kind in kinds:
            val = validate_point(panels["validation"],kind)
            gate = validate_point(panels["gate"],kind)
            objective = .7*criterion(val)+.3*criterion(gate)
            points.append((objective,kind,val))
            all_points.append(dict(variant=name,readout=kind,objective=objective,**val))
        _,kind,val = min(points,key=lambda r:r[0])
        choices,rows = calibrate(panels,name,kind)
        all_bands.extend(rows)
        choice[name] = dict(variant=variant,seeds=list(seeds),temperature=temp,alert_threshold=threshold,
                  point=kind,point_validation=val,intervals=choices,
                  prior_sha256={form:digest(OUT/(name+"_"+form+"_prior.npz")) for form in ("global","stage")},
                  checkpoint_sha256={str(seed):digest(checkpoint(variant,seed)) for seed in seeds})
        print("Validation",name,kind,round(val["turn"],3),choices["cafht_cptc"]["validation"],flush=True)
    pd.DataFrame(all_bands).to_csv(OUT/"validation_band_candidates.csv",index=False)
    pd.DataFrame(all_points).to_csv(OUT/"validation_point_candidates.csv",index=False)
    save_json(OUT/"frozen_before_test.json",dict(experiments=choice,dims=list(d.dims),primary="quality_stage",
       primary_calibration="cafht_cptc",target=.95,roles=d.roles,previous_checkpoints_loaded=0,
       point_criterion="0.7 validation + 0.3 training-gate: 0.45 turn ADE + 0.4 environment macro ADE + 0.15 environment macro FDE",
       stage="Training gate fits probability temperature and a separate alarm threshold. Soft probabilities operate the path mixture and calibration.",
       complete_window_counts={k:len(d.rows(k)) for k in ("fit","gate","cal","validation")},
       selection="Within each calibration family minimize width subject to validation whole-path, turn and per-environment coverage of 95%, plus a descriptive pooled margin. No test fitting.",
       comparison="Previously reported four official test environments. Archived competitors joined by split/environment/record/hand/issue frame only during scoring."))


if __name__=="__main__":
    freeze()
