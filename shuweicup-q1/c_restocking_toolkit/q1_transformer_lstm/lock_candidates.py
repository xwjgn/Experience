"""Fit all scales and decision thresholds without opening any previous study or test outcomes."""
from dataclasses import dataclass
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import precision_recall_curve

from study_paths import DATA,OUT,SEEDS,digest,save
from recurrent_network import MODELS,deterministic
from fit_wrist import Study,center_and_scale,issue_components,load_own,matrix,performance,point_metrics,weights_file
from trajectory_bands import coverage,fit_reference,outer_multiplier,play,recipes

LEVELS = (.95,.96,.97,.975,.98,.985,.99)


def stage_alarm(p,cut):
    return p[:,1]>=cut


def calibrate_quality(d,nets):
    ids=d.rows("gate",stride=1,eligible=False,start=0)
    logits, = issue_components(nets,d.features,ids,quality_only=True)
    x,y = logits.astype(float),d.flag[ids].astype(float)
    def loss(coef):
        h=coef[0]*x+coef[1]
        return np.mean(np.logaddexp(0,h)-y*h)+.0001*(coef[0]-1)**2
    opt=minimize(loss,[1.,np.log(d.flag_rate/(1-d.flag_rate))],bounds=[(.01,12.),(-18.,8.)],method="L-BFGS-B")
    p=expit(opt.x[0]*x+opt.x[1])
    precision,recall,cuts=precision_recall_curve(y,p)
    f1=2*precision[:-1]*recall[:-1]/np.maximum(precision[:-1]+recall[:-1],1e-12)
    return dict(slope=float(opt.x[0]),intercept=float(opt.x[1]),threshold=float(cuts[np.argmax(f1)]),
                gate_positives=int(y.sum()),gate_frames=len(y))


def stage_parameters(d,nets):
    ids=d.rows("gate")
    _,weight,*_ = issue_components(nets,d.features,ids)
    probs=(weight.reshape(len(ids),len(nets),3)*len(nets)).astype(float)
    target=d.stage[ids]
    trials=[]
    for temp in (.75,.9,1.,1.2,1.5):
        softened=probs.clip(1.e-9)**(1/temp)
        p=(softened/softened.sum(2,keepdims=True)).mean(1)
        trials.append((-np.log(p[np.arange(len(p)),target].clip(1.e-9)).mean(),temp,p))
    _,temp,p=min(trials,key=lambda trial:trial[0])
    cuts=[]
    for fraction in (.10,.15,.20,.25,.30,.35):
        cut=float(np.quantile(p[:,1],1-fraction))
        alarm=stage_alarm(p,cut)
        turn=target==1
        cuts.append((2*(alarm&turn).sum()/max(1,alarm.sum()+turn.sum()),-fraction,cut))
    return temp,max(cuts)[-1]


@dataclass
class Outputs:
    ids: np.ndarray
    meta: pd.DataFrame
    complete: np.ndarray
    seen: np.ndarray
    truth: np.ndarray
    stage: np.ndarray
    p: np.ndarray
    weight: np.ndarray
    means: np.ndarray
    sd: np.ndarray
    qlogit: np.ndarray

    def point(self,readout):
        return center_and_scale(self.weight,self.means,self.sd,readout)


def get_outputs(d,role,nets,temp):
    ids=d.rows(role,eligible=False)
    safe=(ids[:,None]+np.arange(1,31)).clip(max=len(d.meta)-1)
    starts=matrix("start")
    seen=(safe-ids[:,None]==np.arange(1,31)) & (starts[safe]==starts[ids,None]) & matrix("good")[safe]
    return Outputs(ids,d.meta.iloc[ids].reset_index(drop=True),d.eligible[ids].all(1),seen,d.y[ids],d.stage[ids],
                   *issue_components(nets,d.features,ids,temp))


def correlation(gate):
    full=gate.complete
    y,means,sd=gate.truth[full],gate.means[full],gate.sd[full]
    std=((y[:,None]-means)/sd[:,:,:,None]).clip(-4,4)
    out=[]
    for k in range(means.shape[1]):
        mode=gate.stage[full]==k%3
        z=std[mode,k] if mode.sum()>=25 else std[:,k]
        a,b=z[:,1:],z[:,:-1]
        r=(a*b).sum(dtype=float)/np.sqrt(np.square(a).sum(dtype=float)*np.square(b).sum(dtype=float))
        out.append(float(np.clip(r,0.,.995)))
    return out


def select_bands(kind,sets,method):
    base={key:v.point(method) for key,v in sets.items()}
    err={key:np.linalg.norm(v.truth-base[key][0],axis=2) for key,v in sets.items()}
    g,c,v=(sets[k] for k in ("gate","cal","validation"))
    policies={}
    trials=[]
    for family in ("fixed","cafht","joint_stage"):
        prior=fit_reference(err["gate"][g.complete],base["gate"][1][g.complete],g.p[g.complete],family=="joint_stage")
        path=OUT/f"{kind}_{family}_reference.npz"
        np.savez(path,**prior)
        options=[]
        for recipe in recipes(family):
            scales={k:play(sets[k].meta,err[k],base[k][1],sets[k].p,sets[k].seen,prior,recipe) for k in ("cal","validation")}
            for level in LEVELS:
                for pool in ("pooled","max_environment"):
                    factor=outer_multiplier(err["cal"][c.complete],scales["cal"][c.complete],c.meta.env.to_numpy()[c.complete],level,pool)
                    if not np.isfinite(factor):
                        continue
                    rad=scales["validation"][v.complete]*factor
                    m=coverage(err["validation"][v.complete],rad,v.stage[v.complete]==1,v.meta.env.to_numpy()[v.complete])
                    spec=dict(**recipe,outer=float(factor),level=level,pool=pool)
                    options.append((spec,m))
                    trials.append(dict(model=kind,**spec,**m))
        def rank(row,per_environment):
            m=row[1]
            deficit=max(0.,.95-m["cov"],.95-m["turn_cov"])
            if per_environment:
                deficit=max(deficit,.97-m["cov"],.95-m["least_env"])
            return deficit,m["radius_cm"],row[0]["eta"]
        spec,metrics=min(options,key=lambda row:rank(row,False))
        policies[family]=dict(recipe=spec,validation=metrics,reference_sha256=digest(path))
        if family=="joint_stage":
            spec,metrics=min(options,key=lambda row:rank(row,True))
            policies["joint_stage_env"]=dict(recipe=spec,validation=metrics,reference_sha256=digest(path))
    return policies,trials


def freeze():
    deterministic()
    d=Study()
    summaries,all_trials,readouts={},[],[]
    for kind in MODELS:
        nets=load_own(kind,SEEDS,d.dims,d.flag_rate,d.device)
        q=calibrate_quality(d,nets)
        temp,cut=stage_parameters(d,nets)
        sets={r:get_outputs(d,r,nets,temp) for r in ("gate","cal","validation")}
        points=[]
        for method in ("mean","median"):
            stats={}
            for key in ("gate","validation"):
                p=sets[key]; chosen,_=p.point(method); use=p.complete
                stats[key]=point_metrics(p.truth[use],chosen[use],p.stage[use],p.meta.env.to_numpy()[use])
            value=.30*performance(stats["gate"])+.70*performance(stats["validation"])
            points.append((value,method,stats["validation"]))
            readouts.append(dict(model=kind,readout=method,objective=value,**stats["validation"]))
        _,method,m=min(points,key=lambda row:row[0])
        bands,rows=select_bands(kind,sets,method)
        all_trials.extend(rows)
        summaries[kind]=dict(seeds=list(SEEDS),params={str(s):digest(weights_file(kind,s)) for s in SEEDS},
                        readout=method,point_validation=m,quality=q,temperature=temp,turn_cut=cut,
                        ar=correlation(sets["gate"]),bands=bands)
        print("Frozen validation",kind,round(m["turn"],3),bands["joint_stage"]["validation"],flush=True)
    pd.DataFrame(all_trials).to_csv(OUT/"validation_bands.csv",index=False)
    pd.DataFrame(readouts).to_csv(OUT/"validation_points.csv",index=False)
    save(OUT/"frozen_before_test.json",dict(models=summaries,dims=d.dims,flag_rate=d.flag_rate,roles=d.roles,
         default_model="M2",enhanced_model="M3",M4="M3 point forecast plus joint_stage",coverage=.95,
         selection="Primary smallest mean validation radius satisfying 95% whole-path and 95% true-turn coverage. Supplementary rule additionally targets 95% in each validation environment and 97% pooled.",
         diagnostics="Fixed/pooled/stage feedback, proper distribution scores and same-mean-width comparison use individually frozen forecasts.",
         partitions={k:len(d.rows(k)) for k in ("fit","gate","cal","validation")},kalman_trained=False,
         old_checkpoint_reads=0,previous_model_inputs=[],data_protocol_sha256=digest(OUT/"release_protocol.json")))


if __name__=="__main__":
    freeze()
