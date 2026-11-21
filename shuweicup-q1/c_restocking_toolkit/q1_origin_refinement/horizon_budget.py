"""Select horizon-specific and origin-specific radii using leave-one-environment-out validation."""
from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import QuantileRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from wrist_history import BANDS,STEPS

FAMILIES = ("identity","short_guard","horizon_only","innovation","proxy")
GAMMA = (.55,1.0,1.45)
DEPTHS = (.04,.08,.12,.22,.32,.42)
LATE_WEIGHT = np.clip((STEPS-3)/10.,0.,1.)
OPPORTUNITIES = ("uniform","physical","physical_turn","low_turn","low_both")
CENTRES = np.array([2.,6.5,15.,25.])
LAGS = (3,10,20,30)


@dataclass
class Windows:
    meta: object
    features: object
    score: np.ndarray
    proxy: np.ndarray
    radius: np.ndarray
    error: np.ndarray

    def take(self,mask):
        return Windows(self.meta.loc[mask].reset_index(drop=True),self.features.loc[mask].reset_index(drop=True),
                       self.score[mask],self.proxy[mask],self.radius[mask],self.error[mask])


def design(features,score,proxy,radius,b):
    lo,hi = BANDS[b]
    typical = radius[:,lo:hi].mean(1)
    end = np.maximum(radius[:,hi-1],.0001)
    lag = LAGS[b]
    noise = features[f"noise_{lag}"].to_numpy()
    return np.column_stack((np.log((proxy[:,b]*hi/30+.0005)/end),
        np.log((noise+.0005)/end),np.log1p(features[f"innovation_ratio_{lag}"]),
        np.log(typical+.0001),np.log1p(features.w_speed_5),score,features.current_dwell))


def train_shape(data):
    static = []
    heads = []
    for b,(lo,hi) in enumerate(BANDS):
        ratio = (data.error[:,lo:hi]/data.radius[:,lo:hi]).max(1)
        y = np.log(np.clip(ratio,.012,7.))
        envs = data.meta.env.to_numpy()
        # Give each training environment an equal contribution to the constant reference profile.
        static.append(float(np.median([np.quantile(y[envs==env],.90) for env in np.unique(envs)])))
        x = design(data.features,data.score,data.proxy,data.radius,b)
        local = {}
        for family,idx in (("innovation",[1,2,3,4,5,6]),("proxy",list(range(x.shape[1])))):
            model = make_pipeline(StandardScaler(),QuantileRegressor(quantile=.85,alpha=.035,solver="highs"))
            local[family] = (idx,model.fit(x[:,idx],y))
        heads.append(local)
    return dict(static=np.array(static),heads=heads)


def risk_estimates(model,features,score,proxy,radius):
    n = len(radius)
    static = model["static"]
    values = {k:np.empty((n,4)) for k in ("innovation","proxy")}
    for b in range(4):
        x = design(features,score,proxy,radius,b)
        for kind in values:
            idx,model_b = model["heads"][b][kind]
            values[kind][:,b] = .20*static[b]+.80*model_b.predict(x[:,idx]).clip(-3,1)
    values["horizon_only"] = np.broadcast_to(static,(n,4))
    return values


def rankings(reference,x):
    return np.searchsorted(reference,x,side="right")/max(1,len(reference))


def context(model,features,score,proxy,radius,risk_reference,turn_reference):
    risk = risk_estimates(model,features,score,proxy,radius)
    difficulty = np.max(risk["proxy"],axis=1)
    physical_rank = rankings(risk_reference,difficulty)
    turn_rank = rankings(turn_reference,score)
    clean = (features.w_recent_flags==0).to_numpy()
    # Very high estimated risk leaves the original window budget in place.
    opportunities = {"uniform":clean.astype(float),
        "physical":np.clip((.85-physical_rank)/.60,0,1)*clean,
        "physical_turn":np.clip((.85-physical_rank)/.60,0,1)*np.clip((.90-turn_rank)/.45,0,1)*clean,
        "low_turn":((turn_rank<=.15)&clean).astype(float),
        "low_both":((turn_rank<=.35)&(physical_rank<=.65)&clean).astype(float)}
    return risk,opportunities


def fit_context(data):
    model = train_shape(data)
    fits = risk_estimates(model,data.features,data.score,data.proxy,data.radius)
    model["risk_reference"] = np.sort(np.max(fits["proxy"],axis=1))
    model["turn_reference"] = np.sort(data.score)
    return model


def describe(model,data):
    return context(model,data.features,data.score,data.proxy,data.radius,
                   model["risk_reference"],model["turn_reference"])


def shape_keys():
    return [("identity",0.)]+[(family,g) for family in FAMILIES[1:] for g in GAMMA]


def equal_width_profile(radius,risk,key):
    family,gamma = key
    if family=="identity":
        return np.ones_like(radius)
    if family=="short_guard":
        log_adjust = np.broadcast_to(.72*np.exp(-(STEPS-1)/5.),radius.shape)
    else:
        log_adjust = np.stack([np.interp(np.log(STEPS),np.log(CENTRES),row) for row in risk[family]])
    relative = np.exp(np.clip(gamma*log_adjust,-5,4))
    centred = relative/(np.sum(relative*radius,axis=1)/np.sum(radius,axis=1))[:,None]-1.
    # Linear damping preserves the exact radius budget while bounding per-step changes.
    damp = np.minimum(1.,np.minimum(.22/np.maximum(-centred.min(1),1.e-8),
                                     1.70/np.maximum(centred.max(1),1.e-8)))
    return 1.+damp[:,None]*centred


def path_metrics(data,radius):
    ratios = np.max(data.error/radius,axis=1)
    hit = ratios<=1
    turn = data.meta.turn.to_numpy()
    env = data.meta.env.to_numpy()
    return dict(n=len(hit),covered=int(hit.sum()),coverage=float(hit.mean()),turns=int(turn.sum()),
            turn_covered=int(hit[turn].sum()),turn_coverage=float(hit[turn].mean()),
            radius_cm=float(100*radius.mean()),least_env=float(min(hit[env==e].mean() for e in np.unique(env))),
            ade_cm=float(100*data.error.mean()),turn_ade_cm=float(100*data.error[turn].mean()))


def select_on_validation(val):
    n = len(val.meta)
    keylist = shape_keys()
    allocations = {k:np.ones_like(val.radius) for k in keylist}
    opportunity = {f:np.zeros(n) for f in OPPORTUNITIES}
    for env in val.meta.env.unique():
        hold = val.meta.env.to_numpy()==env
        train,check = val.take(~hold),val.take(hold)
        model = fit_context(train)
        risk,savings = describe(model,check)
        for key in keylist:
            allocations[key][hold] = equal_width_profile(check.radius,risk,key)
        for family,values in savings.items():
            opportunity[family][hold] = values
    truth = val.meta.turn.to_numpy()
    base_ratio = np.max(val.error/val.radius,axis=1)
    # The previous rule chooses one of two widths per window. Validate both widths separately.
    stress_levels = (.8,.925,1.)
    shape_rows = []
    for key,profile in allocations.items():
        np.testing.assert_allclose((profile*val.radius).sum(1),val.radius.sum(1),rtol=1.e-7)
        error_ratio = np.max(val.error/(val.radius*profile),axis=1)
        lost_turns = [int(((base_ratio<=s)&(error_ratio>s)&truth).sum()) for s in stress_levels]
        weighted_misses = sum(int((error_ratio>s).sum()) for s in stress_levels)
        tail_cost = float(np.mean(np.maximum(error_ratio-.75,0)))
        shape_rows.append(dict(family=key[0],gamma=key[1],turns_lost_at_any_budget=max(lost_turns),
                misses_summed_over_budgets=weighted_misses,tail_cost=tail_cost,
                covered_at_80=int((error_ratio<=.8).sum()),covered_at_100=int((error_ratio<=1).sum())))
    safe = [r for r in shape_rows if r["turns_lost_at_any_budget"]==0]
    best = min(safe,key=lambda r:(r["misses_summed_over_budgets"],r["tail_cost"],r["gamma"]))
    chosen = (best["family"],best["gamma"])
    base_hit = base_ratio<=1
    rows = []
    profiles = {"equal":allocations[chosen],"flat":allocations[("identity",0.)]}
    for shape,profile in profiles.items():
        opts = [("none",0.,"flat",np.ones((n,1)))]
        for name,ease in opportunity.items():
            for timing,curve in (("flat",1.),("late",LATE_WEIGHT)):
                opts.extend((name,depth,timing,1-depth*ease[:,None]*curve) for depth in DEPTHS)
        for style,depth,timing,scale in opts:
            radius = val.radius*profile*scale
            hit=(val.error<=radius).all(1)
            metrics=path_metrics(val,radius)
            rows.append(dict(shape=shape,budget=style,depth=depth,timing=timing,new_turn_misses=int((truth & base_hit & ~hit).sum()),
                    newly_lost=int((base_hit & ~hit).sum()),recovered=int((~base_hit & hit).sum()),**metrics))
    selected = {}
    for name,allow_loss in (("reliable",1),("compact",n)):
        eligible = [r for r in rows if r["new_turn_misses"]==0 and r["coverage"]>=.95 and r["turn_coverage"]>=.95
            and r["least_env"]>=.95 and r["covered"]>=int(base_hit.sum())-allow_loss]
        pick = min(eligible,key=lambda r:(round(r["radius_cm"],7),-r["covered"],r["depth"]))
        selected[name] = dict(shape=pick["shape"],budget=pick["budget"],depth=pick["depth"],timing=pick["timing"],validation=pick)
    return dict(shape=list(chosen),shape_validation=best,policies=selected),shape_rows,rows,fit_context(val)


def issue_radii(features,scores,proxy,radius,calibrator,spec,prior_factor=None):
    if radius.ndim!=2 or radius.shape[1]!=30 or np.any(radius<=0) or not np.isfinite(radius).all():
        raise ValueError("Supply 30 positive finite M4 radii per origin.")
    risk,opportunity = context(calibrator,features,scores,proxy,radius,calibrator["risk_reference"],calibrator["turn_reference"])
    profile = equal_width_profile(radius,risk,spec["shape"])
    out = {}
    if prior_factor is not None:
        out["same_window_budget"] = radius*profile*np.asarray(prior_factor)[:,None]
    out["reshaped_m4"] = radius*profile
    for name,policy in spec["policies"].items():
        timing = LATE_WEIGHT if policy["timing"]=="late" else 1.
        factor = 1. if policy["budget"]=="none" else 1-policy["depth"]*opportunity[policy["budget"]][:,None]*timing
        out[name] = radius*factor*(profile if policy["shape"]=="equal" else 1.)
    return out
