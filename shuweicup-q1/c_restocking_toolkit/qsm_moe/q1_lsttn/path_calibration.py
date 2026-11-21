"""Spherical whole-path CAFHT scores and delayed, predicted-state-specific CPTC-style feedback."""
from collections import deque
import math
import numpy as np

TARGET = .95
INNER_ALPHA = .10


def finite_rank(values,coverage):
    n = len(values)
    k = math.ceil((n+1)*coverage)
    return float(np.partition(values,k-1)[k-1]) if k<=n and k>=1 else float("inf")


def outer(error,base,env,level,pooling):
    # Relative excess of the worst of all 30 three-dimensional balls, as in multiplicative CAFHT.
    scores = np.maximum(0.,(error/base).max(1)-1.)
    factors = [finite_rank(scores,level)]
    if pooling=="max_env":
        factors.extend(finite_rank(scores[env==e],level) for e in np.unique(env))
    return 1+max(factors)


def training_bank(error,radial,prob,form):
    normalized = error/np.maximum(radial,.0001)
    common = np.maximum(np.quantile(normalized,.75,axis=0),.15)
    shape = common
    if form=="stage":
        profiles = []
        for k in range(3):
            group = normalized[prob[:,k]>.30]
            blend = len(group)/(len(group)+75) if len(group)>=25 else 0.
            fitted = np.quantile(group,.75,axis=0) if blend else common
            profiles.append((1-blend)*common+blend*np.maximum(fitted,.15))
        shape = np.stack(profiles)
    scale = np.einsum("nk,kh->nh",prob,shape) if shape.ndim==2 else shape
    scores = (error/(radial*scale)).max(1)
    banks = [np.sort(scores)]
    for k in range(3):
        group = scores[prob[:,k]>.30]
        banks.append(np.sort(group) if len(group)>=25 else banks[0])
    return shape,banks


def save_bank(path,shape,banks):
    np.savez(path,shape=shape,**{"bank"+str(k):v for k,v in enumerate(banks)})


def load_bank(path):
    with np.load(path) as f:
        return f["shape"],[f["bank"+str(k)] for k in range(4)]


def mixed_quantile(ref,history,alpha,local_weight):
    if not history or local_weight==0:
        index = int(np.ceil((1-alpha)*(len(ref)-1)))
        return float(ref[index])
    local = np.fromiter(history,dtype=float)
    fraction = local_weight*len(local)/(len(local)+4.)
    # Weighted empirical CDF combines warm-start and record-local residuals.
    values = np.concatenate((ref,local))
    weights = np.r_[np.full(len(ref),(1-fraction)/len(ref)),np.full(len(local),fraction/len(local))]
    order = values.argsort(kind="stable")
    idx = min(len(values)-1,int(np.searchsorted(np.cumsum(weights[order]),1-alpha)))
    return float(values[order[idx]])


class CalibrationState:
    def __init__(self,recipe,shape,banks):
        self.recipe,self.shape,self.banks = recipe,shape,banks
        self.alpha = np.full(4,INNER_ALPHA)
        self.history = [deque(maxlen=24) for _ in range(4)]
        self.previous_phase = None

    def base(self,radial,phase):
        profile = (phase[:,None]*self.shape).sum(0) if self.shape.ndim==2 else self.shape
        return np.maximum(radial,.0001)*profile

    def provisional(self,radial,phase):
        base = self.base(radial,phase)
        scheme = self.recipe["scheme"]
        if scheme=="static_joint":
            return base
        local = self.recipe["local"]
        if scheme=="cafht":
            factor = mixed_quantile(self.banks[0],self.history[0],self.alpha[0],local)
        else:
            factors = [mixed_quantile(self.banks[k+1],self.history[k+1],self.alpha[k+1],local) for k in range(3)]
            factor = np.sum(phase*factors)
            if self.previous_phase is not None:
                # A change in the issued stage posterior is observable before the next forecast.
                shift = .5*np.abs(phase-self.previous_phase).sum()
                factor *= 1+self.recipe["transition"]*shift
        return base*factor

    def receive(self,error,radial,phase,previous_provisional,valid):
        self.previous_phase = phase.copy()
        if self.recipe["scheme"]=="static_joint" or not np.all(valid):
            return
        score = float((error/self.base(radial,phase)).max())
        miss = float(np.any(error>previous_provisional))
        self.history[0].append(score)
        step = self.recipe["gamma"]*(INNER_ALPHA-miss)
        self.alpha[0] = np.clip(self.alpha[0]+step,.012,.28)
        for k in range(3):
            if phase[k]>.30:
                self.history[k+1].append(score)
            self.alpha[k+1] = np.clip(self.alpha[k+1]+phase[k]*step,.012,.28)


def replay(meta,error,radial,prob,visible,recipe,shape,banks):
    result = np.empty(radial.shape,dtype=np.float64)
    for rows in meta.groupby(["sample","hand"],sort=False).indices.values():
        if not np.array_equal(meta.origin.to_numpy()[rows],10+30*np.arange(len(rows))):
            raise ValueError("Calibration requires the full issue schedule for a single wrist")
        state = CalibrationState(recipe,shape,banks)
        for j,now in enumerate(rows):
            if j:
                past = rows[j-1]
                state.receive(error[past],radial[past],prob[past],result[past],visible[past])
            result[now] = state.provisional(radial[now],prob[now])
    return result


def grid(scheme):
    if scheme=="static_joint":
        return [dict(scheme=scheme,gamma=0.,local=0.,transition=0.)]
    return [dict(scheme=scheme,gamma=g,local=w,transition=t) for g in (.015,.045)
            for w in (0.,.45) for t in ((0.,.20) if scheme=="cafht_cptc" else (0.,))]


def band_stats(error,rad,env,turn):
    hit = (error<=rad).all(1)
    group = float(hit[turn].mean())
    worst = min(float(hit[env==e].mean()) for e in np.unique(env))
    # This margin is a selection heuristic, reported empirically for correlated windows.
    n = len(hit)
    p = hit.mean()
    z = 1.2815515655446004
    lower = (p+z*z/(2*n)-z*np.sqrt(p*(1-p)/n+z*z/(4*n*n)))/(1+z*z/n)
    shortfall = max(0.,TARGET-lower,TARGET-worst,TARGET-group)
    return dict(windows=n,covered=int(hit.sum()),coverage=float(p),turn_covered=int(hit[turn].sum()),
                turns=int(turn.sum()),turn_coverage=group,min_environment=worst,lower90=float(lower),
                radius_cm=float(100*rad.mean()),shortfall=float(shortfall))
