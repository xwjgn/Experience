"""Delayed adaptive intervals for complete 3D paths. All quantities are in metres."""
from collections import deque
import math
import numpy as np

INNER_MISS = .12


def wpercentile(x,w,q):
    order = np.argsort(x,kind="stable")
    cumulative = np.cumsum(w[order],dtype=float)
    idx = min(len(x)-1,np.searchsorted(cumulative,q*cumulative[-1],side="left"))
    return float(x[order[idx]])


def fit_reference(error,scale,prob,conditional):
    ratio = error/np.maximum(scale,.0001)
    base = np.maximum(np.quantile(ratio,.70,axis=0),.12)
    profiles = []
    weight = []
    for k in range(3):
        w = prob[:,k].astype(float)**2
        neff = w.sum()**2/max(np.square(w).sum(),1e-8)
        if neff<25 or prob[:,k].sum()<20:
            profiles.append(base); weight.append(np.ones(len(error)))
            continue
        blend = neff/(neff+70)
        fitted = np.array([wpercentile(col,w,.70) for col in ratio.T])
        profiles.append((1-blend)*base+blend*fitted.clip(.12))
        weight.append(w)
    shape = np.stack(profiles) if conditional else np.tile(base,(3,1))
    local = np.einsum("nk,kh->nh",prob,shape)
    scores = (error/(scale*local)).max(1)
    order = np.argsort(scores,kind="stable")
    weights = np.stack([np.ones(len(error)),*weight])[:,order]
    weights /= weights.sum(1)[:,None]
    return dict(shape=shape,values=scores[order],weight=weights)


def recipes(family):
    if family=="fixed":
        return [dict(family=family,eta=0.,memory=0.,switch=0.)]
    return [dict(family=family,eta=e,memory=m,switch=b) for e in (.018,.05) for m in (0.,.32)
            for b in ((0.,.18) if family=="joint_stage" else (0.,))]


class BandMemory:
    def __init__(self,prior,recipe):
        self.prior,self.recipe = prior,recipe
        self.cdf = np.cumsum(prior["weight"],axis=1)
        self.alpha = np.full(4,INNER_MISS,dtype=float)
        self.past = deque(maxlen=22)
        self.previous_phase = None

    def template(self,scale,p):
        return np.maximum(scale,.0001)*np.einsum("k,kh->h",p,self.prior["shape"])

    def quantile(self,k):
        level = 1-self.alpha[k]
        values = self.prior["values"]
        local = self.recipe["memory"]
        if not self.past or local==0:
            return values[min(len(values)-1,np.searchsorted(self.cdf[k],level))]
        data = np.array([r[0] for r in self.past])
        weights = np.ones(len(data)) if k==0 else np.array([r[1][k-1] for r in self.past])
        mass = local*weights.sum()/(4+weights.sum())
        mixed = np.r_[(1-mass)*self.prior["weight"][k],mass*weights/max(1e-8,weights.sum())]
        return wpercentile(np.r_[values,data],mixed,level)

    def issue(self,scale,p):
        base = self.template(scale,p)
        if self.recipe["family"]=="fixed":
            return base
        if self.recipe["family"]=="cafht":
            return base*self.quantile(0)
        mult = sum(p[k]*self.quantile(k+1) for k in range(3))
        shift = .5*np.abs(p-self.previous_phase).sum() if self.previous_phase is not None else 0.
        return base*mult*(1+self.recipe["switch"]*shift)

    def receive(self,error,scale,p,issued,observed):
        self.previous_phase = p.copy()
        if self.recipe["family"]=="fixed" or not np.all(observed):
            return
        self.past.append((float(np.max(error/self.template(scale,p))),p.copy()))
        step = self.recipe["eta"]*(INNER_MISS-float(np.any(error>issued)))
        self.alpha = np.clip(self.alpha+step*np.r_[1.,p],.015,.30)


def play(meta,error,scale,prob,seen,prior,recipe):
    result = np.empty_like(scale,dtype=float)
    for group in meta.groupby(["sample","hand"],sort=False).indices.values():
        np.testing.assert_array_equal(meta.origin.to_numpy()[group],10+30*np.arange(len(group)))
        state = BandMemory(prior,recipe)
        for j,now in enumerate(group):
            if j:
                old=group[j-1]
                state.receive(error[old],scale[old],prob[old],result[old],seen[old])
            result[now]=state.issue(scale[now],prob[now])
    return result


def finite_quantile(x,q):
    idx=math.ceil((len(x)+1)*q)-1
    return float(np.partition(x,idx)[idx]) if 0<=idx<len(x) else np.inf


def outer_multiplier(error,base,env,level,pool):
    extra = np.maximum(0.,(error/base).max(1)-1.)
    vals = [finite_quantile(extra,level)]
    if pool=="max_environment":
        vals.extend(finite_quantile(extra[env==g],level) for g in np.unique(env))
    return 1+max(vals)


def coverage(err,band,turn,env):
    hit = (err<=band).all(1)
    return dict(windows=len(hit),covered=int(hit.sum()),cov=float(hit.mean()),turns=int(turn.sum()),
            turn_covered=int(hit[turn].sum()),turn_cov=float(hit[turn].mean()),radius_cm=float(band.mean()*100),
            least_env=float(min(hit[env==e].mean() for e in np.unique(env))))
