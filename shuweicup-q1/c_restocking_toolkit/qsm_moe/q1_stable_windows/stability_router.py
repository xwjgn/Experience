"""Learn origin-time movement stability and the risk of tightening a QSM prediction band."""
from dataclasses import dataclass
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor

from stable_data import Windows

FAMILIES = ("joint", "agreement", "stable_first", "persistent", "motion")
FRACTIONS = (.12, .25, .40, .55, .70)
FACTORS = (.68, .78, .86, .93)
PROFILES = ("uniform","early_protected")
PROTECTIONS = (1., 1.12)
TURN_CAP = .25
STRESS = .05

def ranks(reference, values):
    return np.searchsorted(reference,values,side="right")/len(reference)


def geometric_score(x):
    # At rest heading is poorly defined. At higher speeds check alignment and velocity change.
    speed = x.speed15.to_numpy()
    rest = np.exp(-np.maximum(x.speed5,x.speed30).to_numpy()/.10)
    alignment = np.clip((x.direction_cos15.to_numpy()+1)/2,0,1)
    uniform = np.exp(-x.velocity_change15.to_numpy()/(.12+speed)) * np.sqrt(alignment)
    return np.maximum(rest,uniform*x.straightness15.to_numpy())


def persistence_score(x):
    scores = []
    for period in ("recent","prior"):
        near = np.exp(-x["excursion_"+period].to_numpy()/.045)
        alignment = np.clip((1+x["alignment_"+period].to_numpy())/2,0,1)
        steady = x["net_ratio_"+period].to_numpy()*alignment*np.exp(-x["speed_cv_"+period].to_numpy())
        scores.append(np.maximum(near,steady))
    full_history = np.minimum(1,x.span_prior.to_numpy()/.7)
    return scores[0]*(.7+.3*scores[1]*full_history)


def fit_weight(env):
    _,ids,n = np.unique(env,return_inverse=True,return_counts=True)
    weight = 1/np.sqrt(n[ids])
    return weight/weight.mean()


def origin_checks(x, veto):
    reliable = (x.valid.to_numpy()>.5) & (x.recent_valid.to_numpy()>=.8)
    abrupt = (x.speed5.to_numpy()>.12) & (x.direction_cos5.to_numpy()<.5)
    recent_miss = (x.history_count.to_numpy()>0) & (x.history_age.to_numpy()<=2) & (x.last_path_ratio.to_numpy()>1.)
    return reliable & ~abrupt & ~recent_miss & (x.turn_probability.to_numpy()<veto), ~reliable | recent_miss


class OriginRouter:
    def __init__(self):
        self.heads = {}
        self.references = {}
        self.thresholds = {}

    def fit(self, training: Windows, gate: Windows, veto):
        self.veto = float(veto)
        self.columns = list(training.x.columns)
        x = training.x.to_numpy()
        weights = fit_weight(training.meta.env.to_numpy())
        targets = {"turn":training.turn,"stable":training.stable,"fragile":training.turn & (training.ratio>.72)}
        for name,label in targets.items():
            model = HistGradientBoostingClassifier(max_iter=85,max_leaf_nodes=7,min_samples_leaf=75,
                        l2_regularization=6,learning_rate=.055,early_stopping=False,random_state=10841)
            model.fit(x,label,sample_weight=weights)
            self.heads[name] = model
        tail = HistGradientBoostingRegressor(loss="quantile",quantile=.90,max_iter=105,max_leaf_nodes=7,
                    min_samples_leaf=75,l2_regularization=6,learning_rate=.05,early_stopping=False,random_state=10847)
        tail.fit(x,np.log1p(np.minimum(training.ratio,4.)),sample_weight=weights)
        self.heads["tail"] = tail
        values = self.predict_heads(gate.x)
        for key,value in values.items():
            self.references[key] = np.sort(value)
        scores = self.scores(gate.x,values)
        # Rank cutoffs use only the distribution of issue-time scores in three gate-training environments.
        for family in (*FAMILIES,"phase_only"):
            self.thresholds[family] = {str(f):float(np.quantile(scores[family],1-f)) for f in FRACTIONS}
        self.high_risk = float(np.quantile(scores["danger"],.85))
        return self

    def predict_heads(self, features):
        x = features[self.columns].to_numpy()
        vals = {}
        for name,m in self.heads.items():
            vals[name] = np.expm1(m.predict(x)) if name=="tail" else m.predict_proba(x)[:,1]
        return vals

    def scores(self, x, values=None):
        values = self.predict_heads(x) if values is None else values
        safe = np.stack([ranks(self.references["stable"],values["stable"]),
                         1-ranks(self.references["turn"],values["turn"]),
                         1-ranks(self.references["fragile"],values["fragile"]),
                         1-ranks(self.references["tail"],values["tail"])])
        weakest = safe.min(0)
        blend = np.average(safe,axis=0,weights=[.34,.18,.22,.26])
        danger = .50*(1-safe[2])+.32*(1-safe[3])+.18*(1-safe[1])
        return {**values,"motion":geometric_score(x),"persistent":persistence_score(x),
                "stable_first":safe[0]*(.8+.2*safe[1:].min(0)),
                "joint":.75*blend+.25*weakest,"agreement":weakest,
                "phase_only":1-x.turn_probability.to_numpy(),"danger":danger}

    def masks(self, x, scores, family, fraction):
        safeguard, unreliable = origin_checks(x,self.veto)
        stable = safeguard & (scores[family]>=self.thresholds[family][str(fraction)])
        protect = ~stable & ((scores["danger"]>=self.high_risk) | unreliable)
        return stable, protect


def multiplier(profile, factor):
    if profile=="early_protected":
        progress = np.clip((np.arange(1,31)-3)/10,0.,1.)
        return 1-(1-factor)*progress
    return np.full(30,factor)


@dataclass(frozen=True)
class Policy:
    family: str="current"
    fraction: float=0.
    factor: float=1.
    profile: str="uniform"
    protection: float=1.
    core_factor: float=1.

    def issue(self, router, x, radius, scores=None):
        if self.family=="current":
            flat = np.zeros(len(x),bool)
            return radius.copy(),flat,flat.copy()
        scores = router.scores(x) if scores is None else scores
        chosen,high = router.masks(x,scores,self.family,self.fraction)
        core = None
        if self.core_factor<self.factor:
            core = chosen & (scores["stable_first"]>=router.thresholds["stable_first"][str(FRACTIONS[0])])
        return self.apply(radius,chosen,high,core), chosen, high & (self.protection>1.)

    def apply(self,radius,chosen,high,core=None):
        changed = radius.astype(np.float64,copy=True)
        changed[chosen] *= multiplier(self.profile,self.factor)
        if core is not None:
            changed[core] = radius[core]*multiplier(self.profile,self.core_factor)
        if self.protection>1.:
            changed[high] *= self.protection
        return changed


def candidates(families=FAMILIES):
    yield Policy()
    for family in families:
        for fraction in FRACTIONS:
            for factor in FACTORS:
                for profile in PROFILES:
                    for protection in PROTECTIONS:
                        yield Policy(family,fraction,factor,profile,protection)
                        if family in ("joint","agreement","stable_first") and fraction>=.25 and factor in (.86,.93):
                            yield Policy(family,fraction,factor,profile,protection,.78 if factor==.93 else .68)
