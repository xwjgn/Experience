"""Alert thresholds updated only with past, physically observed turn outcomes."""
from collections import deque
import numpy as np
import pandas as pd
from scipy.special import expit,logit
from sklearn.metrics import average_precision_score

from wrist_history import DATA,KEY,M4_DIR,load_record,observe,read,sequence


def combine(m4,score,anchors,share):
    columns = [logit(np.clip(arr,.0001,.9999)) for arr in (m4,score)]
    scaled = [(v-c)/s for v,(c,s) in zip(columns,anchors)]
    return expit((1-share)*scaled[0]+share*scaled[1])


def anchors(m4,score):
    result = []
    for v in (m4,score):
        x = logit(np.clip(v,.0001,.9999))
        lo,mid,hi = np.quantile(x,[.2,.5,.8])
        result.append([float(mid),float(max(.10,hi-lo))])
    return result


def model_utility(y,scores,env):
    def one(label,s):
        k = max(1,round(.25*len(label)))
        top = np.argsort(-s,kind="stable")[:k]
        return .5*average_precision_score(label,s)+.5*label[top].mean()
    local = [one(y[env==g],scores[env==g]) for g in np.unique(env) if y[env==g].any()]
    return float(.65*one(y,scores)+.35*np.mean(local))


class AlarmMemory:
    def __init__(self,negatives,fpr,memory):
        self.original = np.array(negatives,float)
        self.past = deque(maxlen=memory)
        self.fpr = fpr

    def threshold(self):
        if not self.past:
            return float(np.quantile(self.original,1-self.fpr,method="higher"))
        values = np.r_[self.original,list(self.past)]
        weights = np.r_[np.full(len(self.original),20./len(self.original)),np.ones(len(self.past))]
        order = np.argsort(values,kind="stable")
        k = np.searchsorted(np.cumsum(weights[order]),(1-self.fpr)*weights.sum())
        return float(values[order[min(k,len(values)-1)]])

    def receive(self,score,observed,actual_turn):
        if observed and not actual_turn:
            self.past.append(float(score))


ALARM_OPTIONS = tuple((rate,mem) for rate in (.18,.24,.30) for mem in (0,24))


def alarms_on_record(m,grid,score,negatives,options=ALARM_OPTIONS):
    table = {}
    for fpr,memory in options:
        state = AlarmMemory(negatives,fpr,memory)
        cuts = []
        for j,origin in enumerate(grid):
            if j:
                prev=grid[j-1]
                if prev+10 <= origin:
                    state.receive(score[j-1],m.quality[prev:prev+11].all(),m.phase[prev+10]==1)
            cuts.append(state.threshold())
        table[f"cut_{fpr}_{memory}"] = cuts
    return table


def stream(scorer,split,name,negative_prior):
    metadata = pd.read_parquet(DATA/"windows.parquet")
    meta = metadata[metadata.split==split]
    cut = read(M4_DIR/"release_protocol.json")["speed_cut"]
    rows = []
    for (dataset,sample),parts in meta.groupby(["dataset","sample"],sort=False):
        record = load_record(dataset,sample)
        for hand,part in parts.groupby("hand",sort=False):
            movement = observe(record,hand,cut)
            grid = np.arange(10,len(record.t),30)
            probs,_ = scorer.outputs(sequence(movement,grid),movement.tab.iloc[grid],movement.phase[grid])
            alarm_data = alarms_on_record(movement,grid,probs[name],negative_prior)
            table = part.set_index("origin").loc[grid].reset_index()[KEY].copy()
            table["stream_score"] = probs[name]
            rows.append(table.assign(**alarm_data))
    return pd.concat(rows,ignore_index=True)


def alert_metrics(y,alarm):
    y,alarm = np.asarray(y,bool),np.asarray(alarm,bool)
    tp = int((alarm&y).sum())
    n = int(alarm.sum())
    return dict(turns=int(y.sum()),alerts=n,tp=tp,fp=n-tp,recall=tp/max(1,int(y.sum())),
                precision=tp/max(1,n),f1=2*tp/max(1,n+int(y.sum())))


def select_online(y,score,table,baseline):
    base = alert_metrics(y,baseline)
    trials = []
    for fpr,mem in ALARM_OPTIONS:
        cuts = table[f"cut_{fpr}_{mem}"].to_numpy()
        trials.append(dict(fpr=fpr,memory=mem,**alert_metrics(y,score>=cuts)))
    def key(r):
        return (max(0,r["fp"]-base["fp"]),-r["tp"],r["fp"],r["memory"])
    return min(trials,key=key),trials
