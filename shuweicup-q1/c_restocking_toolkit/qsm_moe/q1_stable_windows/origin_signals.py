"""Causal wrist motion descriptors shared by offline training and raw-prefix issuance."""
import numpy as np
import pandas as pd


def norm(x):
    return np.linalg.norm(x, axis=-1)


def cosine(a, b):
    return np.clip((a*b).sum(-1)/np.maximum(norm(a)*norm(b), 1e-9), -1., 1.)


def rolling_fraction(series, steps):
    return pd.Series(series, dtype=float).rolling(steps, min_periods=1).mean().to_numpy()


def raw_signals(t, z, head, valid, origins):
    tail = np.arange(len(t))
    distance = np.r_[0., np.cumsum(norm(np.diff(z,axis=0)))]
    speed5 = norm(z-z[np.maximum(tail-5,0)])/np.maximum(t-t[np.maximum(tail-5,0)], .001)
    out = {"valid":valid[origins].astype(float), "recent_valid":rolling_fraction(valid,10)[origins],
           "low_speed_fraction":rolling_fraction(speed5<.10,30)[origins],
           "speed30_spread":pd.Series(speed5).rolling(30,min_periods=1).std(ddof=0).to_numpy()[origins]}
    for lag in (5, 15, 30):
        past = np.maximum(origins-lag,0)
        older = np.maximum(origins-2*lag,0)
        dt = np.maximum(t[origins]-t[past],.001)
        old_dt = np.maximum(t[past]-t[older],.001)
        v = (z[origins]-z[past])/dt[:,None]
        old = (z[past]-z[older])/old_dt[:,None]
        length = distance[origins]-distance[past]
        out["speed"+str(lag)] = norm(v)
        out["velocity_change"+str(lag)] = norm(v-old)
        out["direction_cos"+str(lag)] = cosine(v,old)
        out["straightness"+str(lag)] = np.divide(norm(z[origins]-z[past]), np.maximum(length,.0001)).clip(0,1)
        out["head_speed"+str(lag)] = norm(head[origins]-head[past])/dt
    # Measure the same type of motion twice in the observed prefix. A quiet wrist
    # that has persisted for two windows differs from a wrist that has just stopped.
    for offset,suffix in ((0,"recent"),(30,"prior")):
        end = np.maximum(origins-offset,0)
        positions = np.maximum(end[:,None]-np.arange(30,-1,-1),0)
        path = z[positions]-z[positions[:,:1]]
        anchors = positions[:,::10]
        changes = np.diff(z[anchors],axis=1)
        speeds = norm(changes)/np.maximum(np.diff(t[anchors],axis=1),.001)
        out["excursion_"+suffix] = norm(path).max(1)
        out["speed_cv_"+suffix] = speeds.std(1)/np.maximum(speeds.mean(1),.03)
        out["alignment_"+suffix] = np.minimum(cosine(changes[:,0],changes[:,1]),cosine(changes[:,1],changes[:,2]))
        out["net_ratio_"+suffix] = norm(path[:,-1])/np.maximum(norm(changes).sum(1),.0001)
        out["span_"+suffix] = t[end]-t[positions[:,0]]
    return out


class PastErrors:
    """Only the most recent completed forecast with valid observations contributes a path score."""
    def __init__(self):
        self.count = 0
        self.last_frame = None
        self.last = .6
        self.smooth = .6
        self.endpoint = .5

    def receive(self, frame, error, radius, valid):
        if valid[-1]:
            last = np.clip(error[-1]/max(radius[-1],1e-6),0.,3.)
            self.endpoint = .6*self.endpoint+.4*last
        if np.all(valid):
            score = float(np.clip(np.max(error/np.maximum(radius,1e-6)),0.,3.))
            self.smooth = .70*self.smooth+.30*score if self.count else score
            self.last = score
            self.last_frame = frame
            self.count += 1

    def values(self, frame):
        age = min(frame-self.last_frame,180)/30 if self.last_frame is not None else 6.
        return dict(history_count=min(self.count,8), last_path_ratio=self.last,
                    recent_path_ratio=self.smooth, recent_endpoint_ratio=self.endpoint, history_age=age)


def issue_features(raw, phase, quality, center, radius, past):
    """Center is a displacement; all extra geometric signals are rotation invariant."""
    midpoint = center[:,9]
    end = center[:,-1]
    out = dict(raw)
    out.update(turn_probability=phase[:,1],dwell_probability=phase[:,0],quality_risk=quality,
               step3_radius=radius[:,2],middle_radius=radius[:,14],end_radius=radius[:,-1],
               midpoint_motion=norm(midpoint),end_motion=norm(end),
               forecast_bend=norm(midpoint-end/3.),forecast_shape=cosine(midpoint,end-midpoint))
    out.update({k:np.asarray([row[k] for row in past]) for k in past[0]})
    frame = pd.DataFrame(out,dtype="float32")
    if not np.isfinite(frame.to_numpy()).all():
        raise ValueError("Invalid feature at issuance")
    return frame


def stable_truth(t, origins, truth):
    """Evaluation/training target only. Classify the geometry of an observed future second."""
    ends = truth[:,[9,19,29]]
    segment = np.diff(ends,axis=1,prepend=np.zeros_like(ends[:,:1]))
    times = t[origins[:,None]+np.array([0,10,20,30])]
    speed = norm(segment)/np.maximum(np.diff(times,axis=1),.001)
    near = norm(truth).max(1)<=.03
    uniform = speed.std(1)<=.35*np.maximum(speed.mean(1),.001)
    aligned = (cosine(segment[:,0],segment[:,1])>=np.cos(np.pi/6)) & (cosine(segment[:,1],segment[:,2])>=np.cos(np.pi/6))
    straight = norm(ends[:,-1])>=.93*np.maximum(norm(segment).sum(1),.0001)
    return near | (uniform & aligned & straight)
