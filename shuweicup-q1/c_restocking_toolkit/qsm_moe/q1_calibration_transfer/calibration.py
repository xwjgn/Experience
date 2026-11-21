"""Original record-level endpoint feedback, maximum path score, and fixed-center controls."""
import joblib
import numpy as np
from scipy.optimize import minimize

from transfer_config import REPLICA, WORK
from config import read_json
from intervals import Feedback, adjust_width, band, order_stat, risk_fields
from score import phase_calls
from settings import array

STEPS = np.arange(1, 31)
MARGINAL_ALPHA = .05
LEVELS = (.95, .97, .985, .99)
SHAPES = ("qsm", "gru")
SCHEMES = ("static", "endpoint", "recent", "endpoint_recent")


def freeze_endpoint_feedback(recipe):
    if recipe.get("scheme") not in ("endpoint", "endpoint_recent"):
        return dict(recipe)
    scheme = "recent" if recipe["scheme"]=="endpoint_recent" else "static"
    return {**recipe, "scheme":scheme}


class Panel:
    """One row per scheduled issue, irrespective of future validity. Scoring has its own mask."""
    def __init__(self, d, allowed, widths):
        ids = widths["row"][allowed[widths["row"]]]
        self.ids = ids
        self.meta = d.meta.iloc[ids].reset_index(drop=True)
        self.env = d.env[ids]
        start = np.asarray(array("start"))
        raw_ok = np.asarray(array("ok"))
        ahead = ids[:, None]+STEPS
        safe = ahead.clip(max=len(start)-1)
        self.visible = (ahead<len(start)) & (start[safe]==start[ids, None]) & raw_ok[safe]
        self.complete = d.grid[ids]
        self.turn = np.asarray(array("next_mode"))[ids]==1
        self.phase = phase_calls(d, ids)
        self.center = np.load(WORK/"corrected_center.npy", mmap_mode="r")[ids]
        prior = np.load(WORK/"reference_center.npy", mmap_mode="r")[ids]
        base = np.load(WORK/"reference_scale.npy", mmap_mode="r")[ids]
        self.error = np.linalg.norm(self.center-d.y[ids], axis=2)
        model, mid = joblib.load(WORK/"path_risk_scale.joblib")
        fields = risk_fields(d.phase[ids], d.state[ids], d.q[ids], d.v[ids], self.center, self.center-prior)
        self.shape = dict(qsm=adjust_width("kinematic", base, fields, model, mid),
                          gru=widths["gru"][np.searchsorted(widths["row"], ids)])
        self.base = base
        self.groups = list(self.meta.groupby(["sample","hand"],sort=False).indices.values())
        for rows in self.groups:
            np.testing.assert_array_equal(self.meta.origin.to_numpy()[rows], 10+30*np.arange(len(rows)))


def incumbent(p):
    plan = read_json(REPLICA/"frozen_before_test.json")["interval"]
    shape = p.shape["qsm"] if plan["shape"]=="kinematic" else p.base
    if plan["scheme"]=="static":
        return shape*plan["factors"]["static"]
    with np.load(WORK/(plan["shape"]+"_feedback_priors.npz")) as prior:
        ref = prior["reference"]
        typical = float(prior["typical"])
    radius = np.empty_like(shape)
    for rows in p.groups:
        tracker = Feedback(ref, typical)
        for j, i in enumerate(rows):
            if j:
                old = rows[j-1]
                tracker.observe(p.phase[old], p.error[old], shape[old], p.visible[old])
            endpoint, whole = tracker.multipliers(p.phase[i])
            parts = dict(static=shape[i], endpoint=shape[i]*endpoint, completed=shape[i]*whole)
            radius[i] = band(parts, plan["scheme"], plan["factors"])
    return radius


class RecordFeedback:
    """A separate instance runs for each wrist. Only a newly matured forecast can update it."""
    def __init__(self, reference, typical):
        self.reference = reference
        self.typical = typical
        self.alpha = MARGINAL_ALPHA
        self.fast = 0.
        self.slow = 0.

    def quantile(self, alpha):
        rank = int(np.ceil((len(self.reference)-1)*(1-alpha)))
        return self.reference[rank]

    def observe(self, errors, scale, observed):
        if observed[-1]:
            missed = errors[-1] > scale[-1]*self.quantile(self.alpha)[-1]
            self.alpha = float(np.clip(self.alpha+.04*(MARGINAL_ALPHA-float(missed)), .01, .20))
        if np.all(observed):
            # This additional control tracks local difficulty of completed paths.
            relative = np.max(errors/scale/self.quantile(MARGINAL_ALPHA))/self.typical
            change = np.clip(np.log(max(relative, .01)), -1.7, 1.7)
            self.fast = .68*self.fast+.32*change
            self.slow = .88*self.slow+.12*change

    def radii(self, scale):
        static = scale*self.quantile(MARGINAL_ALPHA)
        endpoint = scale*self.quantile(self.alpha)
        local = np.clip(np.exp(.65*self.slow+.20*self.fast), .65, 2.1)
        return dict(static=static, endpoint=endpoint, recent=static*local, endpoint_recent=endpoint*local)


def horizon_weights(parameters):
    knots = np.log([1, 3, 8, 16, 30])
    return np.exp(np.interp(np.log(STEPS), knots, np.r_[parameters, 0.]))


def fit_horizon_weights(err, base):
    # Minimize the average width of a jointly calibrated band on *gate-training* windows.
    # The final horizon is the anchor. The separate calibration partition fits the scalar multiplier.
    def width(parameters):
        weight = horizon_weights(parameters)
        score = (err/(base*weight)).max(1)
        return base.mean(0).dot(weight)/30*(.5*np.quantile(score,.95)+.5*np.quantile(score,.985))
    baseline = width(np.zeros(4))
    def objective(parameters):
        return width(parameters)/baseline+.012*np.square(parameters).sum()+.008*np.square(np.diff(np.r_[parameters,0.])).sum()
    fitted = minimize(objective, np.zeros(4), method="Powell", bounds=[(-.55,.55)]*4,
                      options={"maxiter":32, "xtol":.004})
    return horizon_weights(fitted.x if objective(fitted.x)<1 else np.zeros(4))


def fit_prior(p, name):
    take = p.complete
    shape = p.shape[name][take]
    ratio = p.error[take]/shape
    ref = np.sort(ratio, axis=0)
    fixed = ref[int(np.ceil((len(ref)-1)*.95))]
    typical = float(np.median((ratio/fixed).max(1)))
    weights = fit_horizon_weights(p.error[take], shape*fixed)
    return dict(reference=ref, typical=typical, weights=weights)


def replay(p, name, prior):
    scales = p.shape[name]
    results = {s:np.empty_like(scales, dtype=np.float64) for s in SCHEMES}
    for rows in p.groups:
        state = RecordFeedback(prior["reference"], prior["typical"])
        for j, i in enumerate(rows):
            if j:
                old = rows[j-1]
                state.observe(p.error[old], scales[old], p.visible[old])
            for scheme, rad in state.radii(scales[i]).items():
                results[scheme][i] = rad
    return results


def outer_factors(error, base, env, level, pooling):
    scores = (error/base).max(1)
    factors = [1., order_stat(scores, level)]
    if pooling=="max_env":
        factors.extend(order_stat(scores[env==name],level) for name in np.unique(env))
    return max(factors)


def load_priors():
    from transfer_config import OUT
    priors = {}
    for shape in SHAPES:
        with np.load(OUT/(shape+"_training_prior.npz")) as f:
            priors[shape] = dict(reference=f["reference"], typical=float(f["typical"]), weights=f["weights"])
    return priors
