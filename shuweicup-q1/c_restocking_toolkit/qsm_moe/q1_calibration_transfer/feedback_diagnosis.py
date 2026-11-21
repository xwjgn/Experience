"""Post-lock intervention: keep fitted scales, horizon weights, multiplier and floor fixed; freeze feedback."""
import numpy as np
import pandas as pd

from transfer_config import OUT
from calibration import Panel, RecordFeedback, freeze_endpoint_feedback, incumbent, load_priors, replay
from config import read_json
from motion import Inputs
from select_bands import get_radius


def diagnose():
    lock = read_json(OUT/"frozen_before_test.json")
    recipe = lock["selected"]["recipe"]
    if recipe.get("scheme") not in ("endpoint","endpoint_recent"):
        pd.DataFrame(columns=["split","group","windows","with_feedback_cm","without_feedback_cm"]).to_csv(OUT/"feedback_diagnosis.csv",index=False)
        return
    d = Inputs()
    priors = load_priors()
    source = recipe["shape"]
    prior = priors[source]
    rows = []
    with np.load(OUT/"issued_scales.npz") as saved:
        for split,allowed in (("validation",d.val),("test",d.test)):
            p = Panel(d,allowed,saved)
            ref = incumbent(p)
            replays = {source:replay(p,source,prior)}
            off = freeze_endpoint_feedback(recipe)
            with_feedback = get_radius(recipe,p,ref,replays,priors)
            without = get_radius(off,p,ref,replays,priors)
            alpha = np.empty(len(p.ids))
            for track in p.groups:
                state = RecordFeedback(prior["reference"],prior["typical"])
                for j,i in enumerate(track):
                    if j:
                        old = track[j-1]
                        state.observe(p.error[old],p.shape[source][old],p.visible[old])
                    alpha[i] = state.alpha
            splits = {"all":p.complete,"turn":p.complete&p.turn,"nonturn":p.complete&~p.turn}
            splits.update({e:p.complete & (p.env==e) for e in np.unique(p.env)})
            for group,mask in splits.items():
                a,b = with_feedback[mask],without[mask]
                truth = p.error[mask]
                rows.append(dict(split=split,group=group,windows=int(mask.sum()),with_feedback_cm=100*a.mean(),
                                 without_feedback_cm=100*b.mean(),with_feedback_covered=int((truth<=a).all(1).sum()),
                                 without_feedback_covered=int((truth<=b).all(1).sum()),
                                 alpha_below_initial=float((alpha[mask]<.05-1e-10).mean()),
                                 alpha_at_lower_limit=float((alpha[mask]<=.01+1e-10).mean())))
    pd.DataFrame(rows).to_csv(OUT/"feedback_diagnosis.csv",index=False)


if __name__=="__main__":
    diagnose()
