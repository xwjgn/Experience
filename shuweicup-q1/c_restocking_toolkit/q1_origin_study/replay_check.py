"""Reconstruct archived decisions using only a raw prefix and already issued radii."""
import numpy as np
import pandas as pd

from evaluate_origin import save
from issue_origin import OriginDecision
from origin_signals import DATA, KEY, M4_DIR, OUT, load_record


def check():
    table = pd.read_csv(OUT/"paired_origins.csv")
    meta = pd.read_parquet(DATA/"windows.parquet").reset_index(names="row")
    tagged = table.merge(meta[KEY+["dataset","row"]],on=KEY,validate="one_to_one")
    selected = pd.concat([tagged.groupby(["env","hand","low_turn_routed"],sort=False).head(1),
                          tagged.groupby(["dataset","hand"],sort=False).head(1)]).drop_duplicates("row")
    model = OriginDecision()
    largest = 0.
    with np.load(M4_DIR/"M3_issued.npz") as issued:
        lookup = pd.Index(issued["row"])
        radius = issued["radius"]
    for row in selected.itertuples():
        original = radius[lookup.get_loc(row.row)]
        raw = load_record(row.dataset,row.sample,row.origin+1)
        new = model.issue(raw,row.hand,original)
        for name,score in new["turn_scores"].items():
            largest = max(largest,abs(score-getattr(row,name)))
        assert (new["low_turn_radius_factor"] < 1) == row.low_turn_routed
        assert (new["tail_radius_factor"] < 1) == row.tail_routed
        np.testing.assert_allclose(new["low_turn_radius_m"],original*(.8 if row.low_turn_routed else 1.),atol=1.e-7,rtol=0)
    assert largest < 1.e-7
    assert selected.env.nunique()==4 and selected.dataset.nunique()==3
    assert set(selected.hand)=={"left","right"}
    saved = pd.read_csv(OUT/"intervals.csv").set_index("policy")
    for name in ("tail","low_turn"):
        hit = table[name+"_hit"]
        assert int(saved.loc[name,"turn_covered"]) == (hit & table.turn).sum()
        assert int(saved.loc[name,"covered"]) == hit.sum()
    save("prefix_replay.json",dict(raw_prefixes=len(selected),environments=selected.env.nunique(),datasets=selected.dataset.nunique(),
              hands=sorted(selected.hand.unique()),max_abs_turn_score_error=largest,both_radius_decisions_match=True))
    print("Verified",len(selected),"online decisions across both hands and four held-out environments.",flush=True)


if __name__=="__main__":
    check()
