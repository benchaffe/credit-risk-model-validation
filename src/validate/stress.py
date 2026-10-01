"""Sensitivity: shock inputs and check PDs move in the expected direction; worst-case segments."""
import numpy as np
import pandas as pd
from .common import score_models

SHOCKS = {
    "LTV +10 pts": ({"ltv": 10, "cltv": 10}, +1),
    "Credit score -50": ({"fico": -50}, +1),
    "Interest rate +1pt": ({"int_rate": 1.0}, +1),
    "DTI +10 pts": ({"dti": 10}, +1),
    "Credit score +50": ({"fico": 50}, -1),
}


def run(df):
    rows = []
    for s in ["dev_holdout", "crisis"]:
        base = df[df["sample"] == s]
        for name, (delta, direction) in SHOCKS.items():
            sh = base.copy()
            for c, d in delta.items():
                sh[c] = (sh[c] + d).clip(lower=300 if c == "fico" else None, upper=850 if c == "fico" else None)
            sh = score_models(sh)
            for m, col in [("scorecard", "pd_scorecard"), ("lightgbm", "pd_lgbm")]:
                d = (sh[col] - base[col]).to_numpy() * direction
                rows.append({"sample": s, "shock": name, "model": m, "mean_pd_before": base[col].mean(),
                             "mean_pd_after": sh[col].mean(), "rel_change_pct": 100 * (sh[col].mean() / base[col].mean() - 1),
                             "pct_loans_wrong_direction": 100 * np.mean(d < -1e-12)})
    out = pd.DataFrame(rows); out.to_csv("reports/tables/stress.csv", index=False)
    # worst-case segments: observed/predicted in the crisis sample
    c = df[df["sample"] == "crisis"]; seg = []
    for f in ["state", "channel", "property_type", "loan_purpose", "occupancy", "first_time_buyer", "n_borrowers"]:
        for lvl, g in c.groupby(f):
            if len(g) >= 1000:
                for m, col in [("scorecard", "pd_scorecard"), ("lightgbm", "pd_lgbm")]:
                    seg.append({"feature": f, "level": lvl, "model": m, "n": len(g), "obs": g["bad"].mean(),
                                "pred": g[col].mean(), "ratio": g["bad"].mean() / g[col].mean()})
    seg = pd.DataFrame(seg).sort_values("ratio", ascending=False); seg.to_csv("reports/tables/worst_segments_crisis.csv", index=False)
    return out, seg
