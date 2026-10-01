"""Fairness / proxy checks the data allows. The dataset has NO protected characteristics (race, sex, age, ethnicity),
so only proxies are examined: geography (state), borrower count, first-time-buyer status, channel.
Metric: observed / predicted default ratio per group with Wilson CI; large gaps = potential uneven calibration, not proof of bias."""
import pandas as pd
from .calibration import wilson


def run(df):
    rows = []
    for s in ["dev_holdout", "crisis"]:
        d = df[df["sample"] == s]
        for f in ["state", "first_time_buyer", "n_borrowers", "channel", "occupancy"]:
            for lvl, g in d.groupby(f):
                if len(g) < 500: continue
                k = int(g["bad"].sum()); lo, hi = wilson(k, len(g))
                for m, col in [("scorecard", "pd_scorecard"), ("lightgbm", "pd_lgbm")]:
                    rows.append({"sample": s, "feature": f, "level": lvl, "model": m, "n": len(g), "mean_pred": g[col].mean(),
                                 "obs": k / len(g), "ratio": (k / len(g)) / g[col].mean(),
                                 "ratio_lo": lo / g[col].mean(), "ratio_hi": hi / g[col].mean()})
    out = pd.DataFrame(rows); out.to_csv("reports/tables/fairness_proxies.csv", index=False)
    return out
