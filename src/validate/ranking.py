"""Ranking power: AUC, Gini, KS, top-decile lift with bootstrap 95% CIs."""
import numpy as np
import pandas as pd
from ..metrics import auc, gini, ks
from .common import SAMPLES, SEED

MODELS = {"scorecard": "pd_scorecard", "lightgbm": "pd_lgbm"}


def lift_top_decile(y, p):
    y = np.asarray(y); k = max(1, len(y) // 10)
    return y[np.argsort(-p)[:k]].mean() / y.mean()


def run(df, n_boot=200):
    rng = np.random.default_rng(SEED); rows = []
    for s in SAMPLES:
        d = df[df["sample"] == s]; y = d["bad"].to_numpy()
        for m, col in MODELS.items():
            p = d[col].to_numpy(); boots = []
            for _ in range(n_boot):
                i = rng.integers(0, len(y), len(y))
                boots.append((auc(y[i], p[i]), ks(y[i], p[i]), lift_top_decile(y[i], p[i])))
            b = np.array(boots); lo, hi = np.percentile(b, [2.5, 97.5], axis=0)
            rows.append({"sample": s, "model": m, "n": len(d), "bad_rate": y.mean(),
                         "auc": auc(y, p), "auc_lo": lo[0], "auc_hi": hi[0],
                         "gini": gini(y, p), "gini_lo": 2 * lo[0] - 1, "gini_hi": 2 * hi[0] - 1,
                         "ks": ks(y, p), "ks_lo": lo[1], "ks_hi": hi[1],
                         "lift_top10": lift_top_decile(y, p), "lift_lo": lo[2], "lift_hi": hi[2]})
    out = pd.DataFrame(rows); out.to_csv("reports/tables/ranking.csv", index=False)
    return out
