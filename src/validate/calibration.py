"""Calibration: predicted vs observed bad rate by score band (bands fixed on dev_train PD deciles),
Wilson 95% CIs, two-sided exact binomial test per band, and overall bias."""
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from .common import SAMPLES
from .ranking import MODELS

N_BANDS = 10


def wilson(k, n, z=1.96):
    p = k / n; d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d; h = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return c - h, c + h


def run(df):
    rows = []
    tr = df[df["sample"] == "dev_train"]
    for m, col in MODELS.items():
        edges = np.quantile(tr[col], np.linspace(0, 1, N_BANDS + 1)[1:-1])
        for s in SAMPLES:
            d = df[df["sample"] == s].copy()
            d["band"] = np.searchsorted(edges, d[col].to_numpy(), side="right") + 1
            for b, g in d.groupby("band"):
                n, k = len(g), int(g["bad"].sum()); lo, hi = wilson(k, n)
                p_val = binomtest(k, n, float(np.clip(g[col].mean(), 1e-9, 1 - 1e-9))).pvalue
                rows.append({"model": m, "sample": s, "band": b, "n": n, "bads": k,
                             "pred_pd": g[col].mean(), "obs_rate": k / n, "obs_lo": lo, "obs_hi": hi,
                             "binom_p": p_val})
    out = pd.DataFrame(rows)
    out["bias"] = np.where(out["binom_p"] < 0.05, np.where(out["obs_rate"] > out["pred_pd"], "under-predicts", "over-predicts"), "ok")
    out.to_csv("reports/tables/calibration_bands.csv", index=False)
    tot = []
    for m, col in MODELS.items():
        for s in SAMPLES:
            d = df[df["sample"] == s]; k, n = int(d["bad"].sum()), len(d)
            tot.append({"model": m, "sample": s, "n": n, "mean_pred": d[col].mean(), "obs_rate": k / n,
                        "ratio_obs_to_pred": (k / n) / d[col].mean(), "binom_p": binomtest(k, n, d[col].mean()).pvalue})
    tot = pd.DataFrame(tot); tot.to_csv("reports/tables/calibration_overall.csv", index=False)
    return out, tot
