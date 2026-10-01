"""Population Stability Index of the score and each feature vs the development-train distribution."""
import numpy as np
import pandas as pd
from ..features import FEATURES

EPS = 1e-4


def psi(expected: pd.Series, actual: pd.Series, n_bins: int = 10) -> float:
    """PSI of `actual` vs `expected`. Numeric: expected deciles (NaN own bin); otherwise categories."""
    if pd.api.types.is_numeric_dtype(expected):
        edges = np.unique(np.nanquantile(expected, np.linspace(0, 1, n_bins + 1)[1:-1]))
        b = lambda s: pd.Series(np.where(s.isna(), -1, np.searchsorted(edges, s.fillna(0).to_numpy(), side="right")))
        e, a = b(expected), b(actual)
    else:
        e, a = expected.reset_index(drop=True), actual.reset_index(drop=True)
    cats = sorted(set(e) | set(a), key=str)
    pe = e.value_counts(normalize=True).reindex(cats).fillna(0).clip(lower=EPS)
    pa = a.value_counts(normalize=True).reindex(cats).fillna(0).clip(lower=EPS)
    return float(((pa - pe) * np.log(pa / pe)).sum())


def flag(v):
    return "stable" if v < 0.10 else ("moderate shift" if v < 0.25 else "significant shift")


def run(df):
    from .common import SAMPLES   # lazy: common pulls in lightgbm
    tr = df[df["sample"] == "dev_train"]; rows = []
    for var in ["pd_scorecard", "pd_lgbm"] + FEATURES:
        for s in SAMPLES[1:]:
            v = psi(tr[var], df[df["sample"] == s][var]); rows.append({"variable": var, "sample": s, "psi": v, "flag": flag(v)})
    out = pd.DataFrame(rows); out.to_csv("reports/tables/psi.csv", index=False)
    return out
