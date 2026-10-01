"""Champion model: WoE-binned logistic regression scorecard.

Fit on `dev_train` only. WoE = ln(%good / %bad), so higher WoE = safer.
Numeric bins: quantile start, adjacent bins merged until minimum size is met and, where business logic
gives a direction (FICO down, LTV/CLTV/DTI/rate up), bad rate is monotonic. Missing is its own bin.
Scaling: 600 points at 50:1 good:bad odds, 20 points to double the odds.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.api as sm
from .metrics import auc, gini, ks

ART = Path("reports/artifacts")
MIN_BIN_FRAC = 0.05
IV_MIN = 0.02
EXPECTED_DIR = {"fico": -1, "ltv": 1, "cltv": 1, "dti": 1, "int_rate": 1}
NUM = ["fico", "ltv", "cltv", "dti", "mi_pct", "orig_upb", "int_rate", "n_units", "n_borrowers"]
CAT = ["first_time_buyer", "occupancy", "channel", "loan_purpose", "property_type", "state"]
PDO, BASE_SCORE, BASE_ODDS = 20, 600, 50


def _woe_iv(good, bad, tg, tb):
    g, b = (good + 0.5) / tg, (bad + 0.5) / tb
    return np.log(g / b), (g - b) * np.log(g / b)


def _merge_bins(x, y, edges, direction, min_n):
    """edges: interior cut points. Merge until each bin >= min_n and monotone (if direction)."""
    edges = list(edges)
    while True:
        idx = np.searchsorted(edges, x, side="right")
        k = len(edges) + 1
        n = np.bincount(idx, minlength=k); bd = np.bincount(idx, weights=y, minlength=k)
        rate = bd / np.maximum(n, 1)
        small = np.where(n < min_n)[0]
        if len(small) and k > 1:
            i = small[0]
            # merge with neighbour of closest bad rate
            cands = [j for j in (i - 1, i + 1) if 0 <= j < k]
            j = min(cands, key=lambda j: abs(rate[j] - rate[i]))
            edges.pop(min(i, j)); continue
        if direction:
            viol = np.where(np.diff(rate) * direction < 0)[0]
            if len(viol):
                edges.pop(viol[0]); continue
        return edges


def fit_bins(train: pd.DataFrame) -> dict:
    y = train["bad"].to_numpy(); tg, tb = (y == 0).sum(), (y == 1).sum()
    min_n = int(MIN_BIN_FRAC * len(train))
    spec = {}
    for f in NUM:
        s = train[f]; ok = s.notna().to_numpy()
        qs = np.unique(np.quantile(s[ok], np.linspace(0, 1, 11)[1:-1]))
        edges = _merge_bins(s[ok].to_numpy(), y[ok], qs, EXPECTED_DIR.get(f), min_n)
        spec[f] = {"type": "num", "edges": [float(e) for e in edges]}
    for f in CAT:
        vc = train[f].value_counts()
        keep = vc[vc >= 0.01 * len(train)].index.tolist()
        spec[f] = {"type": "cat", "levels": keep}
    # woe per bin
    for f, sp in spec.items():
        b = assign_bins(train[f], sp)
        tab = pd.DataFrame({"b": b, "y": y}).groupby("b")["y"].agg(["size", "sum"])
        tab["bad"] = tab["sum"]; tab["good"] = tab["size"] - tab["sum"]
        w, iv = _woe_iv(tab["good"].to_numpy(), tab["bad"].to_numpy(), tg, tb)
        sp["woe"] = {str(k): float(v) for k, v in zip(tab.index, w)}
        sp["iv"] = float(iv.sum())
        sp["stats"] = {str(k): [int(r["size"]), int(r["bad"])] for k, r in tab.iterrows()}
    return spec


def assign_bins(s: pd.Series, sp: dict) -> pd.Series:
    if sp["type"] == "num":
        out = pd.Series(np.searchsorted(sp["edges"], s.to_numpy(dtype=float), side="right"), index=s.index).astype(str)
        return out.mask(s.isna(), "MISSING")
    return s.where(s.isin(sp["levels"]), "OTHER").astype(str)


def woe_matrix(df, spec, feats):
    X = pd.DataFrame(index=df.index)
    for f in feats:
        b = assign_bins(df[f], spec[f])
        X[f] = b.map(spec[f]["woe"]).fillna(0.0)      # unseen bin -> neutral
    return X


def fit(train, spec):
    feats = [f for f in spec if spec[f]["iv"] >= IV_MIN]
    corr = woe_matrix(train, spec, feats).corr()
    if "cltv" in feats and "ltv" in feats and abs(corr.loc["cltv", "ltv"]) > 0.9:
        feats.remove("cltv" if spec["cltv"]["iv"] <= spec["ltv"]["iv"] else "ltv")
    while True:
        X = sm.add_constant(woe_matrix(train, spec, feats)); y = train["bad"]
        m = sm.Logit(y, X).fit(disp=0)
        wrong = [f for f in feats if m.params[f] >= 0]    # WoE coef must be negative on P(bad)
        if not wrong: return m, feats
        feats.remove(min(wrong, key=lambda f: spec[f]["iv"]))


def points_table(m, spec, feats):
    factor = PDO / np.log(2); offset = BASE_SCORE - factor * np.log(BASE_ODDS)
    n = len(feats); pts = {}
    for f in feats:
        pts[f] = {k: float(round(-(m.params[f] * w + m.params["const"] / n) * factor + offset / n, 1))
                  for k, w in spec[f]["woe"].items()}
    return pts


def score(df, spec, pts, feats):
    s = np.zeros(len(df))
    for f in feats:
        b = assign_bins(df[f], spec[f])
        s += b.map(pts[f]).fillna(np.mean(list(pts[f].values()))).to_numpy()
    return s


def pd_from_model(df, spec, m, feats):
    X = sm.add_constant(woe_matrix(df, spec, feats), has_constant="add")
    return m.predict(X).to_numpy()


def main():
    df = pd.read_parquet("data/processed/model_table.parquet")
    tr = df[df["sample"] == "dev_train"]
    spec = fit_bins(tr)
    m, feats = fit(tr, spec)
    pts = points_table(m, spec, feats)
    iv = pd.Series({f: spec[f]["iv"] for f in spec}).sort_values(ascending=False)
    print("IV:\n", iv.round(3).to_string()); print("\nSelected:", feats); print(m.summary().tables[1])
    ART.mkdir(parents=True, exist_ok=True)
    json.dump({"spec": spec, "features": feats, "coef": m.params.to_dict(), "points": pts,
               "scaling": {"pdo": PDO, "base_score": BASE_SCORE, "base_odds": BASE_ODDS}},
              open(ART / "scorecard.json", "w"), indent=1)
    for name in ["dev_train", "dev_holdout"]:
        d = df[df["sample"] == name]; p = pd_from_model(d, spec, m, feats)
        print(f"{name}: AUC={auc(d.bad, p):.3f} Gini={gini(d.bad, p):.3f} KS={ks(d.bad, p):.3f}")


if __name__ == "__main__":
    main()
