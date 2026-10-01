"""Challenger model: LightGBM on the same origination-only features.

Tuning uses ONLY dev_train, with time-aware folds (expanding window over origination year;
each fold trains on earlier years and tests on the next year). dev_holdout is not touched until evaluation.
Categoricals are native LightGBM categoricals; missing numerics stay NaN.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from .features import FEATURES
from .metrics import auc, gini, ks
from .target import SEED

ART = Path("reports/artifacts")
CATS = ["first_time_buyer", "occupancy", "channel", "loan_purpose", "property_type", "state"]
GRID = [
    {"num_leaves": 8, "min_child_samples": 200, "learning_rate": 0.05, "n_estimators": 300},
    {"num_leaves": 16, "min_child_samples": 200, "learning_rate": 0.05, "n_estimators": 300},
    {"num_leaves": 16, "min_child_samples": 500, "learning_rate": 0.03, "n_estimators": 500},
    {"num_leaves": 31, "min_child_samples": 500, "learning_rate": 0.03, "n_estimators": 500},
]


def prep(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURES].copy()
    for c in CATS:
        X[c] = pd.Categorical(X[c], categories=CATEGORY_LEVELS[c]) if CATEGORY_LEVELS else X[c].astype("category")
    return X


CATEGORY_LEVELS: dict = {}


def make(params):
    return lgb.LGBMClassifier(objective="binary", subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                              reg_lambda=5.0, random_state=SEED, verbose=-1, n_jobs=4, **params)


def tune(tr: pd.DataFrame) -> tuple[dict, list]:
    years = sorted(tr["vintage_year"].unique())
    folds = [(years[:i], years[i]) for i in range(3, len(years))]   # expanding window, test on next year
    res = []
    for p in GRID:
        a = []
        for past, nxt in folds:
            a_tr, a_te = tr[tr["vintage_year"].isin(past)], tr[tr["vintage_year"] == nxt]
            m = make(p).fit(prep(a_tr), a_tr["bad"])
            a.append(auc(a_te["bad"], m.predict_proba(prep(a_te))[:, 1]))
        res.append({"params": p, "cv_auc": float(np.mean(a))})
        print(p, round(np.mean(a), 4))
    return max(res, key=lambda r: r["cv_auc"])["params"], res


def main():
    df = pd.read_parquet("data/processed/model_table.parquet")
    for c in CATS:
        CATEGORY_LEVELS[c] = sorted(df[c].unique())
    tr = df[df["sample"] == "dev_train"]
    best, res = tune(tr)
    model = make(best).fit(prep(tr), tr["bad"])
    model.booster_.save_model(str(ART / "lgbm.txt"))
    json.dump({"best_params": best, "cv": res, "category_levels": CATEGORY_LEVELS, "features": FEATURES},
              open(ART / "lgbm_meta.json", "w"), indent=1)
    for name in ["dev_train", "dev_holdout"]:
        d = df[df["sample"] == name]; p = model.predict_proba(prep(d))[:, 1]
        print(f"{name}: AUC={auc(d.bad, p):.3f} Gini={gini(d.bad, p):.3f} KS={ks(d.bad, p):.3f}")


if __name__ == "__main__":
    main()
