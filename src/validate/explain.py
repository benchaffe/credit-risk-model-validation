"""Explainability: SHAP + partial dependence (LightGBM), scorecard reason codes, model agreement."""
import json
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import lightgbm as lgb
from ..challenger import prep
from ..features import FEATURES
from .common import ART, SEED

FIG = "reports/figures/"


def run(df):
    d = df[df["sample"] == "dev_holdout"].sample(5000, random_state=SEED)
    booster = lgb.Booster(model_file=str(ART / "lgbm.txt"))
    X = prep(d)
    sv = shap.TreeExplainer(booster).shap_values(X)
    sv = sv[1] if isinstance(sv, list) else sv
    imp = pd.Series(np.abs(sv).mean(0), index=FEATURES).sort_values(ascending=False)
    imp.to_csv("reports/tables/shap_importance.csv", header=["mean_abs_shap"])
    fig, ax = plt.subplots(figsize=(7, 5)); imp[::-1].plot.barh(ax=ax, color="#3b6ea5")
    ax.set_title("LightGBM mean |SHAP| (log-odds), hold-out sample"); fig.tight_layout(); fig.savefig(FIG + "shap_importance.png", dpi=150)
    # partial dependence on top numeric features
    top = [f for f in imp.index if pd.api.types.is_numeric_dtype(d[f]) and not f.endswith("_missing")][:4]
    fig, axs = plt.subplots(1, 4, figsize=(16, 3.5))
    for ax, f in zip(axs, top):
        grid = np.unique(np.nanquantile(d[f], np.linspace(0.02, 0.98, 20))); pdv = []
        for g in grid:
            Z = d.copy(); Z[f] = g; pdv.append(booster.predict(prep(Z)).mean())
        ax.plot(grid, pdv); ax.set_title(f); ax.set_ylabel("mean PD")
    fig.tight_layout(); fig.savefig(FIG + "partial_dependence.png", dpi=150)
    # scorecard reason codes: points lost vs best bin, top 3 per loan
    sc = json.loads((ART / "scorecard.json").read_text())
    from ..scorecard import assign_bins
    loss = pd.DataFrame(index=d.index)
    for f in sc["features"]:
        pts = d[f].pipe(lambda s: assign_bins(s, sc["spec"][f])).map(sc["points"][f]).fillna(np.mean(list(sc["points"][f].values())))
        loss[f] = max(sc["points"][f].values()) - pts
    rc = loss.apply(lambda r: r.nlargest(3).index.tolist(), axis=1)
    rc_freq = pd.Series([x for l in rc for x in l]).value_counts(normalize=True)
    rc_freq.to_csv("reports/tables/scorecard_reason_codes.csv", header=["share_of_top3_reasons"])
    # agreement
    rho = spearmanr(d["pd_scorecard"], d["pd_lgbm"]).statistic
    common = [f for f in sc["features"]]
    rank_sc = pd.Series({f: sc["spec"][f]["iv"] for f in common}).rank(ascending=False)
    rank_lg = imp.reindex(common).rank(ascending=False)
    agree = spearmanr(rank_sc, rank_lg).statistic
    pd.DataFrame({"score_rank_corr": [rho], "driver_rank_corr": [agree]}).to_csv("reports/tables/model_agreement.csv", index=False)
    return imp, rc_freq, rho, agree
