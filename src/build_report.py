"""Fill reports/templates/*.md.tmpl with numbers computed from reports/tables and reports/artifacts.
Every number in the report text comes from here, so `make all` regenerates a consistent report.
Placeholders use Python str.format syntax, e.g. {auc_ho_s:.3f}."""
import json
from pathlib import Path
import pandas as pd

T, ART, TPL = Path("reports/tables"), Path("reports/artifacts"), Path("reports/templates")
SMP = {"dev_train": "tr", "dev_holdout": "ho", "out_of_time": "oot", "crisis": "cr", "recent": "rec"}
MOD = {"scorecard": "s", "lightgbm": "l"}
SHOCK = {"Credit score -50": "fico_dn", "Credit score +50": "fico_up", "LTV +10 pts": "ltv", "DTI +10 pts": "dti", "Interest rate +1pt": "rate"}


def pct(x, d=2): return f"{100 * x:.{d}f}%"


def numbers() -> dict:
    n = {}
    rk, co = pd.read_csv(T / "ranking.csv"), pd.read_csv(T / "calibration_overall.csv")
    for _, r in rk.iterrows():
        k = f"{SMP[r['sample']]}_{MOD[r['model']]}"
        for c in ["auc", "auc_lo", "auc_hi", "gini", "ks", "lift_top10"]:
            n[f"{c}_{k}"] = r[c]
        n[f"n_{SMP[r['sample']]}"] = int(r["n"]); n[f"bad_{SMP[r['sample']]}"] = r["bad_rate"] * 100
    for _, r in co.iterrows():
        k = f"{SMP[r['sample']]}_{MOD[r['model']]}"
        n[f"ratio_{k}"] = r["ratio_obs_to_pred"]; n[f"pred_{k}"] = r["mean_pred"] * 100; n[f"p_{k}"] = r["binom_p"]
        n[f"obs_{SMP[r['sample']]}"] = r["obs_rate"] * 100
    for m in "sl":
        for x in ["oot", "cr", "rec"]:
            n[f"dauc_{x}_{m}"] = n[f"auc_ho_{m}"] - n[f"auc_{x}_{m}"]
        n[f"gap_{m}"] = n[f"auc_tr_{m}"] - n[f"auc_ho_{m}"]
    for x in ["oot", "cr", "rec"]:
        n[f"badx_{x}"] = n[f"bad_{x}"] / n["bad_ho"]
    n["auc_diff_cr"] = n["auc_cr_l"] - n["auc_cr_s"]; n["auc_diff_oot"] = n["auc_oot_l"] - n["auc_oot_s"]
    # calibration bands (crisis, scorecard)
    b = pd.read_csv(T / "calibration_bands.csv"); g = b[(b.model == "scorecard") & (b["sample"] == "crisis")]
    n.update(band1_pred=g.iloc[0].pred_pd * 100, band1_obs=g.iloc[0].obs_rate * 100, band10_pred=g.iloc[-1].pred_pd * 100, band10_obs=g.iloc[-1].obs_rate * 100)
    n["bands_under_cr"] = int((g.bias == "under-predicts").sum()); n["bands_total"] = len(g)
    h = b[(b["sample"] == "dev_holdout") & (b.model == "scorecard")]; n["bands_off_ho"] = int((h.bias != "ok").sum())
    # stress
    s = pd.read_csv(T / "stress.csv")
    for _, r in s.iterrows():
        k = f"{SHOCK[r['shock']]}_{MOD[r['model']]}_{SMP[r['sample']]}"
        n[f"rel_{k}"] = r["rel_change_pct"]; n[f"wrong_{k}"] = r["pct_loans_wrong_direction"]
    # segments
    w = pd.read_csv(T / "worst_segments_crisis.csv"); w = w[w.model == "scorecard"]
    for _, r in w.iterrows():
        n[f"seg_{r['feature']}_{r['level']}"] = r["ratio"]; n[f"segn_{r['feature']}_{r['level']}"] = int(r["n"])
    ws = w[w.feature == "state"].sort_values("ratio"); n["best_state1"], n["best_state1_r"] = ws.iloc[0]["level"], ws.iloc[0]["ratio"]
    n["best_state2"], n["best_state2_r"] = ws.iloc[1]["level"], ws.iloc[1]["ratio"]
    # psi
    p = pd.read_csv(T / "psi.csv")
    for _, r in p.iterrows():
        n[f"psi_{r['variable']}_{SMP[r['sample']]}"] = r["psi"]
    # explain
    sh = pd.read_csv(T / "shap_importance.csv", index_col=0)["mean_abs_shap"]
    for i, (f, v) in enumerate(sh.head(6).items(), 1):
        n[f"shap{i}_name"], n[f"shap{i}"] = f, v
    n["shap_rank_state"] = list(sh.index).index("state") + 1
    rc = pd.read_csv(T / "scorecard_reason_codes.csv", index_col=0)["share_of_top3_reasons"]
    for f, v in rc.items(): n[f"rc_{f}"] = v * 100
    n["rc_top1"], n["rc_top1_v"] = rc.index[0], rc.iloc[0] * 100
    ag = pd.read_csv(T / "model_agreement.csv").iloc[0]; n["rho_score"], n["rho_driver"] = ag.score_rank_corr, ag.driver_rank_corr
    # fairness
    f = pd.read_csv(T / "fairness_proxies.csv"); f = f[f.model == "scorecard"]
    for _, r in f[f.feature != "state"].iterrows():
        k = f"fair_{r['feature']}_{r['level']}_{SMP[r['sample']]}"
        n[k], n[k + "_lo"], n[k + "_hi"] = r["ratio"], r["ratio_lo"], r["ratio_hi"]
    # covid
    c = pd.read_csv(T / "covid_check.csv").set_index("vintage")
    for v, r in c.iterrows():
        n[f"cv{v}_pct"], n[f"cv{v}_unadj"], n[f"cv{v}_adj"], n[f"cv{v}_forb"], n[f"cv{v}_bad"] = r.pct_from_2020_03, r.rate_unadjusted_pct, r.rate_adjusted_pct, int(r.with_forbearance), int(r.bad_unadjusted)
    # scorecard / lgbm metadata
    sc = json.loads((ART / "scorecard.json").read_text()); n["n_features_sc"] = len(sc["features"])
    for k, v in sc["spec"].items(): n[f"iv_{k}"] = v["iv"]
    tiny = min(((int(v[0]), v[1] / v[0], f, b) for f, sp in sc["spec"].items() if f in sc["features"] for b, v in sp["stats"].items()), key=lambda t: t[0])
    n["tiny_n"], n["tiny_rate"], n["tiny_feat"], n["tiny_bin"] = tiny[0], tiny[1] * 100, tiny[2], tiny[3]
    meta = json.loads((ART / "lgbm_meta.json").read_text()); best = max(meta["cv"], key=lambda r: r["cv_auc"])
    n.update(cv_auc=best["cv_auc"], lg_leaves=best["params"]["num_leaves"], lg_trees=best["params"]["n_estimators"], lg_lr=best["params"]["learning_rate"], n_inputs_lg=len(meta["features"]))
    fz = json.loads((ART / "FROZEN.json").read_text()); n.update(version=fz["version"], frozen_date=fz["frozen_utc"][:10], freeze_reason=fz["reason"])
    d = pd.read_parquet("data/processed/model_table.parquet")
    n["fico_miss"], n["dti_miss"] = d.fico_missing.mean() * 100, d.dti_missing.mean() * 100
    n["bads_train"] = int(d[d["sample"] == "dev_train"].bad.sum()); n["n_total"] = len(d)
    tr, rc_ = d[d["sample"] == "dev_train"], d[d["sample"] == "recent"]
    n["rate_dev_mean"], n["rate_rec_mean"], n["rate_dev_min"] = tr.int_rate.mean(), rc_.int_rate.mean(), tr.int_rate.min()
    for ch in "BC": n[f"chan_dev_{ch}"], n[f"chan_rec_{ch}"] = (tr.channel == ch).mean() * 100, (rc_.channel == ch).mean() * 100
    # generated tables
    n["tbl_ranking"] = md_table(["Sample", "Scorecard AUC (95% CI)", "LightGBM AUC (95% CI)", "Gini S / L", "KS S / L", "Top-decile lift S / L"],
        [[nm, f"{n[f'auc_{k}_s']:.3f} ({n[f'auc_lo_{k}_s']:.3f}-{n[f'auc_hi_{k}_s']:.3f})", f"{n[f'auc_{k}_l']:.3f} ({n[f'auc_lo_{k}_l']:.3f}-{n[f'auc_hi_{k}_l']:.3f})",
          f"{n[f'gini_{k}_s']:.3f} / {n[f'gini_{k}_l']:.3f}", f"{n[f'ks_{k}_s']:.3f} / {n[f'ks_{k}_l']:.3f}", f"{n[f'lift_top10_{k}_s']:.1f} / {n[f'lift_top10_{k}_l']:.1f}"]
         for nm, k in [("Train", "tr"), ("Hold-out", "ho"), ("2005", "oot"), ("2006-08", "cr"), ("2016-19", "rec")]])
    n["tbl_calib"] = md_table(["Sample", "Loans", "Observed", "Predicted S / L", "Obs ÷ pred S / L", "Binomial p S / L"],
        [[nm, f"{n[f'n_{k}']:,}", f"{n[f'obs_{k}']:.2f}%", f"{n[f'pred_{k}_s']:.2f}% / {n[f'pred_{k}_l']:.2f}%", f"{n[f'ratio_{k}_s']:.2f} / {n[f'ratio_{k}_l']:.2f}",
          f"{n[f'p_{k}_s']:.3f} / {n[f'p_{k}_l']:.3f}"] for nm, k in [("Train", "tr"), ("Hold-out", "ho"), ("2005", "oot"), ("2006-08", "cr"), ("2016-19", "rec")]])
    n["tbl_stress"] = md_table(["Shock", "Scorecard mean PD change", "LightGBM mean PD change", "LightGBM loans moving wrong way (hold-out / crisis)"],
        [[nm, f"{n[f'rel_{c}_s_ho']:+.0f}% / {n[f'rel_{c}_s_cr']:+.0f}%", f"{n[f'rel_{c}_l_ho']:+.0f}% / {n[f'rel_{c}_l_cr']:+.0f}%", f"{n[f'wrong_{c}_l_ho']:.0f}% / {n[f'wrong_{c}_l_cr']:.0f}%"]
         for nm, c in [("Credit score -50", "fico_dn"), ("Credit score +50", "fico_up"), ("LTV +10 pts", "ltv"), ("DTI +10 pts", "dti"), ("Interest rate +1 pt", "rate")]])
    pv = ["pd_scorecard", "pd_lgbm", "int_rate", "orig_upb", "fico", "channel", "property_type", "dti", "ltv"]
    nm = {"pd_scorecard": "Scorecard score", "pd_lgbm": "LightGBM score", "int_rate": "Interest rate", "orig_upb": "Original balance", "fico": "Credit score", "channel": "Channel", "property_type": "Property type", "dti": "DTI", "ltv": "LTV"}
    n["tbl_psi"] = md_table(["Variable (PSI vs development)", "2005", "2006-08", "2016-19"], [[nm[v]] + [f"{n[f'psi_{v}_{x}']:.2f}" for x in ["oot", "cr", "rec"]] for v in pv])
    n["tbl_covid"] = md_table(["Vintage", "Bad loans (unadjusted)", "First hit 90+ DPD from Mar 2020", "Had forbearance flag", "Bad rate unadjusted", "Bad rate adjusted"],
        [[str(v), f"{n[f'cv{v}_bad']:,}", f"{n[f'cv{v}_pct']:.0f}%", f"{n[f'cv{v}_forb']:,}", f"{n[f'cv{v}_unadj']:.2f}%", f"{n[f'cv{v}_adj']:.2f}%"] for v in [2016, 2017, 2018, 2019]])
    sg = pd.read_csv(T / "worst_segments_crisis.csv"); sg = sg[sg.model == "scorecard"].sort_values("ratio", ascending=False).head(8)
    n["tbl_segments"] = md_table(["Segment", "Loans", "Observed", "Predicted", "Obs ÷ pred"], [[f"{r.feature}={r.level}", f"{int(r.n):,}", pct(r.obs), pct(r.pred), f"{r.ratio:.1f}"] for r in sg.itertuples()])
    ft = pd.read_csv(T / "fairness_proxies.csv"); ft = ft[(ft.model == "scorecard") & (ft.feature != "state")]
    n["tbl_fair"] = md_table(["Sample", "Group", "Loans", "Obs ÷ pred (95% CI)"], [[r.sample, f"{r.feature}={r.level}", f"{int(r.n):,}", f"{r.ratio:.2f} ({r.ratio_lo:.2f}-{r.ratio_hi:.2f})"] for r in ft.itertuples()])
    return n


def md_table(head, rows):
    return "\n".join(["| " + " | ".join(head) + " |", "|" + "---|" * len(head)] + ["| " + " | ".join(r) + " |" for r in rows])


class _Fmt(dict):
    def __missing__(self, k): raise KeyError(f"template placeholder '{k}' has no value")


def build():
    n = _Fmt(numbers())
    for t in TPL.glob("*.md.tmpl"):
        out = (Path(".") if t.name.startswith("README") else Path("reports")) / t.name.replace(".md.tmpl", ".md")
        out.write_text(t.read_text().format_map(n)); print("built", out)


if __name__ == "__main__":
    build()
