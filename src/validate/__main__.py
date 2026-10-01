import pandas as pd
from . import ranking, calibration, psi, stress, explain, fairness, plots
from .common import load_scored

pd.set_option("display.width", 220); pd.set_option("display.max_rows", 200)
df = load_scored()
r = ranking.run(df)
bands, tot = calibration.run(df); plots.calibration_plot()
p = psi.run(df)
s, seg = stress.run(df)
imp, rc, rho, agree = explain.run(df)
f = fairness.run(df)
print("== ranking ==\n", r[["sample", "model", "auc", "gini", "ks", "lift_top10"]].round(3).to_string(index=False))
print("== calibration overall ==\n", tot.round(4).to_string(index=False))
print("== PSI (selected) ==\n", p[p.variable.isin(["pd_scorecard", "pd_lgbm", "fico", "ltv", "dti", "int_rate", "orig_upb", "state"])].pivot(index="variable", columns="sample", values="psi").round(3))
print("== stress ==\n", s.round(3).to_string(index=False))
print("== worst crisis segments ==\n", seg[seg.model == "scorecard"].head(8).round(3).to_string(index=False))
print("== SHAP ==\n", imp.round(4).to_string()); print("== reason codes ==\n", rc.round(3).to_string())
print("agreement: score rho", round(rho, 3), "driver rank rho", round(agree, 3))
