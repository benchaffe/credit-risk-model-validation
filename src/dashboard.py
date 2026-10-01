"""Build docs/index.html: a static interactive dashboard from reports/tables (summary statistics only, no loan-level data).
Host with GitHub Pages (Settings > Pages > deploy from branch main, folder /docs)."""
import json, subprocess
from pathlib import Path
import pandas as pd
from .build_report import numbers

T = Path("reports/tables")
FEATURE = {"state": "State", "channel": "Channel", "property_type": "Property", "loan_purpose": "Purpose", "occupancy": "Occupancy",
           "first_time_buyer": "First-time buyer", "n_borrowers": "Borrowers"}
LEVEL = {"channel": {"B": "Broker", "C": "Correspondent", "R": "Retail", "T": "Third party"},
         "property_type": {"SF": "Single family", "PU": "Planned unit", "CO": "Condo", "MH": "Manufactured", "CP": "Co-op"},
         "loan_purpose": {"P": "Purchase", "N": "No-cash refi", "C": "Cash-out refi"}, "occupancy": {"P": "Primary", "I": "Investor", "S": "Second home"},
         "first_time_buyer": {"Y": "Yes", "N": "No"}}


def repo_url() -> str:
    try:
        u = subprocess.run(["git", "remote", "get-url", "origin"], capture_output=True, text=True, check=True).stdout.strip()
        return u.removesuffix(".git")
    except Exception:
        return "https://github.com/benchaffe/credit-risk-model-validation"


def build():
    n = numbers()
    bands = pd.read_csv(T / "calibration_bands.csv"); overall = pd.read_csv(T / "calibration_overall.csv").rename(columns={"ratio_obs_to_pred": "ratio"})
    rk = pd.read_csv(T / "ranking.csv")
    sg = pd.read_csv(T / "worst_segments_crisis.csv")
    sg["key"] = [f"{FEATURE[f]}: {LEVEL.get(f, {}).get(str(l), str(l).removesuffix('.0'))}" for f, l in zip(sg.feature, sg.level)]
    st = pd.read_csv(T / "stress.csv"); st = st[st["sample"] == "dev_holdout"]
    stress = []
    for shock in ["Credit score -50", "Credit score +50", "LTV +10 pts", "DTI +10 pts", "Interest rate +1pt"]:
        s = st[(st.shock == shock) & (st.model == "scorecard")].iloc[0]; l = st[(st.shock == shock) & (st.model == "lightgbm")].iloc[0]
        stress.append({"shock": shock, "scorecard": f"{s.rel_change_pct:+.0f}%", "lightgbm": f"{l.rel_change_pct:+.0f}%", "wrong": f"{l.pct_loans_wrong_direction:.0f}%"})
    ps = pd.read_csv(T / "psi.csv")
    names = {"pd_scorecard": "Scorecard score", "pd_lgbm": "LightGBM score", "int_rate": "Interest rate", "orig_upb": "Original balance", "channel": "Channel",
             "fico": "Credit score", "property_type": "Property type", "dti": "DTI", "ltv": "LTV"}
    psi = []
    for v, nm in names.items():
        row = {"name": nm}; row.update({r["sample"]: float(r.psi) for _, r in ps[ps.variable == v].iterrows()}); psi.append(row)
    data = {
        "bands": bands.to_dict("records"), "overall": overall.to_dict("records"), "ranking": rk.to_dict("records"),
        "segments": sg[sg.model.isin(["scorecard", "lightgbm"])][["key", "model", "n", "obs", "pred", "ratio"]].to_dict("records"),
        "stress": stress, "psi": psi,
        "tiles": {"gap": f"{n['ratio_cr_s']:.1f}×", "auc": f"{n['auc_ho_s']:.2f}", "auc_cr": f"{n['auc_cr_s']:.2f}", "psi": f"{n['psi_int_rate_rec']:.1f}",
                  "ca": f"{n['seg_state_CA']:.0f}×", "rise": f"{n['badx_cr']:.1f}×"},
    }
    html = Path("src/dashboard_template.html").read_text()
    html = html.replace("__DATA__", json.dumps(data, separators=(",", ":"))).replace("__REPO__", repo_url()).replace("__VERSION__", f"v{n['version']}").replace("__BUILT__", n["frozen_date"])
    Path("docs").mkdir(exist_ok=True); Path("docs/index.html").write_text(html); Path("docs/.nojekyll").write_text("")
    print("built docs/index.html", len(html) // 1024, "KB")


if __name__ == "__main__":
    build()
