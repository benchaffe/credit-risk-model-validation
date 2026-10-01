import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from .common import SAMPLES


def calibration_plot():
    b = pd.read_csv("reports/tables/calibration_bands.csv")
    fig, axs = plt.subplots(2, 5, figsize=(18, 6.5), sharey=False)
    for r, m in enumerate(["scorecard", "lightgbm"]):
        for c, s in enumerate(SAMPLES):
            g = b[(b.model == m) & (b["sample"] == s)]; ax = axs[r, c]
            ax.errorbar(g.pred_pd * 100, g.obs_rate * 100, yerr=[(g.obs_rate - g.obs_lo) * 100, (g.obs_hi - g.obs_rate) * 100], fmt="o", ms=4, color="#3b6ea5")
            mx = max(g.pred_pd.max(), g.obs_hi.max()) * 100; ax.plot([0, mx], [0, mx], "k--", lw=1)
            ax.set_title(f"{m} · {s}", fontsize=9); ax.set_xlabel("predicted PD %"); ax.set_ylabel("observed %")
    fig.tight_layout(); fig.savefig("reports/figures/calibration_by_band.png", dpi=130)
