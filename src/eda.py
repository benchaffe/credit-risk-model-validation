"""Week 1 checkpoint chart: bad rate by origination quarter."""
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_parquet("data/processed/model_table.parquet")
df["q"] = df["vintage_year"].astype(str) + df["orig_quarter"]
g = df.groupby("q")["bad"].agg(["mean", "size"]).reset_index()
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(g["q"], g["mean"] * 100, color="#3b6ea5")
ax.set_xticks(range(0, len(g), 4)); ax.set_xticklabels(g["q"][::4], rotation=45)
ax.set_ylabel("24-month bad rate (%)"); ax.set_title("Bad rate by origination quarter (sample files)")
fig.tight_layout(); fig.savefig("reports/figures/bad_rate_by_quarter.png", dpi=150)
print(g.assign(mean=(g["mean"] * 100).round(2)).to_string(index=False))
