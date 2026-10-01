"""Origination-only features and the modelling table.

Leakage rule: every feature comes from the origination file. Nothing from the performance files
enters except the target (target.py). tests/test_leakage.py enforces this.

Missing / placeholder handling (codes from the July 2026 General User Guide):
  FICO 9999, CLTV/LTV/DTI/MI% 999, units 99, borrowers 99, property type 99, and '9' on
  first-time buyer / occupancy / channel / purpose -> set to missing (NaN / 'UNK').
  FICO and DTI also get a *_missing flag (missingness is itself informative; 2.3% of DTI missing overall).
  CLTV < LTV is disclosed as NA by Freddie; kept as missing.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from .target import VINTAGE_SPLIT, SEED

OUT = Path("data/processed")

NUMERIC = {
    "fico": ("classic_fico", 9999),
    "ltv": ("original_loan_to_value_ltv", 999),
    "cltv": ("original_combined_loan_to_value_cltv", 999),
    "dti": ("original_debt_to_income_dti_ratio", 999),
    "mi_pct": ("mortgage_insurance_percentage_mi", 999),
    "orig_upb": ("original_upb", None),
    "int_rate": ("original_interest_rate", None),
    "n_units": ("number_of_units", 99),
    "n_borrowers": ("number_of_borrowers", 99),
}
CATEGORICAL = {
    "first_time_buyer": ("first_time_homebuyer_indicator", "9"),
    "occupancy": ("occupancy_status", "9"),
    "channel": ("channel", "9"),
    "loan_purpose": ("loan_purpose", "9"),
    "property_type": ("property_type", "99"),
    "state": ("property_state", None),
}
FEATURES = list(NUMERIC) + ["fico_missing", "dti_missing"] + list(CATEGORICAL)


def assign_sample(year: int) -> str | None:
    for name, (a, b) in VINTAGE_SPLIT.items():
        if a <= year <= b:
            return name
    return None


def build() -> pd.DataFrame:
    # DuckDB returns rows in a non-deterministic order; sort so the seeded 70/30 split is reproducible
    raw = pd.read_parquet(OUT / "loans.parquet").sort_values("loan_identifier").reset_index(drop=True)
    df = pd.DataFrame({"loan_id": raw["loan_identifier"], "vintage_year": raw["vintage_year"],
                       "orig_quarter": raw["loan_identifier"].str[3:5], "bad": raw["bad"]})
    for new, (col, miss) in NUMERIC.items():
        s = pd.to_numeric(raw[col], errors="coerce")
        df[new] = s.mask(s == miss) if miss is not None else s
    df["fico_missing"] = df["fico"].isna().astype(int)
    df["dti_missing"] = df["dti"].isna().astype(int)
    for new, (col, miss) in CATEGORICAL.items():
        s = raw[col].replace("", np.nan)
        df[new] = (s.mask(s == miss) if miss is not None else s).fillna("UNK")
    df["sample"] = df["vintage_year"].map(assign_sample)
    df = df[df["sample"].notna()].reset_index(drop=True)   # drops stray 2009 loan
    rng = np.random.default_rng(SEED)
    dev = df["sample"] == "development"
    df.loc[dev, "sample"] = np.where(rng.random(dev.sum()) < 0.7, "dev_train", "dev_holdout")
    return df


def main() -> None:
    df = build()
    df.to_parquet(OUT / "model_table.parquet", index=False)
    print(df.groupby("sample").agg(n=("bad", "size"), bad_rate=("bad", "mean")).round(4))


if __name__ == "__main__":
    main()
