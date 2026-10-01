"""Load frozen models and score every loan. Verifies the freeze hashes first."""
import hashlib, json
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb
from ..scorecard import assign_bins
from ..challenger import prep, CATEGORY_LEVELS

ART = Path("reports/artifacts")
SAMPLES = ["dev_train", "dev_holdout", "out_of_time", "crisis", "recent"]
SEED = 42


def verify_frozen():
    man = json.loads((ART / "FROZEN.json").read_text())
    for f, h in man["sha256"].items():
        assert hashlib.sha256((ART / f).read_bytes()).hexdigest() == h, f"{f} changed since freeze"


def score_models(df: pd.DataFrame) -> pd.DataFrame:
    """Add pd_scorecard and pd_lgbm columns using the frozen models."""
    df = df.copy()
    sc = json.loads((ART / "scorecard.json").read_text())
    z = np.full(len(df), sc["coef"]["const"])
    for f in sc["features"]:
        b = assign_bins(df[f], sc["spec"][f])
        z += sc["coef"][f] * b.map(sc["spec"][f]["woe"]).fillna(0.0).to_numpy()
    df["pd_scorecard"] = 1 / (1 + np.exp(-z))
    meta = json.loads((ART / "lgbm_meta.json").read_text())
    CATEGORY_LEVELS.update(meta["category_levels"])
    booster = lgb.Booster(model_file=str(ART / "lgbm.txt"))
    df["pd_lgbm"] = booster.predict(prep(df))
    return df


def load_scored() -> pd.DataFrame:
    verify_frozen()
    return score_models(pd.read_parquet("data/processed/model_table.parquet"))
