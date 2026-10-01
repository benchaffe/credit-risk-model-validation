import duckdb
from src.features import FEATURES, NUMERIC, CATEGORICAL
from src.ingest import ORIG_COLS, PERF_COLS


def test_features_come_from_origination_only():
    sources = [c for c, _ in NUMERIC.values()] + [c for c, _ in CATEGORICAL.values()]
    assert set(sources) <= set(ORIG_COLS)
    perf_only = set(PERF_COLS) - set(ORIG_COLS)
    assert not perf_only & set(sources)
    assert "bad" not in FEATURES


def test_model_table_is_sorted_for_reproducible_split():
    import pandas as pd
    ids = pd.read_parquet("data/processed/model_table.parquet", columns=["loan_id"])["loan_id"]
    assert ids.is_monotonic_increasing
