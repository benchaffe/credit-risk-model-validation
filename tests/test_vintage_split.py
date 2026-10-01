from src.target import VINTAGE_SPLIT, WINDOW_MONTHS


def test_splits_do_not_overlap():
    spans = sorted(VINTAGE_SPLIT.values())
    for (_, end), (start, _) in zip(spans, spans[1:]):
        assert end < start


def test_window():
    assert WINDOW_MONTHS == 24


def test_forbearance_delinquency_not_bad(tmp_path):
    import duckdb
    from src.target import bad_flag_sql
    con = duckdb.connect()
    con.execute("""CREATE TABLE perf AS SELECT * FROM (VALUES
      ('A','03','F','N','5',NULL),   -- 90+ DPD in forbearance -> good
      ('B','03',NULL,'N','5',NULL),  -- 90+ DPD -> bad
      ('C','00',NULL,'N','5','09'),  -- REO disposition -> bad
      ('D','03',NULL,'N','30',NULL), -- outside 24 months -> good
      ('E','03',NULL,'Y','5',NULL)   -- disaster-flagged -> good
    ) t(loan_identifier,current_loan_delinquency_status,borrower_assistance_plan,delinquency_due_to_disaster,loan_age,zero_balance_code)""")
    p = tmp_path / "perf.parquet"
    con.execute(f"COPY perf TO '{p}' (FORMAT PARQUET)")
    got = dict(con.execute(bad_flag_sql(str(p))).fetchall() and
               [(r[0], r[1]) for r in con.execute(bad_flag_sql(str(p))).fetchall()])
    assert got == {"A": 0, "B": 1, "C": 1, "D": 0, "E": 0}
