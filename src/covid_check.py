"""Evidence for the COVID/forbearance adjustment to the target (2016-2019 vintages).
Counts first 'bad' events (adjusted vs unadjusted definition) and when they occurred."""
import duckdb
import pandas as pd
from .target import BAD_ZB_CODES, WINDOW_MONTHS

PERF = "data/processed/perf/*.parquet"
con = duckdb.connect()
con.execute(f"""CREATE VIEW ev AS
 SELECT loan_identifier, CAST(substr(loan_identifier,2,2) AS INT)+2000 vy, CAST(period AS INT) period,
   borrower_assistance_plan bap, delinquency_due_to_disaster dd,
   (current_loan_delinquency_status='RA' OR zero_balance_code IN {BAD_ZB_CODES}) hard,
   (TRY_CAST(current_loan_delinquency_status AS INT) >= 3) dq90
 FROM read_parquet('{PERF}') WHERE CAST(loan_age AS INT) <= {WINDOW_MONTHS}""")
q = """
WITH u AS (SELECT loan_identifier, vy, MIN(period) first_bad, MAX((bap='F')::INT) forb FROM ev WHERE hard OR dq90 GROUP BY 1,2),
     a AS (SELECT loan_identifier, vy FROM ev WHERE hard OR (dq90 AND COALESCE(bap,'')<>'F' AND COALESCE(dd,'')<>'Y') GROUP BY 1,2)
SELECT u.vy vintage, COUNT(*) bad_unadjusted, SUM((first_bad>=202003)::INT) first_bad_from_2020_03, SUM(forb) with_forbearance,
       (SELECT COUNT(*) FROM a WHERE a.vy=u.vy) bad_adjusted
FROM u WHERE u.vy BETWEEN 2016 AND 2019 GROUP BY 1 ORDER BY 1"""
out = con.execute(q).fetchdf()
out["pct_from_2020_03"] = 100 * out["first_bad_from_2020_03"] / out["bad_unadjusted"]
out["rate_unadjusted_pct"] = 100 * out["bad_unadjusted"] / 50000
out["rate_adjusted_pct"] = 100 * out["bad_adjusted"] / 50000
out.to_csv("reports/tables/covid_check.csv", index=False); print(out.round(2).to_string(index=False))
