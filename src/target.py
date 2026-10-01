"""24-month bad flag per loan.

Bad = within 24 months of origination (Loan Age <= 24) the loan is
  - 90+ days delinquent (Current Loan Delinquency Status >= 03, or RA = REO acquisition), or
  - removed with a credit-event Zero Balance Code: 02 third party sale, 03 short sale / charge off,
    09 REO disposition, 15 whole loan sale.
Codes 01 (voluntary payoff), 16 (reperforming securitisation) and 96 (underwriting defect) are NOT defaults.
Prepaid-without-default loans are good (censoring; documented limitation).
Delinquency 'XX' (not available) is ignored.
Delinquency-based events are NOT counted when the loan is in forbearance (Borrower Assistance Plan = 'F') or the
delinquency is flagged as disaster-related (Delinquency Due to Disaster = 'Y'): these are mostly COVID-19 forbearance,
not credit defaults (see findings: 83% / 98% of 2018 / 2019 'bad' loans first hit 90+ DPD from Mar 2020).
REO acquisition and credit-event zero-balance codes always count. Code 15 inclusion is a judgement call: review.
"""
from pathlib import Path
import duckdb

WINDOW_MONTHS = 24
SEED = 42
BAD_ZB_CODES = ("02", "03", "09", "15")

VINTAGE_SPLIT = {
    "development": (1999, 2004),
    "out_of_time": (2005, 2005),
    "crisis": (2006, 2008),
    "recent": (2016, 2019),
}

OUT = Path("data/processed")


def bad_flag_sql(perf_glob: str) -> str:
    return f"""
    SELECT loan_identifier,
           MAX(CASE WHEN CAST(loan_age AS INT) <= {WINDOW_MONTHS}
                     AND ( current_loan_delinquency_status = 'RA'
                        OR (current_loan_delinquency_status NOT IN ('XX','')
                            AND TRY_CAST(current_loan_delinquency_status AS INT) >= 3
                            AND COALESCE(borrower_assistance_plan,'') <> 'F'
                            AND COALESCE(delinquency_due_to_disaster,'') <> 'Y')
                        OR zero_balance_code IN {BAD_ZB_CODES} )
                    THEN 1 ELSE 0 END) AS bad,
           MAX(CAST(loan_age AS INT)) AS max_age_observed
    FROM read_parquet('{perf_glob}')
    GROUP BY loan_identifier
    """


def main() -> None:
    con = duckdb.connect()
    con.execute(
        f"""COPY (
              SELECT o.*, CAST(substr(o.loan_identifier,2,2) AS INT) + CASE WHEN substr(o.loan_identifier,2,2) >= '90' THEN 1900 ELSE 2000 END AS vintage_year,
                     t.bad, t.max_age_observed
              FROM read_parquet('{OUT}/orig/*.parquet') o
              JOIN ({bad_flag_sql(str(OUT) + '/perf/*.parquet')}) t USING (loan_identifier)
            ) TO '{OUT}/loans.parquet' (FORMAT PARQUET)"""
    )
    print(con.execute(
        f"SELECT vintage_year, COUNT(*) n, ROUND(AVG(bad)*100,2) bad_rate_pct FROM read_parquet('{OUT}/loans.parquet') GROUP BY 1 ORDER BY 1"
    ).fetchdf().to_string(index=False))


if __name__ == "__main__":
    main()
