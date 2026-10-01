"""Parse Freddie Mac sample origination and performance files into Parquet.

Layout: July 2026 File Headers (the raw .txt files have no header row).
One Parquet file per file type per vintage, all strings read as VARCHAR so that
special codes (e.g. 9999, 999, XX) are preserved and cleaned explicitly in features.py / target.py.
"""
from pathlib import Path
import duckdb

RAW = Path("data/raw")
OUT = Path("data/processed")
HEADERS = RAW / "file_headers_july_2026"


def _cols(name: str) -> list[str]:
    return HEADERS.joinpath(name).read_text().strip().split("|")


def _snake(c: str) -> str:
    keep = "".join(ch.lower() if ch.isalnum() else "_" for ch in c)
    return "_".join(p for p in keep.split("_") if p)


ORIG_COLS = [_snake(c) for c in _cols("origination_data_file_header.txt")]
PERF_COLS = [_snake(c) for c in _cols("performance_data_file_header.txt")]


def _convert(src: Path, dst: Path, cols: list[str]) -> None:
    con = duckdb.connect()
    names = "{" + ",".join(f"'{c}':'VARCHAR'" for c in cols) + "}"
    con.execute(
        f"COPY (SELECT * FROM read_csv('{src}', delim='|', header=false, columns={names}, quote='')) "
        f"TO '{dst}' (FORMAT PARQUET, COMPRESSION ZSTD)"
    )


def main() -> None:
    (OUT / "orig").mkdir(parents=True, exist_ok=True)
    (OUT / "perf").mkdir(parents=True, exist_ok=True)
    for d in sorted(RAW.glob("sample_*")):
        if not d.is_dir():
            continue
        y = d.name.split("_")[1]
        for kind, cols in (("orig", ORIG_COLS), ("perf", PERF_COLS)):
            dst = OUT / kind / f"{y}.parquet"
            if dst.exists():
                continue
            print("converting", d.name, kind)
            _convert(d / f"sample_{kind}_{y}.txt", dst, cols)


if __name__ == "__main__":
    main()
