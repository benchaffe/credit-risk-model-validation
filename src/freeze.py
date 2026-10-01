"""Freeze both models: write FROZEN.json with SHA-256 hashes. Any change after validation starts = new model version,
with a changelog entry. Usage: python -m src.freeze <version> "<reason>" """
import hashlib, json, datetime, sys
from pathlib import Path

ART = Path("reports/artifacts")
FILES = ["scorecard.json", "lgbm.txt", "lgbm_meta.json"]

if __name__ == "__main__":
    version, reason = sys.argv[1], sys.argv[2]
    man = {"version": version, "reason": reason, "frozen_utc": datetime.datetime.utcnow().isoformat(timespec="seconds"),
           "sha256": {f: hashlib.sha256((ART / f).read_bytes()).hexdigest() for f in FILES}}
    (ART / "FROZEN.json").write_text(json.dumps(man, indent=1)); print(json.dumps(man, indent=1))
