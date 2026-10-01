"""Fail loudly if the model files no longer match the freeze manifest (i.e. the model changed)."""
import hashlib, json, sys
from pathlib import Path

ART = Path("reports/artifacts")
man = json.loads((ART / "FROZEN.json").read_text())
bad = [f for f, h in man["sha256"].items() if hashlib.sha256((ART / f).read_bytes()).hexdigest() != h]
if bad:
    sys.exit(f"MODEL CHANGED since freeze v{man['version']}: {bad}. If intended, run `make freeze` and log a new model version.")
print(f"Models match freeze v{man['version']} ({man['frozen_utc']})")
