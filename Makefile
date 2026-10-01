# One command: `make all` rebuilds data, models, validation tables/figures and PDFs from the raw files in data/raw/.
PYTHON ?= python3

# LightGBM needs libomp. Conda (environment.yml) provides it; on a plain pip setup point at scikit-learn's bundled copy.
OMP ?= $(shell $(PYTHON) -c "import sklearn,os;print(os.path.join(os.path.dirname(sklearn.__file__),'.dylibs'))")
export DYLD_FALLBACK_LIBRARY_PATH := $(OMP)

.PHONY: all data models freeze check validate report test clean-processed
all: test data models check validate report

data:        ## raw .txt -> Parquet -> bad flag -> modelling table -> bad-rate chart
	$(PYTHON) -m src.ingest
	$(PYTHON) -m src.target
	$(PYTHON) -m src.features
	$(PYTHON) -m src.eda
	$(PYTHON) -m src.covid_check

models:      ## fit scorecard and LightGBM (train split only)
	$(PYTHON) -m src.scorecard
	$(PYTHON) -m src.challenger

freeze:      ## record model hashes; ONLY for a deliberate new version: make freeze VERSION=1.2 REASON="..."
	$(PYTHON) -m src.freeze $(VERSION) "$(REASON)"

check:       ## fail if rebuilt models differ from the frozen version
	$(PYTHON) -m src.check_frozen

validate:    ## ranking, calibration, PSI, stress, SHAP, fairness -> reports/tables, reports/figures
	$(PYTHON) -m src.validate

report:      ## fill templates with computed numbers, render PDFs (needs Google Chrome)
	$(PYTHON) -m src.build_report
	$(PYTHON) -m src.render

test:
	$(PYTHON) -m pytest -q

clean-processed:
	rm -rf data/processed
