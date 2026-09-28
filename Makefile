.PHONY: install test smoke rehearsal lint format clean

PYTHON ?= python

install:
	$(PYTHON) -m pip install -e ".[dev]"

test:
	$(PYTHON) -m pytest -q

# Fast subset: unit tests only, skips the slower end-to-end integration tests.
smoke:
	$(PYTHON) -m pytest -q tests/unit

lint:
	$(PYTHON) -m ruff check src tests scripts
	$(PYTHON) -m black --check src tests scripts

format:
	$(PYTHON) -m ruff check --fix src tests scripts
	$(PYTHON) -m black src tests scripts

# Full CPU rehearsal on mock data: proves the entire pipeline (data gen,
# splits, training, checkpoint/resume, frozen inference, scan, baselines,
# the queue runner, and paper-artifact aggregation) runs end-to-end before
# tomorrow's real GPU time. Every artifact this produces carries
# is_synthetic: true and `aggregate` refuses to put it in tables/figures --
# that refusal itself is part of what this target verifies.
rehearsal:
	$(PYTHON) -m pytest -q
	rm -rf runs paper_artifacts/tables/*.tex paper_artifacts/figures/*.png paper_artifacts/figures/*.pdf paper_artifacts/paper_facts.json paper_artifacts/results_summary.md
	$(PYTHON) -m lrmc.cli.main run-queue experiments/queue_rehearsal.yaml
	$(PYTHON) -m lrmc.cli.main aggregate
	@echo "--- rehearsal: verifying aggregate correctly refused synthetic runs ---"
	grep -l "No non-synthetic runs" paper_artifacts/tables/*.tex
	grep -q "No non-synthetic runs" paper_artifacts/results_summary.md
	@echo "REHEARSAL OK: full pipeline ran end-to-end on mock data; aggregate correctly excluded all synthetic runs."

clean:
	rm -rf runs .pytest_cache .ruff_cache
	find . -name "__pycache__" -type d -exec rm -rf {} +
