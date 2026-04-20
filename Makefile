.PHONY: help install lint test smoke focus full resume analyze docs-serve

help:
	@echo "CoveringNumber_GB top-level commands"
	@echo ""
	@echo "  make install     - Install refactored framework dependencies"
	@echo "  make lint        - Run ruff critical checks"
	@echo "  make test        - Run framework validation script"
	@echo "  make smoke       - Run tiny CPU smoke sweep (CI config)"
	@echo "  make focus       - Run focused sweep"
	@echo "  make full        - Run full sweep"
	@echo "  make resume      - Resume interrupted full sweep"
	@echo "  make analyze     - Analyze existing results"
	@echo "  make docs-serve  - Serve docs locally on http://localhost:8000"

install:
	pip install -r cleaning/curse_dimensionality_sim/requirements.txt
	pip install ruff

lint:
	ruff check cleaning/curse_dimensionality_sim --select E9,F63,F7,F82

test:
	python3 cleaning/curse_dimensionality_sim/test_framework.py

smoke:
	python3 cleaning/curse_dimensionality_sim/experiments/run_compositionality_study.py --focus --config .github/ci_smoke_config.yaml

focus:
	python3 cleaning/curse_dimensionality_sim/experiments/run_compositionality_study.py --focus

full:
	python3 cleaning/curse_dimensionality_sim/experiments/run_compositionality_study.py

resume:
	python3 cleaning/curse_dimensionality_sim/experiments/run_compositionality_study.py --resume

analyze:
	python3 cleaning/curse_dimensionality_sim/experiments/run_compositionality_study.py --analyze-only

docs-serve:
	cd docs && python3 -m http.server 8000

