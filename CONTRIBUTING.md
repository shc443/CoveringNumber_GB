# Contributing Guide

Thanks for your interest in improving this project.

## Development Setup

```bash
git clone https://github.com/shc443/CoveringNumber_GB
cd CoveringNumber_GB
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r cleaning/curse_dimensionality_sim/requirements.txt
pip install ruff
```

## Quick Validation Before PR

```bash
ruff check cleaning/curse_dimensionality_sim
python3 cleaning/curse_dimensionality_sim/test_framework.py
python3 cleaning/curse_dimensionality_sim/experiments/run_compositionality_study.py \
  --focus \
  --config .github/ci_smoke_config.yaml
```

## Branch and PR Expectations

- Keep pull requests scoped to one logical change.
- Include a short summary of what changed and why.
- Add or update docs when behavior changes.
- If you touch experiment logic, include reproducibility notes:
  - exact command
  - config file
  - seed
  - output artifact paths

## Coding Conventions

- Follow existing module layout under `cleaning/curse_dimensionality_sim/src`.
- Prefer small, testable functions.
- Keep scripts deterministic where possible.

## Reporting Issues

Please use GitHub issue templates for:
- bug reports
- feature requests

