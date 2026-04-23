# How DNNs Break the Curse of Dimensionality: Compositionality and Symmetry Learning

[![arXiv](https://img.shields.io/badge/arXiv-2407.05664-b31b1b.svg)](https://arxiv.org/abs/2407.05664)
[![ICLR 2025](https://img.shields.io/badge/Conference-ICLR%202025-0c6cf2.svg)](https://iclr.cc/)
[![CI](https://github.com/shc443/CoveringNumber_GB/actions/workflows/ci.yml/badge.svg)](https://github.com/shc443/CoveringNumber_GB/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Official codebase for the ICLR 2025 paper by Arthur Jacot, Seok Hoan Choi, and Yuxiao Wen.

## Project Links

- Paper: https://arxiv.org/abs/2407.05664
- Project page: https://shc443.github.io/CoveringNumber_GB/
- Colab notebook: `cleaning/Compositionality_Learning_Experiments.ipynb`
- Colab guide: [COLAB_GUIDE.md](COLAB_GUIDE.md)
- Pages deploy workflow: [`.github/workflows/pages.yml`](.github/workflows/pages.yml)

## Two-Minute Quickstart

```bash
git clone https://github.com/shc443/CoveringNumber_GB
cd CoveringNumber_GB
pip install -r cleaning/curse_dimensionality_sim/requirements.txt
python3 cleaning/curse_dimensionality_sim/experiments/run_compositionality_study.py --focus
```

Equivalent `make` shortcuts are available:

```bash
make install
make smoke
make focus
```

## Repository Map

- `cleaning/curse_dimensionality_sim/`: refactored, script-first framework (recommended).
- `Makefile`: top-level shortcuts for install, smoke, focus, full, and docs preview.
- `MaternKernelTorch.py`: GPU-accelerated PyTorch Matérn kernel with learnable smoothness.
- `paper_code/`: canonical experiment notebooks used during paper development.
- `notebooks/`: additional exploratory notebooks.
- `data/`: local dataset artifacts.
- `docs/`: GitHub Pages site.
- `archive/`: superseded / scratch files retained for provenance — see [`archive/README.md`](archive/README.md).

## Reproducibility Commands

All commands below are run from repository root.

| Goal | Command | Expected outputs |
|---|---|---|
| CI smoke validation on CPU | `python3 cleaning/curse_dimensionality_sim/experiments/run_compositionality_study.py --focus --config .github/ci_smoke_config.yaml` | `results_ci/parameter_sweep_results.csv`, `results_ci/figures/*.png` |
| Focused local run | `python3 cleaning/curse_dimensionality_sim/experiments/run_compositionality_study.py --focus` | `cleaning/curse_dimensionality_sim/results/parameter_sweep_results.csv` |
| Full sweep | `python3 cleaning/curse_dimensionality_sim/experiments/run_compositionality_study.py` | `cleaning/curse_dimensionality_sim/results/` |
| Resume interrupted full sweep | `python3 cleaning/curse_dimensionality_sim/experiments/run_compositionality_study.py --resume` | Updated `parameter_sweep_results.csv` |
| Analysis-only mode | `python3 cleaning/curse_dimensionality_sim/experiments/run_compositionality_study.py --analyze-only` | `summary_report.json`, figure files |
| Framework smoke test | `python3 cleaning/curse_dimensionality_sim/test_framework.py` | console validation of data/models/training |

## Data and Artifacts

- Pre-generated synthetic dataset (Hugging Face):
  - https://huggingface.co/datasets/shc443/MaternKernel_compositionality
- Optional Google Drive dataset workflow:
  - see [COLAB_GUIDE.md](COLAB_GUIDE.md)
- Output conventions:
  - [`results/README.md`](results/README.md)
  - [`cleaning/curse_dimensionality_sim/results/README.md`](cleaning/curse_dimensionality_sim/results/README.md)

## Framework Defaults (Refactored Path)

- Smoothness sweep: `nu_g, nu_h in [0.5, 10.0]` (step `0.5`)
- Architectures: `accordion`, `deep`, `shallow`
- Fixed train/test protocol: train pool of `50,000` with fixed test set `2,500`
- Multi-stage training schedule is config-driven via:
  - `cleaning/curse_dimensionality_sim/config/experiment_config.yaml`

## Citation

If you use this repository, please cite:

```bibtex
@inproceedings{jacot2025dnns,
  title={How DNNs Break the Curse of Dimensionality: Compositionality and Symmetry Learning},
  author={Jacot, Arthur and Choi, Seok Hoan and Wen, Yuxiao},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2025}
}
```

Machine-readable citation metadata is in [`CITATION.cff`](CITATION.cff).

## Contributing and Policies

- Contributing: [CONTRIBUTING.md](CONTRIBUTING.md)
- Code of conduct: [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md)
- Security: [SECURITY.md](SECURITY.md)
- Changelog: [CHANGELOG.md](CHANGELOG.md)

## License

This project is released under the [MIT License](LICENSE).
