# Archive

Historical artifacts preserved from the pre-refactor layout of this repository.
Nothing here is required to reproduce the ICLR 2025 results — the canonical,
maintained code lives under [`cleaning/curse_dimensionality_sim/`](../cleaning/curse_dimensionality_sim/).

Moved on 2026-04-23 to clean up the top-level layout while retaining history.
All files were moved with `git mv` (where tracked) so `git log --follow` still
works for each file.

## Contents

### `legacy_scripts/`
Root-level Python modules from the original paper-development era.
- `MaternKernel.py` — original NumPy Matérn kernel implementation. Superseded by:
  - `cleaning/curse_dimensionality_sim/src/data/kernels.py::MaternKernel` (refactored CPU/GPU)
  - `MaternKernelTorch.py` at repo root (PyTorch, learnable `nu`, GPU-accelerated)
- `net_model.py` — original `Net` class. Superseded by the models under `cleaning/curse_dimensionality_sim/src/models/`.
- `trainNet.py` — original top-level training entrypoint (imports `net_model`). Superseded by
  `cleaning/curse_dimensionality_sim/experiments/run_compositionality_study.py`.
- `util.py` — original helpers. No longer referenced.

`THEORETICAL_BACKGROUND.md` still shows a `MaternKernel(X, nu)` code snippet; the
snippet remains valid because the refactored `MaternKernel` class under
`cleaning/.../src/data/kernels.py` exposes the same constructor signature.

### `paper_code_scratch/`
Jupyter auto-saved "Copy" variants of notebooks that already have a canonical
version under [`../paper_code/`](../paper_code/). These were produced by
Jupyter's File → Make a Copy feature during iterative exploration.

Canonical notebooks retained in `paper_code/`:
- `final_submit_test1.ipynb`
- `final_submit_test_rerun.ipynb`
- `PlotHeat_L1.ipynb`
- `progress_check.ipynb`
- `pruning_shallow.ipynb`

Archived scratch versions (13 files):
- `FCNN_newPlot-Copy1.ipynb` (no non-Copy original ever existed)
- `PlotHeat_L1-Copy10.ipynb` (7.8 MB with embedded outputs)
- `final_submit_test1-Copy{1,2,3,4,6,8,12}.ipynb`
- `final_submit_test_rerun-Copy{1,2,3,4}.ipynb`

### `requirements_root_v20260423.txt`
The original unpinned, bare-package root `requirements.txt`. Kept for reference;
superseded by the pinned file under
`cleaning/curse_dimensionality_sim/requirements.txt`, which the new root
`requirements.txt` now redirects to with `-r`.

Note: the old file listed `sklearn`, which is a deprecated placeholder on PyPI —
the correct package name is `scikit-learn`.

### `Untitled.ipynb` and `__pycache___v20260423/`
Scratch / build artifacts. Both are in `.gitignore` and are not tracked; they
are kept on disk only to honor the project convention of never deleting files
outright.

## If you need to restore something

Use `git mv archive/<path> <original-location>`. Example:

```bash
git mv archive/legacy_scripts/trainNet.py trainNet.py
```

## See also

- Top-level README: [`../README.md`](../README.md)
- Refactored framework: [`../cleaning/curse_dimensionality_sim/`](../cleaning/curse_dimensionality_sim/)
- Changelog: [`../CHANGELOG.md`](../CHANGELOG.md)
