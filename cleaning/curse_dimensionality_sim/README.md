# Curse of Dimensionality Simulation Framework

Structured implementation for the ICLR 2025 paper *"How DNNs Break the Curse of Dimensionality: Compositionality and Symmetry Learning"* (Jacot, Choi, Wen).

## Research Overview

This framework studies how Deep Neural Networks overcome the curse of dimensionality by learning compositional representations f = h∘g, where:
- g: R^d → R^k (dimensionality reduction mapping)
- h: R^k → R^m (function in reduced space)

## Project Structure

```
curse_dimensionality_sim/
├── config/
│   └── experiment_config.yaml    # Experiment configuration
├── src/
│   ├── data/
│   │   ├── kernels.py            # Matérn kernel data generation
│   │   └── __init__.py
│   ├── models/
│   │   ├── base_net.py           # Abstract base network
│   │   ├── accordion_net.py      # AccNets implementation
│   │   ├── deep_net.py           # Standard deep networks
│   │   ├── shallow_net.py        # Shallow baselines
│   │   └── __init__.py
│   ├── training/
│   │   └── trainer.py            # Multi-stage training pipeline
│   ├── experiments/
│   │   ├── runner.py             # Parameter sweep orchestration
│   │   └── analysis.py           # Results analysis
│   ├── utils/
│   │   └── config.py             # Configuration management
│   └── __init__.py
├── experiments/
│   └── run_compositionality_study.py  # Main experiment script
└── results/                           # Generated results
```

## Quick Start

Install dependencies:
```bash
pip install torch numpy pandas matplotlib seaborn scipy tqdm pyyaml
```

Run a focused experiment:
```bash
cd experiments
python run_compositionality_study.py --focus
```

Run the full parameter sweep:
```bash
python run_compositionality_study.py
```

Resume an interrupted run:
```bash
python run_compositionality_study.py --resume
```

Analyze existing results:
```bash
python run_compositionality_study.py --analyze-only
```

## Experimental Design

### Network architectures
- AccNets: accordion networks with alternating expansion/contraction
- Deep: standard deep networks with uniform hidden layers
- Shallow: single hidden-layer baselines

### Data generation
- Matérn kernels: synthetic compositional data with controllable smoothness
- Parameter sweep: ν ∈ [0.5, 10] for both g and h components
- Compositional structure: Y = h(g(X)) with different regularity levels

### Training protocol
1. Stage 1: lr = 1.5e-3, no regularization, 1200 epochs
2. Stage 2: lr = 4e-4, light regularization (2e-3), 1200 epochs
3. Stage 3: lr = 1e-4, stronger regularization (5e-3), 1200 epochs

### Complexity bounds implemented
- Neyshabur et al. (2015): spectral norm
- Bartlett et al. (2017): Frobenius norm
- Neyshabur et al. (2018): Path-SGD
- Ledent et al. (2021): path norm
- Ours (2024): rank-aware complexity measures

## Features

- Modular design with unified interface across network types
- Configuration-driven experiments
- Multi-stage training with automatic scheduling
- Checkpoint saving and resumption
- Automatic heatmap, learning-curve, and correlation plots
- Seed management for reproducibility

## Visualization

The framework generates:
- Performance heatmaps — test loss vs (ν_g, ν_h) for each architecture
- Complexity heatmaps — generalization bounds vs smoothness parameters
- Learning curves — performance vs dataset size
- Correlation analysis — bounds vs empirical performance

## Configuration

Edit `config/experiment_config.yaml` to customize:
- Network architectures and hyperparameters
- Data generation parameters
- Training schedules and optimization settings
- Parameter sweep ranges
- Output and logging preferences

## Research Questions Addressed

1. Can deep networks learn f = h∘g more efficiently than shallow networks?
2. How do AccNets compare to standard deep networks?
3. How does function regularity (ν parameters) affect learnability?
4. Which complexity measures best predict performance?
5. When and how do deep networks overcome high-dimensional challenges?

## Mathematical Framework

The framework implements the theoretical results showing that networks can efficiently learn compositions with:
- F₁-norm-bounded components (controlled Barron norm)
- Covering-number arguments for dimension-dependent learning rates
- Rank-aware bounds via stable ranks (effective parameter counts)
