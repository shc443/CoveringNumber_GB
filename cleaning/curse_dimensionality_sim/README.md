# Curse of Dimensionality Simulation Framework

Clean, structured implementation for studying **"How DNNs Break the Curse of Dimensionality: Compositionality and Symmetry Learning"**

## 🎯 Research Overview

This framework studies how Deep Neural Networks (DNNs) overcome the curse of dimensionality by learning compositional representations f = h∘g, where:
- **g**: R^d → R^k (dimensionality reduction mapping)
- **h**: R^k → R^m (function in reduced space)

## 📁 Project Structure

```
curse_dimensionality_sim/
├── config/
│   └── experiment_config.yaml    # Experiment configuration
├── src/
│   ├── data/
│   │   ├── kernels.py           # Matérn kernel data generation
│   │   └── __init__.py
│   ├── models/
│   │   ├── base_net.py          # Abstract base network
│   │   ├── accordion_net.py     # AccNets implementation  
│   │   ├── deep_net.py          # Standard deep networks
│   │   ├── shallow_net.py       # Shallow baselines
│   │   └── __init__.py
│   ├── training/
│   │   └── trainer.py           # Multi-stage training pipeline
│   ├── experiments/
│   │   ├── runner.py           # Parameter sweep orchestration
│   │   └── analysis.py         # Results analysis
│   ├── utils/
│   │   └── config.py           # Configuration management
│   └── __init__.py
├── experiments/
│   └── run_compositionality_study.py  # Main experiment script
└── results/                          # Generated results
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install torch numpy pandas matplotlib seaborn scipy tqdm pyyaml
```

### 2. Run Focused Experiment

```bash
cd experiments
python run_compositionality_study.py --focus
```

### 3. Run Full Parameter Sweep

```bash
python run_compositionality_study.py
```

### 4. Resume Interrupted Experiment

```bash
python run_compositionality_study.py --resume
```

### 5. Analyze Existing Results

```bash
python run_compositionality_study.py --analyze-only
```

## 🔬 Experimental Design

### Network Architectures
- **AccNets**: Accordion networks with alternating expansion/contraction
- **Deep**: Standard deep networks with uniform hidden layers  
- **Shallow**: Single hidden layer baselines

### Data Generation
- **Matérn Kernels**: Synthetic compositional data with controllable smoothness
- **Parameter Sweep**: ν ∈ [0.5, 10] for both g and h components
- **Compositional Structure**: Y = h(g(X)) with different regularity levels

### Training Protocol
1. **Stage 1**: High LR (1.5e-3), no regularization, 1200 epochs
2. **Stage 2**: Medium LR (4e-4), light regularization (2e-3), 1200 epochs  
3. **Stage 3**: Low LR (1e-4), stronger regularization (5e-3), 1200 epochs

### Complexity Analysis
- **Neyshabur et al. (2015)**: Spectral norm bounds
- **Bartlett et al. (2017)**: Frobenius norm bounds
- **Neyshabur et al. (2018)**: Path-SGD bounds
- **Ledent et al. (2021)**: Path norm bounds
- **Our bounds (2024)**: Rank-aware complexity measures

## 📊 Key Features

### ✅ **Clean Architecture**
- Modular design with clear separation of concerns
- Unified interface across all network types
- Configuration-driven experiments

### ✅ **Comprehensive Analysis**
- Multiple generalization bounds implementation
- Automatic heatmap and learning curve generation
- Correlation analysis between bounds and performance

### ✅ **Robust Experimentation**
- Multi-stage training with automatic scheduling
- Checkpoint saving and experiment resumption
- Error handling and logging throughout

### ✅ **Research Reproducibility**  
- Seed management for reproducible results
- Systematic parameter sweep orchestration
- Comprehensive result storage and retrieval

## 🎨 Visualization Examples

The framework automatically generates:
- **Performance Heatmaps**: Test loss vs (ν_g, ν_h) for each architecture
- **Complexity Heatmaps**: Generalization bounds vs smoothness parameters
- **Learning Curves**: Performance vs dataset size comparisons
- **Correlation Analysis**: Relationship between complexity bounds and empirical performance

## ⚙️ Configuration

Edit `config/experiment_config.yaml` to customize:
- Network architectures and hyperparameters
- Data generation parameters  
- Training schedules and optimization settings
- Parameter sweep ranges
- Output and logging preferences

## 📈 Research Questions Addressed

1. **Compositional Learning**: Can deep networks learn f = h∘g more efficiently than shallow networks?
2. **Architecture Impact**: How do AccNets compare to standard deep networks?
3. **Smoothness Effects**: How does function regularity (ν parameters) affect learnability?
4. **Generalization Bounds**: Which complexity measures best predict performance?
5. **Curse Breaking**: When and how do deep networks overcome high-dimensional challenges?

## 🧮 Mathematical Framework

The framework implements the theoretical results showing that networks can efficiently learn compositions with:
- **F₁-norm bounded components**: Functions with controlled Barron norm
- **Covering number arguments**: Dimension-dependent learning rates  
- **Rank-aware bounds**: Incorporating effective parameter counts via stable ranks

---

*This implementation provides a clean, extensible foundation for studying compositionality in deep learning while maintaining full compatibility with the original research findings.*
