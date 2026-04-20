# API Reference Documentation

## Table of Contents
- [Data Module](#data-module)
- [Models Module](#models-module)
- [Training Module](#training-module)
- [Experiments Module](#experiments-module)
- [Utils Module](#utils-module)

---

## Data Module

### `MaternKernel`

Generates synthetic data using Matérn kernels with controllable smoothness.

```python
class MaternKernel:
    def __init__(self, X: torch.Tensor, nu: float, device: str = 'cuda')
```

**Parameters:**
- `X` (torch.Tensor): Input points of shape `[n_points, d_input]`
- `nu` (float): Smoothness parameter (ν ∈ [0.5, ∞))
  - 0.5: Non-differentiable (Exponential kernel)
  - 1.5: Once differentiable
  - 2.5: Twice differentiable
  - ∞: Infinitely differentiable (RBF kernel)
- `device` (str): Computation device ('cuda' or 'cpu')

**Methods:**

#### `compute_kernel_matrix() -> np.ndarray`
Computes the Matérn kernel matrix.

**Returns:** Kernel matrix K of shape `[n_points, n_points]`

#### `sample(shape: Tuple[int, ...], tikhonov_reg: bool = True) -> torch.Tensor`
Samples from the Matérn kernel distribution.

**Parameters:**
- `shape`: Output shape for sampled functions
- `tikhonov_reg`: Add diagonal regularization for numerical stability

**Returns:** Sampled tensor of shape `[n_points, *shape]`

---

### `CompositionalDataGenerator`

Generates compositional datasets Y = h(g(X)) with different smoothness levels.

```python
class CompositionalDataGenerator:
    def __init__(self, 
                 d_input: int = 15,
                 d_intermediate: int = 3, 
                 d_output: int = 20,
                 n_samples: int = 50000,
                 device: str = 'cuda')
```

**Parameters:**
- `d_input`: Input dimension
- `d_intermediate`: Intermediate dimension (bottleneck)
- `d_output`: Output dimension
- `n_samples`: Number of samples to generate
- `device`: Computation device

**Methods:**

#### `generate_compositional_data(nu_g: float, nu_h: float, seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]`

Generates compositional dataset Y = h(g(X)).

**Parameters:**
- `nu_g`: Smoothness parameter for function g
- `nu_h`: Smoothness parameter for function h
- `seed`: Random seed for reproducibility

**Returns:** Tuple of (X, Y) tensors

---

## Models Module

### `BaseNet`

Abstract base class for all network architectures.

```python
class BaseNet(nn.Module):
    def __init__(self, 
                 widths: List[int], 
                 nonlin: callable = F.relu,
                 loss_type: str = 'L2',
                 test_loss_type: str = 'L2',
                 prof_jacot: bool = False,
                 reduction: str = 'mean')
```

**Parameters:**
- `widths`: Layer widths `[d_in, hidden_widths..., d_out]`
- `nonlin`: Activation function
- `loss_type`: Training loss ('L1', 'L2', 'fro', 'nuc')
- `test_loss_type`: Test loss type
- `prof_jacot`: Use Prof. Jacot's L1 definition
- `reduction`: Loss reduction method ('mean', 'sum')

**Abstract Methods:**
- `forward(x: torch.Tensor) -> torch.Tensor`: Forward pass
- `compute_complexity_bounds(X_train: torch.Tensor) -> Dict[str, float]`: Compute generalization bounds

**Concrete Methods:**

#### `compute_loss(y_pred: torch.Tensor, y_true: torch.Tensor, loss_type: str = None) -> torch.Tensor`
Unified loss computation supporting multiple loss types.

#### `compute_weight_norms() -> List[float]`
Compute Frobenius norms of weight matrices.

#### `compute_lipschitz_constants() -> List[float]`
Compute layer-wise Lipschitz constants (spectral norms).

#### `compute_ranks(atol: float = 0.1, rtol: float = 0.1) -> List[int]`
Compute ranks of weight matrices.

#### `compute_stable_ranks() -> List[float]`
Compute stable ranks: `||W||_F^2 / ||W||_2^2`

---

### `AccordionNet`

Accordion Network with alternating expansion/contraction structure.

```python
class AccordionNet(BaseNet):
    def __init__(self, 
                 widths: List[int],
                 L: int = 5,
                 **kwargs)
```

**Parameters:**
- `widths`: `[d_input, d_full, d_mid, d_output]` defining accordion structure
- `L`: Number of accordion layers (expansion-contraction pairs)
- `**kwargs`: Additional arguments for BaseNet

**Additional Methods:**

#### `compute_accordion_norms() -> List[float]`
Compute norms specific to accordion structure.

#### `compute_accordion_lipschitz() -> List[float]`
Compute Lipschitz constants for accordion layer pairs.

---

### `DeepNet`

Standard deep neural network with uniform hidden layers.

```python
class DeepNet(BaseNet):
    def __init__(self, 
                 widths: List[int],
                 **kwargs)
```

**Parameters:**
- `widths`: Layer widths `[d_input, hidden_width, ..., d_output]`
- `**kwargs`: Additional arguments for BaseNet

**Complexity Bounds Implemented:**
- Neyshabur et al. (2015)
- Bartlett et al. (2017)
- Neyshabur et al. (2018)
- Ledent et al. (2021)
- Our bound (2024)

---

### `ShallowNet`

Single hidden layer network used as baseline.

```python
class ShallowNet(BaseNet):
    def __init__(self, 
                 widths: List[int],
                 **kwargs)
```

**Parameters:**
- `widths`: Layer widths `[d_input, hidden_width, d_output]` (exactly 3 values)
- `**kwargs`: Additional arguments for BaseNet

---

## Training Module

### `CompositionTrainer`

Multi-stage trainer for compositionality learning experiments.

```python
class CompositionTrainer:
    def __init__(self, 
                 model: BaseNet,
                 device: str = 'cuda')
```

**Parameters:**
- `model`: Neural network to train
- `device`: Device for computation

**Methods:**

#### `train_stage(...) -> Tuple[float, float]`

Trains for one stage with specified hyperparameters.

```python
train_stage(X_train, Y_train, X_test, Y_test,
           lr: float,
           weight_decay: float = 0.0,
           epochs: int = 1200,
           num_batches: int = 5,
           log_interval: int = 100)
```

**Returns:** Tuple of (final_train_loss, final_test_loss)

#### `train_full_schedule(...) -> Dict[str, float]`

Executes full multi-stage training schedule.

```python
train_full_schedule(X_train, Y_train, X_test, Y_test,
                   lr_scale: float = 1.0,
                   base_weight_decay: float = 0.0)
```

**Training Schedule:**
1. Stage 1: LR=1.5e-3 × lr_scale, no regularization
2. Stage 2: LR=4e-4 × lr_scale, weight_decay=2e-3
3. Stage 3: LR=1e-4 × lr_scale, weight_decay=5e-4

**Returns:** Dictionary with final metrics and complexity bounds

---

## Experiments Module

### `CompositionExperimentRunner`

Orchestrates parameter sweeps and systematic experiments.

```python
class CompositionExperimentRunner:
    def __init__(self,
                 config: Dict[str, Any],
                 results_dir: str = './results',
                 device: str = 'cuda')
```

**Parameters:**
- `config`: Experiment configuration dictionary
- `results_dir`: Directory for saving results
- `device`: Device for computation

**Methods:**

#### `run_single_experiment(...) -> Dict[str, Any]`

Runs single experiment with specified parameters.

```python
run_single_experiment(architecture: str,
                     nu_g: float,
                     nu_h: float,
                     N: int,
                     trial: int = 0,
                     save_model: bool = True)
```

#### `run_parameter_sweep(...) -> pd.DataFrame`

Runs full parameter sweep across all combinations.

```python
run_parameter_sweep(architectures: List[str],
                   nu_values: np.ndarray,
                   N_values: List[int],
                   num_trials: int = 3,
                   save_frequency: int = 10)
```

#### `resume_interrupted_sweep(...) -> pd.DataFrame`

Resumes interrupted parameter sweep by checking existing results.

---

### `CompositionAnalyzer`

Analysis and visualization tools for experimental results.

```python
class CompositionAnalyzer:
    def __init__(self, results_df: pd.DataFrame)
```

**Methods:**

#### `create_performance_heatmap(...) -> plt.Figure`

Creates heatmap showing performance vs smoothness parameters.

```python
create_performance_heatmap(metric: str = 'final_test_loss',
                          architecture: str = 'accordion',
                          N: int = 50000,
                          aggregate_trials: bool = True,
                          save_path: Optional[str] = None)
```

#### `create_learning_curves(...) -> plt.Figure`

Creates learning curves showing performance vs dataset size.

```python
create_learning_curves(architectures: List[str],
                      nu_g: float,
                      nu_h: float,
                      metric: str = 'final_test_loss',
                      save_path: Optional[str] = None)
```

#### `analyze_complexity_correlation(...) -> plt.Figure`

Analyzes correlation between complexity bounds and actual performance.

#### `generate_summary_report(...) -> Dict[str, Any]`

Generates comprehensive summary of experimental results.

---

## Utils Module

### `ConfigManager`

Manages experiment configuration loading and validation.

```python
class ConfigManager:
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]
```

**Methods:**

#### `load_config(config_path: str) -> Dict[str, Any]`

Loads experiment configuration from YAML file.

**Parameters:**
- `config_path`: Path to configuration file

**Returns:** Validated configuration dictionary

**Configuration Structure:**
```yaml
data:
  d_input: 15
  d_intermediate: 3
  d_output: 20
  n_samples: 52500

models:
  accordion:
    d_full: 900
    d_mid: 100
    L: 5
    nonlin: 'relu'
    
training:
  test_size: 2500
  num_trials: 3
  lr_scale: 1.0
  loss_type: 'L2'
  
sweep:
  nu_values: [0.5, 1.0, ..., 20.0]
  N_values: [100, 200, ..., 50000]
  architectures: ['accordion', 'deep', 'shallow']
```

---

## Loss Functions

The framework supports multiple loss types:

| Loss Type | Formula | Use Case |
|-----------|---------|----------|
| `'L2'` | MSE: `||Y_pred - Y_true||²` | Standard regression |
| `'L1'` | MAE: `||Y_pred - Y_true||` | Robust to outliers |
| `'L1'` (prof_jacot) | `mean(√Σ(Y_pred - Y_true)²)` | Prof. Jacot's definition |
| `'fro'` | Frobenius norm | Matrix-valued outputs |
| `'nuc'` | Nuclear norm | Low-rank regularization |

---

## Complexity Bounds

### Implemented Bounds

| Bound | Formula | Reference |
|-------|---------|-----------|
| Neyshabur 2015 | `∏ ||W_i||_2` | Spectral norm product |
| Bartlett 2017 | `Σ ||W_i||_F²` | Frobenius norm sum |
| Neyshabur 2018 | `∏ ||W_i||_2 × Σ ||W_i||_F` | Path-SGD bound |
| Ledent 2021 | `Σ (||W_i||_F × ||W_i||_2)` | Path norm |
| Our 2024 | `∏ L_i × Σ(||W_i||_F/L_i × √(d_i + d_{i+1}))` | Rank-aware |

Where:
- `||W_i||_2`: Spectral norm (largest singular value)
- `||W_i||_F`: Frobenius norm
- `L_i`: Lipschitz constant of layer i
- `d_i`: Dimension of layer i

---

## Device Management

All modules support CUDA acceleration:

```python
# Automatic device selection
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Manual specification
model = AccordionNet(widths, device='cuda:0')
trainer = CompositionTrainer(model, device='cuda:1')
```

---

## Random Seed Management

For reproducible experiments:

```python
import torch
import numpy as np

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    
# Use in experiments
generator.generate_compositional_data(nu_g=2.0, nu_h=8.0, seed=42)
```

---

## Error Handling

The framework includes comprehensive error handling:

```python
try:
    results = runner.run_single_experiment(
        'accordion', nu_g, nu_h, N, trial
    )
except ValueError as e:
    logger.error(f"Invalid configuration: {e}")
except RuntimeError as e:
    logger.error(f"Training failed: {e}")
```

Common exceptions:
- `ValueError`: Invalid configuration parameters
- `RuntimeError`: CUDA out of memory
- `FileNotFoundError`: Missing configuration or data files
- `KeyError`: Missing required configuration fields