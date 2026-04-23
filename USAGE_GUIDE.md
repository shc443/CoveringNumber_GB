# Usage Guide: Compositionality Learning Experiments

This guide provides step-by-step instructions for running experiments and reproducing results from the paper.

---

## 🎯 Getting Started

### Prerequisites

```bash
# System requirements
- Python 3.8+
- CUDA-capable GPU (recommended)
- 16GB+ RAM for full parameter sweeps

# Install dependencies
pip install torch numpy pandas matplotlib seaborn scipy tqdm pyyaml
```

### Environment Setup

```bash
# Clone and setup
git clone https://github.com/shc443/CoveringNumber_GB
cd CoveringNumber_GB

# Verify GPU availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## 🔬 Running Experiments

### 1. Quick Validation Experiment

Test the framework with a focused parameter space:

```bash
cd cleaning/curse_dimensionality_sim/experiments
python run_compositionality_study.py --focus
```

**What this does:**
- Tests 4 ν values: {2.0, 5.0, 8.0, 10.0}
- 3 dataset sizes: {1000, 5000, 20000}
- All 3 architectures: AccNets, Deep, Shallow
- Total: 144 experiments (~30 minutes on GPU)

### 2. Full Parameter Sweep

Run the complete experimental study:

```bash
python run_compositionality_study.py
```

**Warning:** This is computationally intensive:
- 400 ν parameter combinations
- 9 dataset sizes  
- 3 architectures
- 3 trials each
- Total: 32,400 experiments (~48 hours on modern GPU)

### 3. Resume Interrupted Experiments

If experiments are interrupted, resume where you left off:

```bash
python run_compositionality_study.py --resume
```

The framework automatically detects completed experiments and only runs missing ones.

### 4. Analysis Only

Generate visualizations from existing results:

```bash
python run_compositionality_study.py --analyze-only
```

---

## 📊 Understanding the Output

### Directory Structure After Running

```
results/
├── models/                           # Trained model weights
│   ├── accordion_N1000_nu_g2.0_nu_h8.0_trial0.pth
│   └── ...
├── figures/                          # Generated visualizations
│   ├── heatmap_accordion_test_loss.png
│   ├── heatmap_deep_complexity.png
│   ├── learning_curves_nu_g2.0_nu_h8.0.png
│   └── complexity_correlation.png
├── parameter_sweep_results.csv       # All experimental data
└── summary_report.json              # Statistical summary
```

### Key Visualizations

#### Performance Heatmaps
Show test loss as function of (ν_g, ν_h):
- **Dark regions**: Low test loss (good performance)
- **Light regions**: High test loss (poor performance)
- **Pattern**: AccNets excel when ν_g small, ν_h large

#### Learning Curves  
Log-log plots of test error vs dataset size N:
- **Slope -0.5**: Indicates optimal O(1/√N) rate
- **Slope < -0.5**: Better than theory predicts
- **Slope > -0.5**: Suffering from curse of dimensionality

#### Complexity Correlation
Scatter plots of complexity bounds vs empirical performance:
- **Strong correlation**: Bound is predictive
- **Weak correlation**: Bound is loose

---

## 🛠️ Customizing Experiments

### Configuration File

Edit `config/experiment_config.yaml` to customize:

```yaml
# Data parameters
data:
  d_input: 15           # Input dimension
  d_intermediate: 3     # Bottleneck dimension
  d_output: 20         # Output dimension
  n_samples: 52500     # Total samples

# Model architectures  
models:
  accordion:
    d_full: 900        # Expansion layer width
    d_mid: 100         # Contraction layer width
    L: 5               # Number of layers
    
  deep:
    hidden_width: 500  # Hidden layer width
    depth: 12          # Number of hidden layers
    
# Training settings
training:
  num_trials: 3        # Trials per configuration
  lr_scale: 1.0        # Learning rate scaling
  loss_type: 'L2'      # Training loss

# Parameter sweep
sweep:
  nu_values: !!python/object/apply:numpy.arange
    - [0.5, 20.5, 0.5]
  N_values: [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
  architectures: ['accordion', 'deep', 'shallow']
```

### Custom Parameter Studies

#### Study Effect of Architecture Depth

```python
# Edit config file
models:
  deep_shallow: {hidden_width: 500, depth: 3}
  deep_medium: {hidden_width: 500, depth: 8}  
  deep_very_deep: {hidden_width: 500, depth: 16}

sweep:
  architectures: ['deep_shallow', 'deep_medium', 'deep_very_deep']
```

#### Study Effect of Width

```python
models:
  narrow: {hidden_width: 100, depth: 12}
  medium: {hidden_width: 500, depth: 12}
  wide: {hidden_width: 1000, depth: 12}
```

#### Study Specific Smoothness Regime

```python
sweep:
  nu_values: [0.5, 1.0, 1.5, 2.0, 2.5]  # Focus on low smoothness
  N_values: [10000, 50000]               # Large datasets only
```

---

## 🧪 Using Individual Components

### Generate Custom Datasets

```python
from src.data.kernels import CompositionalDataGenerator

# Create generator
generator = CompositionalDataGenerator(
    d_input=20,        # Custom input dimension
    d_intermediate=5,  # Custom bottleneck
    d_output=10,       # Custom output dimension
    n_samples=10000
)

# Generate data with specific smoothness
X, Y = generator.generate_compositional_data(nu_g=1.5, nu_h=10.0)

# Save for later use
torch.save({'X': X, 'Y': Y}, 'custom_data.pt')
```

### Train Single Model

```python
from src.models import AccordionNet
from src.training import CompositionTrainer

# Load custom data
data = torch.load('custom_data.pt')
X, Y = data['X'], data['Y']

# Split data
N_train = 8000
X_train, Y_train = X[:N_train], Y[:N_train]
X_test, Y_test = X[N_train:], Y[N_train:]

# Create and train model
model = AccordionNet(
    widths=[20, 1000, 50, 10],  # Custom architecture
    L=3,                        # 3 accordion layers
    loss_type='L1'              # L1 loss
)

trainer = CompositionTrainer(model)
results = trainer.train_full_schedule(X_train, Y_train, X_test, Y_test)

print(f"Final test loss: {results['final_test_loss']:.4f}")
print(f"Complexity bound: {results['ours_standard_rank']:.2e}")
```

### Analyze Pretrained Models

```python
# Load pretrained model
checkpoint = torch.load('results/models/accordion_N50000_nu_g2.0_nu_h8.0_trial0.pth')
model.load_state_dict(checkpoint['model_state_dict'])

# Compute complexity measures
bounds = model.compute_complexity_bounds(X_train)

# Analyze learned representations
representations = model.get_layer_representations(X_test[:100])
bottleneck_repr = representations['layer_1_post']  # After first contraction

print(f"Bottleneck dimension: {bottleneck_repr.shape}")
print(f"Effective rank: {torch.linalg.matrix_rank(bottleneck_repr)}")
```

---

## 📈 Visualization and Analysis

### Creating Custom Plots

```python
from src.experiments.analysis import CompositionAnalyzer

# Load results
results_df = pd.read_csv('results/parameter_sweep_results.csv')
analyzer = CompositionAnalyzer(results_df)

# Custom performance heatmap
fig = analyzer.create_performance_heatmap(
    metric='final_test_loss',
    architecture='accordion',
    N=20000,
    save_path='custom_heatmap.png'
)

# Learning curves for interesting parameter combinations
fig = analyzer.create_learning_curves(
    architectures=['accordion', 'deep'],
    nu_g=2.0,   # Rough g function
    nu_h=10.0,  # Smooth h function
    save_path='learning_curves.png'
)

# Complexity analysis
fig = analyzer.analyze_complexity_correlation(
    complexity_metric='ours_standard_rank',
    performance_metric='final_test_loss',
    save_path='complexity_analysis.png'
)
```

### Statistical Analysis

```python
# Generate summary statistics
summary = analyzer.generate_summary_report()

print("Performance Summary:")
for arch, stats in summary['performance_summary'].items():
    print(f"{arch}: {stats['mean_test_loss']:.4f} ± {stats['std_test_loss']:.4f}")

# Find best performing configurations
best_configs = results_df.nsmallest(10, 'final_test_loss')
print("\nBest Configurations:")
print(best_configs[['architecture', 'nu_g', 'nu_h', 'N', 'final_test_loss']])
```

---

## 🐛 Troubleshooting

### Common Issues

#### CUDA Out of Memory
```python
# Reduce batch size in training
# Edit config file:
training:
  batch_size: 1000  # Reduce from default 5000
```

#### Missing Data Files
```bash
# Download synthetic datasets
gdown --fuzzy https://drive.google.com/file/d/YOUR_LINK
unzip sampled_kernel.zip -d data/synthetic/
```

#### Slow Training
```python
# Use CPU if GPU is unavailable
hardware:
  device: 'cpu'
  
# Or reduce parameter space for testing
sweep:
  nu_values: [1.0, 5.0, 10.0]  # Fewer values
  N_values: [1000, 5000]       # Smaller datasets
```

#### Import Errors
```bash
# Ensure proper path setup
cd cleaning/curse_dimensionality_sim
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
python experiments/run_compositionality_study.py
```

### Performance Optimization

#### Multi-GPU Training
```python
# Use DataParallel for multiple GPUs
if torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
```

#### Memory Management
```python
# Clear cache between experiments  
torch.cuda.empty_cache()

# Use gradient checkpointing for large models
model.gradient_checkpointing = True
```

---

## 📝 Best Practices

### Reproducible Experiments

```python
# Always set seeds
import random
import numpy as np
import torch

def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
set_all_seeds(42)
```

### Monitoring Long Experiments

```python
# Use logging to track progress
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('experiment.log'),
        logging.StreamHandler()
    ]
)
```

### Saving Intermediate Results

The framework automatically saves results every 10 experiments to prevent data loss during long runs.

---

*For additional questions, refer to the [API Reference](API_REFERENCE.md) or [Theoretical Background](THEORETICAL_BACKGROUND.md) documentation.*