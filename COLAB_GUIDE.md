# Google Colab Guide: Compositionality Learning Experiments

Guide for running compositionality learning experiments on Google Colab with GPU acceleration.

## Colab Notebook

Main notebook: [`Compositionality_Learning_Experiments.ipynb`](cleaning/Compositionality_Learning_Experiments.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shc443/CoveringNumber_GB/blob/main/cleaning/Compositionality_Learning_Experiments.ipynb)

---

## Quick Setup

### 1. Enable GPU Runtime

1. **Runtime** → **Change runtime type**
2. **Hardware accelerator** → **GPU** (T4, V100, or A100)
3. **Save** and the notebook will restart

### 2. Run Setup Cells

Execute the first few cells to:
- Install dependencies (`torch`, `numpy`, `pandas`, etc.)
- Clone the repository
- Check GPU availability
- Import the framework

### 3. Choose Data Source

Option A: Load from Hugging Face (recommended)
```python
USE_HUGGINGFACE_DATA = True  # Uses pre-generated datasets
```

Option B: Generate fresh data
```python
USE_HUGGINGFACE_DATA = False  # Generates new Matérn kernel data
```

---

## Experiment Options

### Quick Demo (5–10 minutes)

Trains AccNets and Deep networks on one parameter combination:
- ν_g = 2.0, ν_h = 8.0 (compositional case)
- 8,000 training samples
- 200 epochs per architecture

### Mini Parameter Sweep (20–30 minutes)

Systematic experiment across a reduced parameter space:
- ν values: {1.0, 3.0, 6.0, 10.0}
- Dataset sizes: {2,000, 5,000}
- Architectures: AccNets, Deep
- Total: 32 experiments

### Full Parameter Sweep (20+ hours)

Complete experiment from the paper (commented out by default):
- ν ∈ [0.5, 10] with 0.5 increments (400 combinations)
- N ∈ {100, ..., 50,000} (9 sizes)
- 3 architectures × 3 trials = 32,400 total experiments

---

## Generated Outputs

### Visualizations

1. Performance heatmaps — test loss vs (ν_g, ν_h) parameters
2. Learning curves — performance vs dataset size with theoretical rates
3. Complexity analysis — relationship between bounds and performance
4. Architecture comparison — side-by-side performance metrics

### Data Files

All results saved to `/content/colab_results/`:
- `mini_parameter_sweep.csv` — experimental results
- `complexity_analysis.csv` — complexity bound data
- `accordion_model_example.pth` — trained model weights
- `experiment_summary.json` — statistical summary

---

## Key Insights to Look For

### 1. Compositional learning evidence

AccNets outperform Deep networks when:
- ν_g is small (rough dimensionality reduction)
- ν_h is large (smooth function in reduced space)

Interpretation: the network learns to exploit the compositional structure f = h∘g.

### 2. Breaking the curse of dimensionality

Learning-curve slopes approaching −0.5 in log-log plots:
- Optimal: slope ≈ −0.5 (O(1/√N) rate)
- Cursed: slope ≈ −1/15 ≈ −0.067 (O(1/N^(1/d)) rate)

### 3. Architecture-specific patterns

- AccNets: best for compositional tasks; efficient parameter usage; clear bottleneck representations.
- Deep networks: good general performance; more parameters required; less interpretable structure.
- Shallow networks: struggle with compositional structure; suffer from the curse of dimensionality; useful as a baseline.

---

## Colab Limitations and Tips

### Memory management

```python
# Monitor GPU memory
print(f"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB")

# Clear cache between experiments
torch.cuda.empty_cache()

# Reduce batch sizes if OOM
num_batches = 8  # Increase for smaller batches
```

### Time limits

- Free Colab: ~12-hour sessions
- Pro / Pro+: longer sessions, faster GPUs
- Strategy: save checkpoints frequently

### Storage

- Session storage: files lost when session ends
- Google Drive: mount the drive to persist results
- Download: save important files locally

```python
# Mount Google Drive (optional)
from google.colab import drive
drive.mount('/content/drive')

# Save to Drive
!cp -r /content/colab_results /content/drive/MyDrive/compositionality_results
```

---

## Troubleshooting

### Import errors
```python
# If framework imports fail, run inline setup:
exec(open('/content/CoveringNumber_GB/cleaning/curse_dimensionality_sim/src/data/kernels.py').read())
```

### CUDA out of memory
```python
# Reduce model sizes
configs['accordion']['widths'] = [15, 200, 25, 20]  # Smaller
configs['deep']['widths'] = [15] + [100] * 4 + [20]  # Fewer / smaller layers
```

### Slow training
```python
# Reduce epochs and dataset size
epochs = 50       # Instead of 200+
N_train = 2000    # Instead of 8000+
```

### Dataset loading issues
```python
# Fallback to local data generation
USE_HUGGINGFACE_DATA = False
```

### Mixed-precision training
```python
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():
    output = model(input)
    loss = criterion(output, target)
```

### Batch processing
```python
for batch_start in range(0, len(X), batch_size):
    batch_X = X[batch_start:batch_start + batch_size]
    # Process batch
```

---

## Educational Use

### For students

1. Start with the quick demo to understand the basic concepts.
2. Analyze the visualizations to see compositional learning in action.
3. Modify parameters to experiment with different ν values.
4. Compare architectures to understand trade-offs.

### For researchers

1. Validate the framework and ensure results match expectations.
2. Test new ideas by modifying architectures or training procedures.
3. Generate hypotheses using the visualizations.
4. Benchmark new methods against the provided baselines.

### For practitioners

1. Learn when to use different network types.
2. Understand the relationship between theory and practice via complexity analysis.
3. Transfer insights to real-world problems.
