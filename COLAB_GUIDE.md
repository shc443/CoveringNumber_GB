# 🚀 Google Colab Guide: Compositionality Learning Experiments

Quick guide for running compositionality learning experiments on Google Colab with GPU acceleration.

## 📱 Colab Notebook

**Main Notebook:** [`Compositionality_Learning_Experiments.ipynb`](cleaning/Compositionality_Learning_Experiments.ipynb)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/shc443/CoveringNumber_GB/blob/main/cleaning/Compositionality_Learning_Experiments.ipynb)

---

## ⚡ Quick Setup

### 1. Enable GPU Runtime

1. **Runtime** → **Change runtime type**
2. **Hardware accelerator** → **GPU** (T4, V100, or A100)
3. **Save** and the notebook will restart

### 2. Run Setup Cells

Execute the first few cells to:
- Install dependencies (`torch`, `numpy`, `pandas`, etc.)
- Clone repository 
- Check GPU availability
- Import framework

### 3. Choose Data Source

**Option A: Load from Hugging Face** (Recommended)
```python
USE_HUGGINGFACE_DATA = True  # Uses pre-generated datasets
```

**Option B: Generate Fresh Data**
```python
USE_HUGGINGFACE_DATA = False  # Generates new Matérn kernel data
```

---

## 🎯 Experiment Options

### 🏃‍♂️ Quick Demo (5-10 minutes)

Trains AccNets and Deep networks on one parameter combination:
- ν_g = 2.0, ν_h = 8.0 (compositional case)
- 8,000 training samples
- 200 epochs per architecture

### 🔬 Mini Parameter Sweep (20-30 minutes)

Systematic experiment across reduced parameter space:
- ν values: {1.0, 3.0, 6.0, 10.0}
- Dataset sizes: {2,000, 5,000}
- Architectures: AccNets, Deep
- Total: 32 experiments

### 🏭 Full Parameter Sweep (20+ hours)

Complete experiment from paper (commented out by default):
- ν ∈ [0.5, 10] with 0.5 increments (400 combinations)
- N ∈ {100, ..., 50,000} (9 sizes)
- 3 architectures × 3 trials = 32,400 total experiments

---

## 📊 Generated Outputs

### Visualizations

1. **Performance Heatmaps**: Test loss vs (ν_g, ν_h) parameters
2. **Learning Curves**: Performance vs dataset size with theoretical rates
3. **Complexity Analysis**: Relationship between bounds and performance
4. **Architecture Comparison**: Side-by-side performance metrics

### Data Files

All results saved to `/content/colab_results/`:
- `mini_parameter_sweep.csv`: Experimental results
- `complexity_analysis.csv`: Complexity bound data
- `accordion_model_example.pth`: Trained model weights
- `experiment_summary.json`: Statistical summary

---

## 🧮 Key Insights to Look For

### 1. **Compositional Learning Evidence**

**Look for:** AccNets outperforming Deep networks when:
- ν_g is small (rough dimensionality reduction)  
- ν_h is large (smooth function in reduced space)

**Interpretation:** Network learns to exploit compositional structure f = h∘g

### 2. **Curse of Dimensionality Breaking**

**Look for:** Learning curve slopes approaching -0.5 in log-log plots
- **Optimal:** slope ≈ -0.5 (O(1/√N) rate)
- **Cursed:** slope ≈ -1/15 ≈ -0.067 (O(1/N^(1/d)) rate)

### 3. **Architecture-Specific Patterns**

**AccNets:** 
- Best for compositional tasks
- Efficient parameter usage
- Clear bottleneck representations

**Deep Networks:**
- Good general performance
- More parameters required
- Less interpretable structure

**Shallow Networks:**
- Struggle with compositional structure
- Suffer from curse of dimensionality
- Good baseline for comparison

---

## ⚠️ Colab Limitations and Tips

### Memory Management

```python
# Monitor GPU memory
print(f\"GPU Memory: {torch.cuda.memory_allocated() / 1e9:.2f} GB\")

# Clear cache between experiments
torch.cuda.empty_cache()

# Reduce batch sizes if OOM
num_batches = 8  # Increase for smaller batches
```

### Time Limits

- **Free Colab**: ~12 hour sessions
- **Pro/Pro+**: Longer sessions, faster GPUs
- **Strategy**: Save checkpoints frequently

### Storage

- **Session storage**: Files lost when session ends
- **Google Drive**: Mount drive to persist results
- **Download**: Save important files locally

```python
# Mount Google Drive (optional)
from google.colab import drive
drive.mount('/content/drive')

# Save to Drive
!cp -r /content/colab_results /content/drive/MyDrive/compositionality_results
```

---

## 🔧 Troubleshooting

### Common Issues

#### **Import Errors**
```python
# If framework imports fail, run inline setup:
exec(open('/content/CoveringNumber_GB/cleaning/curse_dimensionality_sim/src/data/kernels.py').read())
```

#### **CUDA Out of Memory**
```python
# Reduce model sizes
configs['accordion']['widths'] = [15, 200, 25, 20]  # Smaller
configs['deep']['widths'] = [15] + [100] * 4 + [20]  # Fewer/smaller layers
```

#### **Slow Training**
```python
# Reduce epochs and dataset size
epochs = 50  # Instead of 200+
N_train = 2000  # Instead of 8000+
```

#### **Dataset Loading Issues**
```python
# Fallback to local data generation
USE_HUGGINGFACE_DATA = False
```

### Performance Optimization

#### **Efficient GPU Usage**
```python
# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler

scaler = GradScaler()
with autocast():\n    output = model(input)\n    loss = criterion(output, target)
```

#### **Batch Processing**
```python
# Process data in smaller chunks
for batch_start in range(0, len(X), batch_size):\n    batch_X = X[batch_start:batch_start+batch_size]\n    # Process batch
```

---

## 📚 Educational Use

### For Students

1. **Start with Quick Demo**: Understand basic concepts
2. **Analyze Visualizations**: See compositional learning in action
3. **Modify Parameters**: Experiment with different ν values
4. **Compare Architectures**: Understand architectural trade-offs

### For Researchers

1. **Validate Framework**: Ensure results match expectations
2. **Test New Ideas**: Modify architectures or training procedures
3. **Generate Hypotheses**: Use visualizations to guide research
4. **Benchmark Methods**: Compare against our baseline results

### For Practitioners

1. **Architecture Insights**: Learn when to use different network types
2. **Complexity Analysis**: Understand relationship between theory and practice
3. **Transfer to Applications**: Apply insights to real-world problems

---

**🎯 Ready to explore how DNNs break the curse of dimensionality? Open the Colab notebook and start experimenting!**
