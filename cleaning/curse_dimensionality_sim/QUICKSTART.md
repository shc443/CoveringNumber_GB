# 🚀 One-Line Execution Guide

Run all compositionality learning experiments with a single command!

## Instant Start

### Option 1: Using Make (Simplest)
```bash
make demo      # 5-minute quick demo
make mini      # 30-minute mini sweep  
make full      # Full 48-hour experiments
```

### Option 2: Using Python CLI
```bash
python run.py demo      # 5-minute quick demo
python run.py mini      # 30-minute mini sweep
python run.py full      # Full 48-hour experiments
```

### Option 3: Interactive Quickstart
```bash
./quickstart.sh        # Interactive guided setup
```

## All Available Commands

### Core Experiments
| Command | Runtime | Description |
|---------|---------|-------------|
| `make demo` | 5 min | Quick demonstration with one parameter set |
| `make mini` | 30 min | Mini parameter sweep (4 ν values, 3 N values) |
| `make full` | 48 hours | Full paper experiments (400 ν combinations) |
| `make test` | 1 min | Validate framework functionality |

### Custom Experiments
```bash
# Run with specific parameters
make custom NU_G=2.0 NU_H=8.0 N=10000

# Or using Python CLI with more options
python run.py custom --nu_g 2.0 --nu_h 8.0 --N 10000 --architecture deep
```

### Analysis & Utilities
```bash
make analyze    # Generate visualizations from existing results
make clean      # Clean cache files
make validate   # Check environment setup
make benchmark  # Compare all architectures
```

## Examples

### Example 1: Quick Validation
Just want to see if everything works? Run:
```bash
make demo
```

Output:
```
🚀 COMPOSITIONALITY LEARNING - QUICK DEMO
Device: cuda
Expected runtime: ~5 minutes
📊 Parameters: ν_g=2.0, ν_h=8.0, N=8000
✅ Final Test Loss: 0.3521
💾 Results saved to: results/demo_results_20241204_143022.json
```

### Example 2: Research Experiment
Running the mini parameter sweep for a paper figure:
```bash
make mini
```

### Example 3: Custom Parameters
Testing specific smoothness parameters:
```bash
python run.py custom --nu_g 1.5 --nu_h 10.0 --N 20000 --save
```

### Example 4: Resume Interrupted Run
If a long experiment was interrupted:
```bash
make resume  # or: python run.py full --resume
```

## Output Structure

All results are saved in `./results/`:
```
results/
├── figures/              # Generated visualizations
│   ├── heatmap_*.png
│   └── learning_curves_*.png
├── models/               # Saved model weights
├── *.csv                 # Experiment data
└── *.json                # Summary reports
```

## Tips

1. **Start with `make demo`** to verify everything works
2. **Use `make mini`** for publication-quality results in 30 minutes
3. **Run `make full` overnight** or on a compute cluster
4. **Interrupt safely** - experiments auto-save every 10-50 iterations
5. **GPU recommended** but not required (CPU will be slower)

## Troubleshooting

If you get errors:
```bash
make validate     # Check environment
make install      # Install missing dependencies
make test         # Run framework tests
```

## Advanced Usage

### Parallel Experiments
Run multiple experiments in parallel:
```bash
# Terminal 1
python run.py custom --nu_g 1.0 --nu_h 10.0 --N 10000

# Terminal 2  
python run.py custom --nu_g 2.0 --nu_h 8.0 --N 10000
```

### Batch Processing
Create a batch script:
```bash
#!/bin/bash
for nu_g in 1.0 2.0 5.0 10.0; do
    for nu_h in 1.0 2.0 5.0 10.0; do
        python run.py custom --nu_g $nu_g --nu_h $nu_h --N 10000 --save
    done
done
```

### Docker Support
```bash
make docker-build
make docker-run
```

---

**That's it!** You can now run all experiments with a single command. 🎉