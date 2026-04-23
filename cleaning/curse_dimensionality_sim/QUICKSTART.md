# Quick Start

Run compositionality learning experiments with a single command.

## Instant Start

### Option 1: Make (simplest)
```bash
make demo      # 5-minute quick demo
make mini      # 30-minute mini sweep
make full      # Full 48-hour experiments
```

### Option 2: Python CLI
```bash
python run.py demo      # 5-minute quick demo
python run.py mini      # 30-minute mini sweep
python run.py full      # Full 48-hour experiments
```

### Option 3: Interactive
```bash
./quickstart.sh
```

## Available Commands

### Core experiments
| Command | Runtime | Description |
|---------|---------|-------------|
| `make demo` | 5 min | Single-parameter-set demonstration |
| `make mini` | 30 min | Mini parameter sweep (4 ν values, 3 N values) |
| `make full` | 48 hours | Full paper experiments (400 ν combinations) |
| `make test` | 1 min | Validate framework functionality |

### Custom experiments
```bash
# Run with specific parameters
make custom NU_G=2.0 NU_H=8.0 N=10000

# Python CLI with more options
python run.py custom --nu_g 2.0 --nu_h 8.0 --N 10000 --architecture deep
```

### Analysis and utilities
```bash
make analyze    # Generate visualizations from existing results
make clean      # Clean cache files
make validate   # Check environment setup
make benchmark  # Compare all architectures
```

## Examples

### Quick validation
```bash
make demo
```

Sample output:
```
COMPOSITIONALITY LEARNING - QUICK DEMO
Device: cuda
Expected runtime: ~5 minutes
Parameters: nu_g=2.0, nu_h=8.0, N=8000
Final Test Loss: 0.3521
Results saved to: results/demo_results_20241204_143022.json
```

### Research experiment
```bash
make mini
```

### Custom parameters
```bash
python run.py custom --nu_g 1.5 --nu_h 10.0 --N 20000 --save
```

### Resume an interrupted run
```bash
make resume   # or: python run.py full --resume
```

## Output Structure

All results are saved under `./results/`:
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

1. Start with `make demo` to verify the environment.
2. Use `make mini` for publication-quality results in 30 minutes.
3. Run `make full` overnight or on a compute cluster.
4. Interrupts are safe — experiments auto-save every 10–50 iterations.
5. GPU recommended but not required.

## Troubleshooting

```bash
make validate     # Check environment
make install      # Install missing dependencies
make test         # Run framework tests
```

## Advanced Usage

### Parallel experiments
Run multiple experiments in parallel terminals:
```bash
# Terminal 1
python run.py custom --nu_g 1.0 --nu_h 10.0 --N 10000

# Terminal 2
python run.py custom --nu_g 2.0 --nu_h 8.0 --N 10000
```

### Batch processing
```bash
#!/bin/bash
for nu_g in 1.0 2.0 5.0 10.0; do
    for nu_h in 1.0 2.0 5.0 10.0; do
        python run.py custom --nu_g $nu_g --nu_h $nu_h --N 10000 --save
    done
done
```

### Docker
```bash
make docker-build
make docker-run
```
