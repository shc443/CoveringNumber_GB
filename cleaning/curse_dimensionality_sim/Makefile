# Makefile for Compositionality Learning Experiments
# One-line commands for all simulations

.PHONY: help demo mini full test clean install analyze

# Default target
help:
	@echo "Compositionality Learning Experiments - One-Line Commands"
	@echo "========================================================="
	@echo ""
	@echo "  make demo      - Run 5-minute quick demonstration"
	@echo "  make mini      - Run 30-minute mini parameter sweep"
	@echo "  make full      - Run full 48-hour experiments"
	@echo "  make test      - Run framework validation tests"
	@echo "  make analyze   - Analyze existing results"
	@echo "  make clean     - Clean results and cache"
	@echo "  make install   - Install dependencies"
	@echo ""
	@echo "Custom experiments:"
	@echo "  make custom NU_G=2.0 NU_H=8.0 N=10000"
	@echo ""

# Quick 5-minute demo
demo:
	@echo "🚀 Running quick demonstration..."
	@python run.py demo

# Mini parameter sweep (~30 minutes)
mini:
	@echo "🔬 Running mini parameter sweep..."
	@python run.py mini

# Full experiments (~48 hours)
full:
	@echo "🏭 Running full parameter sweep (48 hours)..."
	@python run.py full --force

# Resume interrupted full experiments
resume:
	@echo "🔄 Resuming interrupted experiments..."
	@python run.py full --resume --force

# Run tests
test:
	@echo "🧪 Running framework validation tests..."
	@python test_framework.py

# Analyze existing results
analyze:
	@echo "📊 Analyzing existing results..."
	@python run.py analyze

# Custom experiment with parameters
custom:
	@echo "🎯 Running custom experiment..."
	@python run.py custom --nu_g $(NU_G) --nu_h $(NU_H) --N $(N) --save

# Clean results and cache
clean:
	@echo "🧹 Cleaning results and cache..."
	@rm -rf results/__pycache__
	@rm -rf src/__pycache__
	@rm -rf src/*/__pycache__
	@rm -f *.log
	@echo "✅ Cleaned"

# Clean all results (careful!)
clean-all: clean
	@echo "⚠️  Removing all results..."
	@rm -rf results/*
	@echo "✅ All results removed"

# Install dependencies
install:
	@echo "📦 Installing dependencies..."
	@pip install torch numpy pandas matplotlib seaborn scipy tqdm pyyaml
	@echo "✅ Dependencies installed"

# Quick validation
validate:
	@echo "✓ Checking Python..."
	@python --version
	@echo "✓ Checking PyTorch..."
	@python -c "import torch; print(f'PyTorch {torch.__version__}')"
	@echo "✓ Checking CUDA..."
	@python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
	@echo "✅ Validation complete"

# Performance benchmark
benchmark:
	@echo "⚡ Running performance benchmark..."
	@python run.py custom --nu_g 2.0 --nu_h 8.0 --N 1000 --architecture accordion
	@python run.py custom --nu_g 2.0 --nu_h 8.0 --N 1000 --architecture deep
	@python run.py custom --nu_g 2.0 --nu_h 8.0 --N 1000 --architecture shallow

# Docker support
docker-build:
	@echo "🐳 Building Docker image..."
	@docker build -t compositionality-learning .

docker-run:
	@echo "🐳 Running in Docker..."
	@docker run --gpus all -v $(PWD)/results:/app/results compositionality-learning make demo

# Jupyter notebook
notebook:
	@echo "📓 Starting Jupyter notebook..."
	@jupyter notebook cleaning/Compositionality_Learning_Experiments.ipynb

# Documentation
docs:
	@echo "📚 Opening documentation..."
	@open README.md