#!/bin/bash

# Quickstart Script for Compositionality Learning Experiments
# One-line setup and execution

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}============================================${NC}"
echo -e "${BLUE}Compositionality Learning - Quick Start${NC}"
echo -e "${BLUE}============================================${NC}"

# Check Python
echo -e "\n${YELLOW}Checking Python installation...${NC}"
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo -e "${RED}Error: Python not found${NC}"
    exit 1
fi

echo -e "${GREEN}✓ Python found: $($PYTHON --version)${NC}"

# Check PyTorch
echo -e "\n${YELLOW}Checking PyTorch...${NC}"
if $PYTHON -c "import torch" 2>/dev/null; then
    echo -e "${GREEN}✓ PyTorch installed${NC}"
    
    # Check CUDA
    if $PYTHON -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo -e "${GREEN}✓ CUDA available${NC}"
        DEVICE="GPU"
    else
        echo -e "${YELLOW}⚠ CUDA not available - will use CPU${NC}"
        DEVICE="CPU"
    fi
else
    echo -e "${YELLOW}PyTorch not installed. Installing...${NC}"
    pip install torch numpy pandas matplotlib seaborn scipy tqdm pyyaml
fi

# Create directories
echo -e "\n${YELLOW}Setting up directories...${NC}"
mkdir -p results/figures
mkdir -p results/models
mkdir -p data/synthetic
echo -e "${GREEN}✓ Directories created${NC}"

# Select experiment type
echo -e "\n${BLUE}Select experiment to run:${NC}"
echo "1) Demo (5 minutes)"
echo "2) Mini sweep (30 minutes)"
echo "3) Full experiments (48 hours)"
echo "4) Custom experiment"
echo "5) Just validate setup"

read -p "Enter choice [1-5]: " choice

case $choice in
    1)
        echo -e "\n${GREEN}Running quick demo...${NC}"
        echo -e "${YELLOW}Device: $DEVICE${NC}"
        echo -e "${YELLOW}Expected time: ~5 minutes${NC}\n"
        $PYTHON run.py demo
        ;;
    2)
        echo -e "\n${GREEN}Running mini parameter sweep...${NC}"
        echo -e "${YELLOW}Device: $DEVICE${NC}"
        echo -e "${YELLOW}Expected time: ~30 minutes${NC}\n"
        $PYTHON run.py mini
        ;;
    3)
        echo -e "\n${RED}Warning: Full experiments will take ~48 hours!${NC}"
        read -p "Are you sure? [y/N]: " confirm
        if [ "$confirm" = "y" ] || [ "$confirm" = "Y" ]; then
            echo -e "\n${GREEN}Starting full parameter sweep...${NC}"
            echo -e "${YELLOW}Device: $DEVICE${NC}"
            echo -e "${YELLOW}Expected time: ~48 hours${NC}"
            echo -e "${YELLOW}Tip: You can interrupt and resume with 'make resume'${NC}\n"
            $PYTHON run.py full --force
        else
            echo "Aborted."
        fi
        ;;
    4)
        echo -e "\n${GREEN}Custom experiment setup${NC}"
        read -p "Enter nu_g value (default 2.0): " nu_g
        nu_g=${nu_g:-2.0}
        read -p "Enter nu_h value (default 8.0): " nu_h
        nu_h=${nu_h:-8.0}
        read -p "Enter N samples (default 10000): " n_samples
        n_samples=${n_samples:-10000}
        read -p "Architecture [accordion/deep/shallow] (default accordion): " arch
        arch=${arch:-accordion}
        
        echo -e "\n${GREEN}Running custom experiment...${NC}"
        echo -e "${YELLOW}Parameters: ν_g=$nu_g, ν_h=$nu_h, N=$n_samples${NC}"
        echo -e "${YELLOW}Architecture: $arch${NC}"
        echo -e "${YELLOW}Device: $DEVICE${NC}\n"
        
        $PYTHON run.py custom --nu_g $nu_g --nu_h $nu_h --N $n_samples --architecture $arch --save
        ;;
    5)
        echo -e "\n${GREEN}Running validation tests...${NC}"
        $PYTHON test_framework.py
        ;;
    *)
        echo -e "${RED}Invalid choice${NC}"
        exit 1
        ;;
esac

echo -e "\n${GREEN}============================================${NC}"
echo -e "${GREEN}Complete! Results saved in ./results/${NC}"
echo -e "${GREEN}============================================${NC}"