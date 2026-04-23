#!/usr/bin/env python3
"""
Load pre-generated compositional datasets from Hugging Face.

Dataset: shc443/MaternKernel_compositionality
Contains pre-computed Matérn kernel datasets for various ν parameters.
"""

import torch
import numpy as np
from datasets import load_dataset
import os
import sys

# Add parent dir to path for imports
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'src'))

from src.models.accordion_net import AccordionNet
from src.models.deep_net import DeepNet
from src.models.shallow_net import ShallowNet
from src.training.trainer import CompositionTrainer


def load_compositional_dataset(nu_g: float = 2.0, nu_h: float = 8.0, split: str = 'train'):
    """
    Load pre-generated dataset from Hugging Face.
    
    Args:
        nu_g: Smoothness parameter for function g
        nu_h: Smoothness parameter for function h
        split: Dataset split ('train', 'test', or 'validation')
        
    Returns:
        X, Y tensors
    """
    print(f"Loading dataset from Hugging Face (ν_g={nu_g}, ν_h={nu_h})...")
    
    try:
        # Load dataset from Hugging Face
        dataset = load_dataset("shc443/MaternKernel_compositionality", split=split)
        
        # Filter for specific nu values
        filtered = dataset.filter(
            lambda x: abs(x['nu_g'] - nu_g) < 0.1 and abs(x['nu_h'] - nu_h) < 0.1
        )
        
        if len(filtered) == 0:
            print(f"Warning: no data found for nu_g={nu_g}, nu_h={nu_h}. Using closest available.")
            # Use first available dataset as fallback
            filtered = dataset.select(range(1))
        
        # Extract data
        sample = filtered[0]
        X = torch.tensor(sample['X'], dtype=torch.float32)
        Y = torch.tensor(sample['Y'], dtype=torch.float32)
        
        print(f"Loaded data: X shape={X.shape}, Y shape={Y.shape}")
        return X, Y
        
    except Exception as e:
        print(f"Warning: could not load from Hugging Face: {e}")
        print("Falling back to local generation...")
        
        # Fallback to local generation
        from src.data.kernels import CompositionalDataGenerator
        
        generator = CompositionalDataGenerator(
            d_input=15,
            d_intermediate=3,
            d_output=20,
            n_samples=10000,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        
        X, Y = generator.generate_compositional_data(nu_g, nu_h)
        print(f"Generated data locally: X shape={X.shape}, Y shape={Y.shape}")
        return X, Y


def quick_experiment(nu_g: float = 2.0, nu_h: float = 8.0, N_train: int = 8000):
    """
    Run quick experiment with Hugging Face data.
    
    Args:
        nu_g: Smoothness parameter for g
        nu_h: Smoothness parameter for h
        N_train: Number of training samples
    """
    print("=" * 60)
    print("Quick Experiment with Hugging Face Data")
    print("=" * 60)
    
    # Load data
    X, Y = load_compositional_dataset(nu_g, nu_h)
    
    # Split data
    X_train, Y_train = X[:N_train], Y[:N_train]
    X_test, Y_test = X[N_train:N_train+2000], Y[N_train:N_train+2000]
    
    print(f"\nTraining set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Create models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    models = {
        'AccordionNet': AccordionNet([15, 900, 100, 20], L=3, device=device),
        'DeepNet': DeepNet([15] + [500]*12 + [20], device=device),
        'ShallowNet': ShallowNet([15, 1500, 20], device=device)
    }
    
    results = {}
    
    # Train each model
    for name, model in models.items():
        print(f"\n{'='*40}")
        print(f"Training {name}...")
        print(f"{'='*40}")
        
        trainer = CompositionTrainer(model, device=device)
        
        # Quick training (fewer epochs for demo)
        train_loss, test_loss = trainer.train_stage(
            X_train, Y_train, X_test, Y_test,
            lr=1e-3,
            epochs=100,  # Reduced for quick demo
            num_batches=8,
            log_interval=50
        )
        
        results[name] = {
            'train_loss': train_loss,
            'test_loss': test_loss
        }
        
        print(f"{name}: Train Loss={train_loss:.4f}, Test Loss={test_loss:.4f}")
    
    # Print summary
    print("\n" + "=" * 60)
    print("Results Summary")
    print("=" * 60)
    
    for name, res in results.items():
        print(f"{name:15} | Train: {res['train_loss']:.4f} | Test: {res['test_loss']:.4f}")
    
    # Find best model
    best_model = min(results, key=lambda x: results[x]['test_loss'])
    print(f"\nBest Model: {best_model}")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Load and experiment with Hugging Face datasets')
    parser.add_argument('--nu_g', type=float, default=2.0, help='Smoothness parameter for g')
    parser.add_argument('--nu_h', type=float, default=8.0, help='Smoothness parameter for h')
    parser.add_argument('--N_train', type=int, default=8000, help='Number of training samples')
    parser.add_argument('--quick', action='store_true', help='Run quick experiment')
    
    args = parser.parse_args()
    
    if args.quick:
        quick_experiment(args.nu_g, args.nu_h, args.N_train)
    else:
        # Just load and display data
        X, Y = load_compositional_dataset(args.nu_g, args.nu_h)
        print(f"\nData loaded successfully!")
        print(f"X: {X.shape}, dtype={X.dtype}, device={X.device}")
        print(f"Y: {Y.shape}, dtype={Y.dtype}, device={Y.device}")
        print(f"X stats: mean={X.mean():.3f}, std={X.std():.3f}")
        print(f"Y stats: mean={Y.mean():.3f}, std={Y.std():.3f}")