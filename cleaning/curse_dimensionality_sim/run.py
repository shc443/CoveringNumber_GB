#!/usr/bin/env python3
"""
Master CLI for Compositionality Learning Experiments

One-line execution for all experiments:
    python run.py demo      # 5-minute demo
    python run.py mini      # 30-minute mini sweep  
    python run.py full      # Full 48-hour experiments
    python run.py custom    # Custom parameters
    python run.py analyze   # Analysis only

Author: Compositionality Research Team
Paper: How DNNs Break the Curse of Dimensionality
"""

import argparse
import sys
import os
import time
import json
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from src.utils.config import ConfigManager
from src.experiments.runner import CompositionExperimentRunner
from src.experiments.analysis import CompositionAnalyzer
from src.data.kernels import CompositionalDataGenerator


class CompositionCLI:
    """Master CLI for all compositionality experiments."""
    
    def __init__(self):
        """Initialize CLI with configuration."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.results_dir = Path('./results')
        self.results_dir.mkdir(exist_ok=True)
        
        # Load default config
        self.config = self._load_default_config()
        
    def _load_default_config(self):
        """Load default configuration."""
        return {
            'data': {
                'd_input': 15,
                'd_intermediate': 3,
                'd_output': 20,
                'n_samples': 52500
            },
            'models': {
                'accordion': {
                    'd_full': 900,
                    'd_mid': 100,
                    'L': 5,
                    'nonlin': F.relu
                },
                'deep': {
                    'hidden_width': 500,
                    'depth': 12,
                    'nonlin': F.relu
                },
                'shallow': {
                    'hidden_width': 50000,
                    'nonlin': F.relu
                }
            },
            'training': {
                'train_pool_size': 50000,
                'test_size': 2500,
                'num_trials': 3,
                'lr_scale': 1.0,
                'base_weight_decay': 0.0,
                'loss_type': 'L2',
                'test_loss_type': 'L2',
                'prof_jacot': False,
                'stages': {
                    'stage_1': {'lr': 1.5e-3, 'weight_decay': 0.0, 'epochs': 1200},
                    'stage_2': {'lr': 4.0e-4, 'weight_decay': 2.0e-3, 'epochs': 1200},
                    'stage_3': {'lr': 1.0e-4, 'weight_decay': 5.0e-3, 'epochs': 1200}
                }
            }
        }
    
    def demo(self, args):
        """Run quick 5-minute demonstration."""
        print("=" * 70)
        print("COMPOSITIONALITY LEARNING - QUICK DEMO")
        print("=" * 70)
        print(f"Device: {self.device}")
        print("Expected runtime: ~5 minutes")
        print("-" * 70)
        
        start_time = time.time()
        
        # Demo parameters
        nu_g, nu_h = 2.0, 8.0
        N_train = 8000
        epochs_per_stage = 50  # Reduced for demo
        
        print(f"\nParameters: ν_g={nu_g}, ν_h={nu_h}, N={N_train}")
        
        # Generate or load data
        generator = CompositionalDataGenerator(
            **self.config['data'],
            device=self.device
        )
        
        print("Generating data...")
        X, Y = generator.generate_compositional_data(nu_g, nu_h, seed=42)
        
        # Split data
        X_train = X[:N_train]
        Y_train = Y[:N_train]
        X_test = X[50000:52500]
        Y_test = Y[50000:52500]
        
        # Train AccordionNet only for demo
        from src.models.accordion_net import AccordionNet
        from src.training.trainer import CompositionTrainer
        
        print("\nTraining AccordionNet...")
        model = AccordionNet([15, 900, 100, 20], L=5, device=self.device)
        
        trainer = CompositionTrainer(model, self.device)
        
        # Quick training
        results = trainer.train_full_schedule(
            X_train, Y_train, X_test, Y_test,
            lr_scale=1.0,
            stage_config={
                'stage_1': {'epochs': epochs_per_stage},
                'stage_2': {'epochs': epochs_per_stage},
                'stage_3': {'epochs': epochs_per_stage}
            }
        )
        
        # Compute complexity bounds
        bounds = model.compute_complexity_bounds(X_train)
        
        # Display results
        print("\n" + "=" * 70)
        print("DEMO RESULTS")
        print("=" * 70)
        print(f"Final Test Loss: {results['final_test_loss']:.4f}")
        print(f"Complexity Bound: {bounds['complexity_standard']:.2e}")
        print(f"Runtime: {time.time() - start_time:.1f} seconds")
        
        # Save demo results
        demo_file = self.results_dir / f'demo_results_{datetime.now():%Y%m%d_%H%M%S}.json'
        with open(demo_file, 'w') as f:
            json.dump({
                'test_loss': float(results['final_test_loss']),
                'complexity': float(bounds['complexity_standard']),
                'nu_g': nu_g,
                'nu_h': nu_h,
                'N': N_train
            }, f, indent=2)
        
        print(f"\nResults saved to: {demo_file}")
        return 0
    
    def mini(self, args):
        """Run mini parameter sweep (~30 minutes)."""
        print("=" * 70)
        print("COMPOSITIONALITY LEARNING - MINI PARAMETER SWEEP")
        print("=" * 70)
        print(f"Device: {self.device}")
        print("Expected runtime: ~30 minutes")
        print("-" * 70)
        
        start_time = time.time()
        
        # Mini sweep parameters
        nu_values = np.array([2.0, 5.0, 8.0, 10.0])
        N_values = [1000, 5000, 20000]
        architectures = ['accordion', 'deep', 'shallow']
        
        total_experiments = len(nu_values) * len(nu_values) * len(N_values) * len(architectures)
        print(f"\nTotal experiments: {total_experiments}")
        print(f"ν values: {nu_values}")
        print(f"N values: {N_values}")
        print(f"Architectures: {architectures}")
        
        # Create runner
        runner = CompositionExperimentRunner(
            self.config,
            results_dir=str(self.results_dir),
            device=self.device
        )
        
        # Run sweep
        print("\nStarting parameter sweep...")
        results_df = runner.run_parameter_sweep(
            architectures=architectures,
            nu_values=nu_values,
            N_values=N_values,
            num_trials=1,  # Single trial for mini
            save_frequency=10
        )
        
        # Analyze results
        print("\nGenerating visualizations...")
        analyzer = CompositionAnalyzer(results_df)
        figures_dir = self.results_dir / 'figures'
        figures_dir.mkdir(exist_ok=True)
        
        # Create key visualizations
        for arch in architectures:
            analyzer.create_performance_heatmap(
                architecture=arch,
                N=5000,
                save_path=str(figures_dir / f'mini_heatmap_{arch}.png')
            )
        
        # Summary statistics
        summary = analyzer.generate_summary_report()
        
        # Display results
        print("\n" + "=" * 70)
        print("MINI SWEEP RESULTS")
        print("=" * 70)
        
        for arch in architectures:
            arch_data = results_df[results_df['architecture'] == arch]
            mean_loss = arch_data['final_test_loss'].mean()
            print(f"{arch:12} | Mean Test Loss: {mean_loss:.4f}")
        
        runtime = time.time() - start_time
        print(f"\n⏱️  Total runtime: {runtime/60:.1f} minutes")
        
        # Save results
        results_file = self.results_dir / 'mini_parameter_sweep.csv'
        results_df.to_csv(results_file, index=False)
        print(f"Results saved to: {results_file}")
        
        return 0
    
    def full(self, args):
        """Run full parameter sweep from paper (~48 hours)."""
        print("=" * 70)
        print("COMPOSITIONALITY LEARNING - FULL PARAMETER SWEEP")
        print("=" * 70)
        print(f"Device: {self.device}")
        print("Warning: Expected runtime: ~48 hours")
        print("-" * 70)
        
        if not args.force:
            response = input("\nWarning: This will take ~48 hours. Continue? [y/N]: ")
            if response.lower() != 'y':
                print("Aborted.")
                return 1
        
        start_time = time.time()
        
        # Full sweep parameters
        nu_values = np.arange(0.5, 10.5, 0.5)
        N_values = [100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000]
        architectures = ['accordion', 'deep', 'shallow']
        num_trials = 3
        
        total_experiments = len(nu_values)**2 * len(N_values) * len(architectures) * num_trials
        print(f"\nTotal experiments: {total_experiments:,}")
        print(f"ν values: {len(nu_values)} values from 0.5 to 10.0")
        print(f"N values: {N_values}")
        print(f"Architectures: {architectures}")
        print(f"Trials per config: {num_trials}")
        
        # Estimate completion time
        exp_per_hour = 20  # Conservative estimate
        eta_hours = total_experiments / exp_per_hour
        eta_time = datetime.now() + timedelta(hours=eta_hours)
        print(f"\n⏰ Estimated completion: {eta_time:%Y-%m-%d %H:%M}")
        
        # Create runner
        runner = CompositionExperimentRunner(
            self.config,
            results_dir=str(self.results_dir),
            device=self.device
        )
        
        # Run sweep with resume support
        print("\nStarting full parameter sweep...")
        print("Tip: You can safely interrupt and resume with 'python run.py full --resume'")
        
        if args.resume:
            results_df = runner.resume_interrupted_sweep(
                architectures=architectures,
                nu_values=nu_values,
                N_values=N_values,
                num_trials=num_trials
            )
        else:
            results_df = runner.run_parameter_sweep(
                architectures=architectures,
                nu_values=nu_values,
                N_values=N_values,
                num_trials=num_trials,
                save_frequency=50
            )
        
        # Full analysis
        print("\nGenerating comprehensive analysis...")
        analyzer = CompositionAnalyzer(results_df)
        
        # Generate all visualizations
        figures_dir = self.results_dir / 'figures'
        figures_dir.mkdir(exist_ok=True)
        
        for arch in architectures:
            for N in [1000, 10000, 50000]:
                analyzer.create_performance_heatmap(
                    architecture=arch,
                    N=N,
                    save_path=str(figures_dir / f'heatmap_{arch}_N{N}.png')
                )
        
        # Learning curves for key parameter combinations
        key_params = [(1.0, 10.0), (2.0, 8.0), (5.0, 5.0)]
        for nu_g, nu_h in key_params:
            analyzer.create_learning_curves(
                architectures=architectures,
                nu_g=nu_g,
                nu_h=nu_h,
                save_path=str(figures_dir / f'learning_curves_g{nu_g}_h{nu_h}.png')
            )
        
        # Complexity analysis
        analyzer.analyze_complexity_correlation(
            save_path=str(figures_dir / 'complexity_correlation.png')
        )
        
        # Generate report
        summary = analyzer.generate_summary_report()
        with open(self.results_dir / 'full_experiment_report.json', 'w') as f:
            json.dump(summary, f, indent=2)
        
        # Display completion
        runtime = time.time() - start_time
        print("\n" + "=" * 70)
        print("FULL SWEEP COMPLETE")
        print("=" * 70)
        print(f"Total experiments: {len(results_df):,}")
        print(f"Total runtime: {runtime/3600:.1f} hours")
        print(f"Results saved to: {self.results_dir}")
        
        return 0
    
    def custom(self, args):
        """Run custom experiment with specified parameters."""
        print("=" * 70)
        print("COMPOSITIONALITY LEARNING - CUSTOM EXPERIMENT")
        print("=" * 70)
        
        # Custom parameters
        nu_g = args.nu_g
        nu_h = args.nu_h
        N = args.N
        architecture = args.architecture

        if N > 50000:
            raise ValueError("N must be <= 50000 for the fixed 50k/2.5k train-test split.")
        
        print(f"Parameters: ν_g={nu_g}, ν_h={nu_h}, N={N}")
        print(f"Architecture: {architecture}")
        print(f"Device: {self.device}")
        
        # Generate data
        generator = CompositionalDataGenerator(
            **self.config['data'],
            device=self.device
        )
        
        X, Y = generator.generate_compositional_data(nu_g, nu_h, seed=args.seed)
        
        # Split data
        X_train = X[:N]
        Y_train = Y[:N]
        X_test = X[50000:52500]
        Y_test = Y[50000:52500]
        
        # Create model
        if architecture == 'accordion':
            from src.models.accordion_net import AccordionNet
            model = AccordionNet([15, 900, 100, 20], L=5, device=self.device)
        elif architecture == 'deep':
            from src.models.deep_net import DeepNet
            model = DeepNet([15] + [500] * 12 + [20], device=self.device)
        else:
            from src.models.shallow_net import ShallowNet
            model = ShallowNet([15, 50000, 20], device=self.device)
        
        # Train
        from src.training.trainer import CompositionTrainer
        trainer = CompositionTrainer(model, self.device)
        
        print(f"\nTraining {architecture}...")
        results = trainer.train_full_schedule(
            X_train, Y_train, X_test, Y_test,
            lr_scale=args.lr_scale
        )
        
        # Display results
        print("\n" + "=" * 70)
        print("CUSTOM EXPERIMENT RESULTS")
        print("=" * 70)
        print(f"Final Test Loss: {results['final_test_loss']:.4f}")
        
        if args.save:
            # Save results
            output = {
                'architecture': architecture,
                'nu_g': nu_g,
                'nu_h': nu_h,
                'N': N,
                'test_loss': float(results['final_test_loss']),
                'complexity_bounds': {
                    k: float(v) if v is not None and not isinstance(v, list) else v
                    for k, v in results.items()
                    if k != 'final_test_loss'
                }
            }
            
            output_file = self.results_dir / f'custom_{architecture}_g{nu_g}_h{nu_h}_N{N}.json'
            with open(output_file, 'w') as f:
                json.dump(output, f, indent=2)
            
            print(f"Results saved to: {output_file}")
        
        return 0
    
    def analyze(self, args):
        """Analyze existing results without running experiments."""
        print("=" * 70)
        print("COMPOSITIONALITY LEARNING - ANALYSIS MODE")
        print("=" * 70)
        
        # Find results file
        if args.results_file:
            results_file = Path(args.results_file)
        else:
            # Look for most recent results
            csv_files = list(self.results_dir.glob('*.csv'))
            if not csv_files:
                print("No results files found. Run experiments first.")
                return 1
            
            results_file = max(csv_files, key=lambda p: p.stat().st_mtime)
        
        print(f"Loading results from: {results_file}")
        
        # Load results
        results_df = pd.read_csv(results_file)
        print(f"Loaded {len(results_df)} experiments")
        
        # Create analyzer
        analyzer = CompositionAnalyzer(results_df)
        
        # Generate analysis
        figures_dir = self.results_dir / 'figures'
        figures_dir.mkdir(exist_ok=True)
        
        # Get unique architectures
        architectures = results_df['architecture'].unique()
        
        # Generate heatmaps
        print("\nGenerating performance heatmaps...")
        for arch in architectures:
            # Find common N value
            N_values = results_df[results_df['architecture'] == arch]['N'].unique()
            N = N_values[len(N_values)//2] if len(N_values) > 0 else 5000
            
            fig = analyzer.create_performance_heatmap(
                architecture=arch,
                N=int(N),
                save_path=str(figures_dir / f'analysis_heatmap_{arch}.png')
            )
            print(f"  {arch} heatmap saved")
        
        # Generate summary
        print("\nComputing summary statistics...")
        summary = analyzer.generate_summary_report()
        
        # Display summary
        print("\n" + "=" * 70)
        print("ANALYSIS SUMMARY")
        print("=" * 70)
        
        for arch, stats in summary['performance_summary'].items():
            print(f"\n{arch}:")
            print(f"  Mean Test Loss: {stats['mean_test_loss']:.4f} ± {stats['std_test_loss']:.4f}")
            print(f"  Best Test Loss: {stats['best_test_loss']:.4f}")
        
        # Find best configurations
        print("\nBest Configurations:")
        best = results_df.nsmallest(5, 'final_test_loss')
        for _, row in best.iterrows():
            print(f"  {row['architecture']:10} | ν_g={row['nu_g']:4.1f}, ν_h={row['nu_h']:4.1f}, N={row['N']:6d} | Loss={row['final_test_loss']:.4f}")
        
        # Save summary
        summary_file = figures_dir.parent / 'analysis_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nAnalysis saved to: {figures_dir.parent}")
        
        return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='Master CLI for Compositionality Learning Experiments',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py demo                    # Quick 5-minute demo
  python run.py mini                    # 30-minute mini sweep
  python run.py full --force            # Full 48-hour experiments
  python run.py custom --nu_g 2 --nu_h 8 --N 10000
  python run.py analyze                 # Analyze existing results
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Demo command
    demo_parser = subparsers.add_parser('demo', help='Quick 5-minute demonstration')
    
    # Mini command
    mini_parser = subparsers.add_parser('mini', help='Mini parameter sweep (~30 minutes)')
    
    # Full command
    full_parser = subparsers.add_parser('full', help='Full parameter sweep (~48 hours)')
    full_parser.add_argument('--force', action='store_true', help='Skip confirmation prompt')
    full_parser.add_argument('--resume', action='store_true', help='Resume interrupted sweep')
    
    # Custom command
    custom_parser = subparsers.add_parser('custom', help='Custom experiment')
    custom_parser.add_argument('--nu_g', type=float, default=2.0, help='Smoothness for g')
    custom_parser.add_argument('--nu_h', type=float, default=8.0, help='Smoothness for h')
    custom_parser.add_argument('--N', type=int, default=10000, help='Training samples')
    custom_parser.add_argument('--architecture', choices=['accordion', 'deep', 'shallow'], 
                               default='accordion', help='Network architecture')
    custom_parser.add_argument('--lr_scale', type=float, default=1.0, help='Learning rate scale')
    custom_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    custom_parser.add_argument('--save', action='store_true', help='Save results')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze existing results')
    analyze_parser.add_argument('--results_file', help='Path to results CSV file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    # Create CLI instance
    cli = CompositionCLI()
    
    # Route to appropriate command
    if args.command == 'demo':
        return cli.demo(args)
    elif args.command == 'mini':
        return cli.mini(args)
    elif args.command == 'full':
        return cli.full(args)
    elif args.command == 'custom':
        return cli.custom(args)
    elif args.command == 'analyze':
        return cli.analyze(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
