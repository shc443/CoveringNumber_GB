#!/usr/bin/env python3
"""
Main Experiment Script: Compositionality Learning Study

This script runs the full experimental study from:
"How DNNs Break the Curse of Dimensionality: Compositionality and Symmetry Learning"

Usage:
    python run_compositionality_study.py --config config/experiment_config.yaml
    python run_compositionality_study.py --config config/experiment_config.yaml --focus  # Quick test
    python run_compositionality_study.py --resume  # Resume interrupted experiment
"""

import argparse
import logging
import os
import sys
import numpy as np
import matplotlib.pyplot as plt

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.config import ConfigManager
from src.experiments.runner import CompositionExperimentRunner
from src.experiments.analysis import CompositionAnalyzer


def setup_logging(log_level: str = 'INFO'):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler('experiment.log'),
            logging.StreamHandler()
        ]
    )


def main():
    """Main experimental pipeline."""
    default_config_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), '..', 'config', 'experiment_config.yaml')
    )
    parser = argparse.ArgumentParser(
        description='Run compositionality learning experiments'
    )
    parser.add_argument(
        '--config', 
        type=str, 
        default=default_config_path,
        help='Path to experiment configuration file'
    )
    parser.add_argument(
        '--focus',
        action='store_true',
        help='Run focused experiment with reduced parameter space'
    )
    parser.add_argument(
        '--resume',
        action='store_true', 
        help='Resume interrupted experiment'
    )
    parser.add_argument(
        '--analyze-only',
        action='store_true',
        help='Only run analysis on existing results'
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = ConfigManager.load_config(args.config)
    setup_logging(config['output']['log_level'])
    
    logger = logging.getLogger(__name__)
    logger.info("Starting compositionality learning experiment")
    
    # Initialize experiment runner
    runner = CompositionExperimentRunner(
        config=config,
        results_dir=config['output']['results_dir'],
        device=config['hardware']['device']
    )
    
    if not args.analyze_only:
        # Determine parameter space
        if args.focus:
            logger.info("Running focused experiment")
            nu_values = np.array(config['sweep']['focus_nu_values'])
            N_values = config['sweep']['focus_N_values']
        else:
            logger.info("Running full parameter sweep")
            nu_values = np.arange(0.5, 10.5, 0.5)  # [0.5, 1.0, ..., 10.0]
            N_values = config['sweep']['N_values']
        
        architectures = config['sweep']['architectures']
        num_trials = config['training']['num_trials']
        
        # Run experiments
        if args.resume:
            logger.info("Resuming interrupted experiment")
            results_df = runner.resume_interrupted_sweep(
                architectures, nu_values, N_values, num_trials
            )
        else:
            logger.info("Starting new experiment")
            results_df = runner.run_parameter_sweep(
                architectures, nu_values, N_values, num_trials
            )
    else:
        # Load existing results
        results_df = runner.load_existing_results()
        if results_df is None:
            logger.error("No existing results found for analysis")
            return
    
    # Analysis and visualization
    logger.info("Starting analysis and visualization")
    analyzer = CompositionAnalyzer(results_df)
    
    # Create output directory for figures
    figures_dir = os.path.join(config['output']['results_dir'], 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    
    # Generate key visualizations
    try:
        # Performance heatmaps for each architecture
        for arch in ['accordion', 'deep', 'shallow']:
            if arch in results_df['architecture'].values:
                available_N = sorted(results_df[results_df['architecture'] == arch]['N'].unique())
                target_N = 50000 if 50000 in available_N else int(max(available_N))
                fig = analyzer.create_performance_heatmap(
                    metric='final_test_loss',
                    architecture=arch,
                    N=target_N,
                    save_path=f'{figures_dir}/heatmap_{arch}_test_loss.png'
                )
                plt.close(fig)
                
        # Complexity heatmaps
        complexity_metric_by_arch = {
            'accordion': 'complexity_standard',
            'deep': 'ours_standard_rank',
            'shallow': 'frobenius_bound'
        }
        for arch in ['accordion', 'deep', 'shallow']:
            if arch in results_df['architecture'].values:
                metric = complexity_metric_by_arch[arch]
                if metric not in results_df.columns:
                    continue
                available_N = sorted(results_df[results_df['architecture'] == arch]['N'].unique())
                target_N = 50000 if 50000 in available_N else int(max(available_N))
                fig = analyzer.create_performance_heatmap(
                    metric=metric,
                    architecture=arch, 
                    N=target_N,
                    save_path=f'{figures_dir}/heatmap_{arch}_complexity.png'
                )
                plt.close(fig)
                
        # Learning curves for interesting parameter combinations
        interesting_params = [(2.0, 8.0), (8.0, 2.0), (8.0, 8.0)]
        for nu_g, nu_h in interesting_params:
            fig = analyzer.create_learning_curves(
                architectures=['accordion', 'deep', 'shallow'],
                nu_g=nu_g,
                nu_h=nu_h,
                save_path=f'{figures_dir}/learning_curves_nu_g{nu_g}_nu_h{nu_h}.png'
            )
            plt.close(fig)
            
        # Complexity-performance correlation analysis
        fig = analyzer.analyze_complexity_correlation(
            save_path=f'{figures_dir}/complexity_correlation.png'
        )
        plt.close(fig)
        
        logger.info(f"Visualizations saved to {figures_dir}")
        
    except Exception as e:
        logger.error(f"Error during visualization: {e}")
    
    # Generate summary report
    summary = analyzer.generate_summary_report(
        save_path=os.path.join(config['output']['results_dir'], 'summary_report.json')
    )
    
    logger.info("Experiment completed successfully")
    logger.info(f"Results summary: {summary['total_experiments']} experiments completed")
    
    # Print key findings
    print("\n" + "="*50)
    print("EXPERIMENT SUMMARY")
    print("="*50)
    print(f"Total experiments: {summary['total_experiments']}")
    print(f"Architectures tested: {', '.join(summary['architectures'])}")
    print(f"ν parameter range: [{summary['nu_range']['min']}, {summary['nu_range']['max']}]")
    print(f"Dataset sizes: {summary['N_values']}")
    
    print("\nPerformance Summary (Test Loss):")
    for arch, perf in summary['performance_summary'].items():
        print(f"  {arch.capitalize():12s}: {perf['mean_test_loss']:.4f} ± {perf['std_test_loss']:.4f}")
    
    print(f"\nResults saved to: {config['output']['results_dir']}")
    print(f"Figures saved to: {figures_dir}")


if __name__ == '__main__':
    main()
