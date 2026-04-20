"""
Analysis and Visualization Tools

Tools for analyzing experimental results and creating visualizations
for compositionality learning research.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any
import os


class CompositionAnalyzer:
    """
    Analyzer for compositionality experiment results.
    
    Provides methods for creating heatmaps, learning curves,
    and comparative analysis across different architectures.
    """
    
    def __init__(self, results_df: pd.DataFrame):
        """
        Initialize analyzer.
        
        Args:
            results_df: DataFrame containing experimental results
        """
        self.results_df = results_df
        
    def create_performance_heatmap(self,
                                 metric: str = 'final_test_loss',
                                 architecture: str = 'accordion',
                                 N: int = 50000,
                                 aggregate_trials: bool = True,
                                 save_path: Optional[str] = None) -> plt.Figure:
        """
        Create heatmap showing performance vs smoothness parameters.
        
        Args:
            metric: Metric to plot ('final_test_loss', 'complexity_standard', etc.)
            architecture: Network architecture to analyze
            N: Dataset size to focus on
            aggregate_trials: Whether to average across trials
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        # Filter data
        filtered_df = self.results_df[
            (self.results_df['architecture'] == architecture) & 
            (self.results_df['N'] == N)
        ].copy()
        
        if filtered_df.empty:
            raise ValueError(f"No data found for {architecture} with N={N}")
        
        # Aggregate trials if requested
        if aggregate_trials:
            filtered_df = filtered_df.groupby(['nu_g', 'nu_h'])[metric].mean().reset_index()
        
        # Create pivot table for heatmap
        heatmap_data = filtered_df.pivot(index='nu_h', columns='nu_g', values=metric)
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 10))
        
        sns.heatmap(heatmap_data, 
                   annot=True, 
                   fmt='.2e' if 'complexity' in metric else '.3f',
                   cmap='viridis',
                   ax=ax,
                   cbar_kws={'label': metric})
        
        ax.set_title(f'{metric.replace("_", " ").title()}: {architecture.title()}, N={N}')
        ax.set_xlabel('ν_g (smoothness of g)')
        ax.set_ylabel('ν_h (smoothness of h)')
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def create_learning_curves(self,
                             architectures: List[str],
                             nu_g: float,
                             nu_h: float,
                             metric: str = 'final_test_loss',
                             save_path: Optional[str] = None) -> plt.Figure:
        """
        Create learning curves showing performance vs dataset size.
        
        Args:
            architectures: List of architectures to compare
            nu_g, nu_h: Fixed smoothness parameters
            metric: Metric to plot
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for arch in architectures:
            # Filter data for this architecture and parameter combination
            arch_data = self.results_df[
                (self.results_df['architecture'] == arch) &
                (self.results_df['nu_g'] == nu_g) &
                (self.results_df['nu_h'] == nu_h)
            ].copy()
            
            if arch_data.empty:
                continue
                
            # Aggregate across trials
            summary_data = arch_data.groupby('N').agg({
                metric: ['mean', 'std']
            }).reset_index()
            
            summary_data.columns = ['N', f'{metric}_mean', f'{metric}_std']
            
            # Plot with error bars
            ax.errorbar(summary_data['N'], 
                       summary_data[f'{metric}_mean'],
                       yerr=summary_data[f'{metric}_std'],
                       label=arch.title(),
                       marker='o',
                       capsize=5)
        
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel('Dataset Size (N)')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'Learning Curves: ν_g={nu_g}, ν_h={nu_h}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def analyze_complexity_correlation(self,
                                     complexity_metric: str = 'complexity_standard',
                                     performance_metric: str = 'final_test_loss',
                                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Analyze correlation between complexity bounds and actual performance.
        
        Args:
            complexity_metric: Complexity bound to analyze
            performance_metric: Performance metric to correlate
            save_path: Path to save figure
            
        Returns:
            Matplotlib figure object
        """
        # Remove missing values
        valid_data = self.results_df.dropna(subset=[complexity_metric, performance_metric])
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Scatter plot
        for arch in valid_data['architecture'].unique():
            arch_data = valid_data[valid_data['architecture'] == arch]
            axes[0].scatter(arch_data[complexity_metric], 
                          arch_data[performance_metric],
                          label=arch.title(),
                          alpha=0.6)
        
        axes[0].set_xscale('log')
        axes[0].set_yscale('log')
        axes[0].set_xlabel(complexity_metric.replace('_', ' ').title())
        axes[0].set_ylabel(performance_metric.replace('_', ' ').title())
        axes[0].set_title('Complexity vs Performance')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Correlation by architecture
        correlations = []
        architectures = []
        
        for arch in valid_data['architecture'].unique():
            arch_data = valid_data[valid_data['architecture'] == arch]
            if len(arch_data) > 1:
                corr = arch_data[complexity_metric].corr(arch_data[performance_metric])
                correlations.append(corr)
                architectures.append(arch)
        
        axes[1].bar(architectures, correlations)
        axes[1].set_ylabel('Correlation Coefficient')
        axes[1].set_title('Complexity-Performance Correlation by Architecture')
        axes[1].set_ylim(-1, 1)
        
        for i, (arch, corr) in enumerate(zip(architectures, correlations)):
            axes[1].text(i, corr + 0.05 * np.sign(corr), f'{corr:.3f}', 
                        ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            
        return fig
    
    def generate_summary_report(self, save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Generate comprehensive summary of experimental results.
        
        Args:
            save_path: Path to save summary report
            
        Returns:
            Dictionary containing summary statistics
        """
        summary = {
            'total_experiments': len(self.results_df),
            'architectures': list(self.results_df['architecture'].unique()),
            'nu_range': {
                'min': self.results_df[['nu_g', 'nu_h']].min().min(),
                'max': self.results_df[['nu_g', 'nu_h']].max().max()
            },
            'N_values': sorted(self.results_df['N'].unique()),
            'performance_summary': {},
            'complexity_summary': {}
        }
        
        # Performance summary by architecture
        for arch in summary['architectures']:
            arch_data = self.results_df[self.results_df['architecture'] == arch]
            summary['performance_summary'][arch] = {
                'mean_test_loss': arch_data['final_test_loss'].mean(),
                'std_test_loss': arch_data['final_test_loss'].std(),
                'best_test_loss': arch_data['final_test_loss'].min(),
                'worst_test_loss': arch_data['final_test_loss'].max()
            }
        
        # Complexity summary
        complexity_cols = [col for col in self.results_df.columns if 'complexity' in col]
        for col in complexity_cols:
            if col in self.results_df.columns:
                summary['complexity_summary'][col] = {
                    'mean': self.results_df[col].mean(),
                    'std': self.results_df[col].std(),
                    'min': self.results_df[col].min(),
                    'max': self.results_df[col].max()
                }
        
        if save_path:
            import json
            with open(save_path, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
        
        return summary
