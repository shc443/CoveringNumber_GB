"""
Experiment Runner for Compositionality Research

Orchestrates parameter sweeps and systematic experiments studying
how DNNs break the curse of dimensionality through compositional learning.
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from tqdm import tqdm
import os
import logging

from ..data.kernels import CompositionalDataGenerator
from ..models.base_net import BaseNet
from ..models.accordion_net import AccordionNet
from ..models.deep_net import DeepNet  
from ..models.shallow_net import ShallowNet
from ..training.trainer import CompositionTrainer


class CompositionExperimentRunner:
    """
    Orchestrates systematic experiments on compositional learning.
    
    Manages parameter sweeps across:
    - Network architectures (AccNets, Deep, Shallow)
    - Smoothness parameters (ν_g, ν_h)
    - Dataset sizes (N)
    - Multiple random trials
    """
    
    def __init__(self,
                 config: Dict[str, Any],
                 results_dir: str = './results',
                 device: str = 'cuda'):
        """
        Initialize experiment runner.
        
        Args:
            config: Experiment configuration dictionary
            results_dir: Directory for saving results
            device: Device for computation
        """
        self.config = config
        self.results_dir = results_dir
        self.device = device
        
        # Create results directory structure
        os.makedirs(results_dir, exist_ok=True)
        os.makedirs(f'{results_dir}/models', exist_ok=True)
        os.makedirs(f'{results_dir}/data', exist_ok=True)
        os.makedirs(f'{results_dir}/analysis', exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Initialize data generator
        self.data_generator = CompositionalDataGenerator(
            d_input=config['data']['d_input'],
            d_intermediate=config['data']['d_intermediate'],
            d_output=config['data']['d_output'],
            n_samples=config['data']['n_samples'],
            random_seed=config.get('random_seed'),
            device=device
        )

    def _get_model_kwargs(self, architecture: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build model kwargs from architecture config + training config.

        Model configs contain both structural fields (used for width construction)
        and trainable-network fields (passed to model constructors).
        """
        training_cfg = self.config.get('training', {})
        kwargs = {
            'nonlin': model_config.get('nonlin'),
            'loss_type': training_cfg.get('loss_type', 'L2'),
            'test_loss_type': training_cfg.get('test_loss_type', 'L2'),
            'prof_jacot': training_cfg.get('prof_jacot', False),
            'device': self.device
        }

        if architecture == 'accordion':
            kwargs['L'] = model_config.get('L', 5)

        return kwargs
        
    def _create_model(self, 
                     architecture: str, 
                     widths: List[int],
                     **model_kwargs) -> BaseNet:
        """Create model of specified architecture."""
        if architecture == 'accordion':
            return AccordionNet(widths, **model_kwargs)
        elif architecture == 'deep':
            return DeepNet(widths, **model_kwargs)
        elif architecture == 'shallow':
            return ShallowNet(widths, **model_kwargs)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
    
    def run_single_experiment(self,
                            architecture: str,
                            nu_g: float,
                            nu_h: float,
                            N: int,
                            trial: int = 0,
                            save_model: bool = True,
                            precomputed_data: Optional[Tuple[torch.Tensor, torch.Tensor]] = None) -> Dict[str, Any]:
        """
        Run single experiment with specified parameters.
        
        Args:
            architecture: Network architecture ('accordion', 'deep', 'shallow')
            nu_g: Smoothness parameter for function g
            nu_h: Smoothness parameter for function h  
            N: Training dataset size
            trial: Trial number for multiple runs
            save_model: Whether to save trained model
            precomputed_data: Optional pre-generated (X, Y) for fair architecture comparison
            
        Returns:
            Dictionary containing experiment results
        """
        self.logger.info(
            f'Running experiment: {architecture}, ν_g={nu_g}, ν_h={nu_h}, '
            f'N={N}, trial={trial}'
        )
        
        # Generate data for this parameter combination
        if precomputed_data is None:
            X, Y = self.data_generator.generate_compositional_data(
                nu_g, nu_h, seed=trial
            )
        else:
            X, Y = precomputed_data
        
        # Split into train/test
        train_pool_size = self.config['training'].get('train_pool_size', 50000)
        test_N = self.config['training']['test_size']

        if N > train_pool_size:
            raise ValueError(f'N={N} exceeds fixed train pool size ({train_pool_size})')

        if X.shape[0] < train_pool_size + test_N:
            raise ValueError(
                f'Not enough samples ({X.shape[0]}) for fixed split '
                f'({train_pool_size} train pool + {test_N} test)'
            )

        X_train, Y_train = X[:N], Y[:N]
        X_test = X[train_pool_size:train_pool_size + test_N]
        Y_test = Y[train_pool_size:train_pool_size + test_N]
        
        # Create model
        model_config = self.config['models'][architecture]
        widths = self._get_widths(architecture, model_config)
        model_kwargs = self._get_model_kwargs(architecture, model_config)
        
        model = self._create_model(architecture, widths, **model_kwargs)
        
        # Train model
        trainer = CompositionTrainer(model, self.device)
        training_results = trainer.train_full_schedule(
            X_train, Y_train, X_test, Y_test,
            lr_scale=self.config['training']['lr_scale'],
            base_weight_decay=self.config['training'].get('base_weight_decay', 0.0),
            stage_config=self.config['training'].get('stages')
        )
        
        # Save model if requested
        if save_model:
            model_filename = (
                f'{architecture}_N{N}_nu_g{nu_g}_nu_h{nu_h}_trial{trial}.pth'
            )
            model_path = os.path.join(self.results_dir, 'models', model_filename)
            trainer.save_model(model_path)
        
        # Compile results
        results = {
            'architecture': architecture,
            'nu_g': nu_g,
            'nu_h': nu_h,
            'N': N,
            'trial': trial,
            **training_results,
            'model_path': model_path if save_model else None
        }
        
        return results
    
    def run_parameter_sweep(self,
                          architectures: List[str],
                          nu_values: np.ndarray,
                          N_values: List[int],
                          num_trials: int = 3,
                          save_frequency: int = 10) -> pd.DataFrame:
        """
        Run full parameter sweep across all specified combinations.
        
        Args:
            architectures: List of architectures to test
            nu_values: Array of ν values to sweep
            N_values: List of dataset sizes to test
            num_trials: Number of random trials per configuration
            save_frequency: How often to save intermediate results
            
        Returns:
            DataFrame containing all experimental results
        """
        all_results = []
        experiment_count = 0
        
        total_experiments = (
            len(architectures) * len(nu_values) * len(nu_values) * 
            len(N_values) * num_trials
        )
        
        self.logger.info(f'Starting parameter sweep: {total_experiments} total experiments')
        
        with tqdm(total=total_experiments, desc="Parameter Sweep") as pbar:
            for nu_g in nu_values:
                for nu_h in nu_values:
                    for trial in range(num_trials):
                        X, Y = self.data_generator.generate_compositional_data(
                            nu_g, nu_h, seed=trial
                        )
                        for N in N_values:
                            for arch in architectures:
                                try:
                                    results = self.run_single_experiment(
                                        arch,
                                        nu_g,
                                        nu_h,
                                        N,
                                        trial,
                                        precomputed_data=(X, Y)
                                    )
                                    all_results.append(results)
                                except Exception as e:
                                    self.logger.error(
                                        f'Failed experiment {arch}, ν_g={nu_g}, '
                                        f'ν_h={nu_h}, N={N}, trial={trial}: {e}'
                                    )
                                finally:
                                    experiment_count += 1
                                    pbar.update(1)

                                # Periodic saving
                                if experiment_count % save_frequency == 0:
                                    self._save_intermediate_results(all_results)
        
        # Convert to DataFrame and save final results
        results_df = pd.DataFrame(all_results)
        results_path = os.path.join(self.results_dir, 'parameter_sweep_results.csv')
        results_df.to_csv(results_path, index=False)
        
        self.logger.info(f'Parameter sweep completed. Results saved to {results_path}')
        return results_df
    
    def _get_widths(self, architecture: str, model_config: Dict) -> List[int]:
        """Get width configuration for specified architecture."""
        if architecture == 'accordion':
            return [
                self.config['data']['d_input'],
                model_config['d_full'],
                model_config['d_mid'],
                self.config['data']['d_output']
            ]
        elif architecture == 'deep':
            return (
                [self.config['data']['d_input']] +
                [model_config['hidden_width']] * model_config['depth'] +
                [self.config['data']['d_output']]
            )
        elif architecture == 'shallow':
            return [
                self.config['data']['d_input'],
                model_config['hidden_width'],
                self.config['data']['d_output']
            ]
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
    
    def _save_intermediate_results(self, results: List[Dict]) -> None:
        """Save intermediate results during long parameter sweeps."""
        df = pd.DataFrame(results)
        backup_path = os.path.join(self.results_dir, 'intermediate_results.csv')
        df.to_csv(backup_path, index=False)
        
    def load_existing_results(self) -> Optional[pd.DataFrame]:
        """Load existing results if available."""
        results_path = os.path.join(self.results_dir, 'parameter_sweep_results.csv')
        if os.path.exists(results_path):
            return pd.read_csv(results_path)
        return None
    
    def resume_interrupted_sweep(self,
                               architectures: List[str],
                               nu_values: np.ndarray,
                               N_values: List[int],
                               num_trials: int = 3) -> pd.DataFrame:
        """
        Resume interrupted parameter sweep by checking existing results.
        
        Useful for long experiments that may be interrupted.
        """
        existing_results = self.load_existing_results()
        
        if existing_results is not None:
            self.logger.info(f'Found {len(existing_results)} existing results')
            existing_configs = set(
                (row.architecture, row.nu_g, row.nu_h, row.N, row.trial)
                for _, row in existing_results.iterrows()
            )
        else:
            existing_configs = set()
        
        # Generate list of missing experiments
        missing_experiments = []
        for arch in architectures:
            for nu_g in nu_values:
                for nu_h in nu_values:
                    for N in N_values:
                        for trial in range(num_trials):
                            config = (arch, nu_g, nu_h, N, trial)
                            if config not in existing_configs:
                                missing_experiments.append(config)
        
        self.logger.info(f'Found {len(missing_experiments)} missing experiments')
        
        # Run missing experiments
        new_results = []
        for arch, nu_g, nu_h, N, trial in tqdm(missing_experiments, desc="Missing Experiments"):
            try:
                results = self.run_single_experiment(arch, nu_g, nu_h, N, trial)
                new_results.append(results)
            except Exception as e:
                self.logger.error(
                    f'Failed experiment {(arch, nu_g, nu_h, N, trial)}: {e}'
                )
                continue
        
        # Combine with existing results
        if existing_results is not None:
            all_results = pd.concat([existing_results, pd.DataFrame(new_results)], 
                                  ignore_index=True)
        else:
            all_results = pd.DataFrame(new_results)
        
        # Save updated results
        results_path = os.path.join(self.results_dir, 'parameter_sweep_results.csv')
        all_results.to_csv(results_path, index=False)
        
        return all_results
