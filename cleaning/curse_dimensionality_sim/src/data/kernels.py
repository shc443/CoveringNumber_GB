"""
Matérn Kernel Implementation for Compositionality Learning Research

This module implements Matérn kernel sampling for generating synthetic datasets
that exhibit compositional structure, as used in the study of how DNNs break
the curse of dimensionality through compositionality and symmetry learning.
"""

import torch
import numpy as np
from scipy.special import kv, gamma
from typing import Tuple, Optional
import logging


class MaternKernel:
    """
    Matérn kernel implementation for generating compositional datasets.
    
    The Matérn kernel with smoothness parameter ν controls the differentiability
    of sampled functions, enabling the study of how neural networks learn
    functions with different regularity properties.
    """
    
    def __init__(self, X: torch.Tensor, nu: float, device: str = 'cuda'):
        """
        Initialize Matérn kernel.
        
        Args:
            X: Input points tensor of shape [n_points, d_input]
            nu: Smoothness parameter (ν ∈ [0.5, ∞))
            device: Device for computation ('cuda' or 'cpu')
        """
        self.X = X.cpu().numpy()
        self.n_points = X.shape[0]
        self.nu = nu
        self.device = device
        
        self.logger = logging.getLogger(__name__)
        
    def _compute_distances(self) -> np.ndarray:
        """Compute pairwise Euclidean distances."""
        dists = np.zeros((self.n_points, self.n_points))
        for i in range(self.n_points):
            dists[i, :] = np.sqrt(((self.X - self.X[i, :]) ** 2).mean(-1))
        return dists
    
    def compute_kernel_matrix(self) -> np.ndarray:
        """
        Compute Matérn kernel matrix.
        
        Returns:
            Kernel matrix K of shape [n_points, n_points]
        """
        dists = self._compute_distances()
        
        # Special cases for computational efficiency
        if self.nu == 0.5:
            K = np.exp(-dists)
        elif self.nu == 1.5:
            K = dists * np.sqrt(3)
            K = (1.0 + K) * np.exp(-K)
        elif self.nu == 2.5:
            K = dists * np.sqrt(5)
            K = (1.0 + K + K**2 / 3.0) * np.exp(-K)
        elif self.nu == np.inf:
            K = np.exp(-(dists**2) / 2.0)
        else:
            # General case using modified Bessel function
            K = dists.copy()
            K[K == 0.0] += np.finfo(float).eps
            tmp = np.sqrt(2 * self.nu) * K
            K.fill((2 ** (1.0 - self.nu)) / gamma(self.nu))
            K *= tmp**self.nu
            K *= kv(self.nu, tmp)
            
        np.fill_diagonal(K, 1)
        return K
    
    def sample(self,
               shape: Tuple[int, ...],
               tikhonov_reg: bool = True,
               seed: Optional[int] = None) -> torch.Tensor:
        """
        Sample from Matérn kernel distribution.
        
        Args:
            shape: Output shape for sampled functions
            tikhonov_reg: Add small diagonal regularization for numerical stability
            seed: Optional seed for deterministic sampling
            
        Returns:
            Sampled tensor of shape [n_points, *shape]
        """
        self.logger.info(f'Sampling Matérn kernel with ν={self.nu}')
        
        K = self.compute_kernel_matrix()
        
        if tikhonov_reg:
            # Add stronger regularization for numerical stability
            K[np.diag_indices_from(K)] += 1e-4
            
        generator = np.random.default_rng(seed)
        Y = generator.multivariate_normal(
            np.zeros(self.n_points), 
            K, 
            size=shape,
            check_valid='warn',
            method='cholesky'
        ).T
        
        return torch.tensor(Y, dtype=torch.float32, device=self.device)


class CompositionalDataGenerator:
    """
    Generator for compositional datasets using Matérn kernels.
    
    Creates datasets where Y = h(g(X)) with different smoothness levels
    for the constituent functions g and h.
    """
    
    def __init__(self, 
                 d_input: int = 15,
                 d_intermediate: int = 3, 
                 d_output: int = 20,
                 n_samples: int = 50000,
                 random_seed: Optional[int] = None,
                 device: str = 'cuda'):
        """
        Initialize compositional data generator.
        
        Args:
            d_input: Input dimension
            d_intermediate: Intermediate dimension (bottleneck)
            d_output: Output dimension
            n_samples: Number of samples to generate
            random_seed: Optional seed for deterministic input point generation
            device: Device for computation
        """
        self.d_input = d_input
        self.d_intermediate = d_intermediate
        self.d_output = d_output
        self.n_samples = n_samples
        self.device = device
        
        self.logger = logging.getLogger(__name__)

        # Generate fixed input points once for all experiments.
        if random_seed is None:
            X_cpu = torch.randn(n_samples, d_input, dtype=torch.float32)
        else:
            generator = torch.Generator(device='cpu')
            generator.manual_seed(int(random_seed))
            X_cpu = torch.randn(
                n_samples,
                d_input,
                generator=generator,
                dtype=torch.float32
            )

        self.X = X_cpu.to(device)
        
    def generate_compositional_data(self, 
                                  nu_g: float, 
                                  nu_h: float,
                                  seed: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Generate compositional dataset Y = h(g(X)).
        
        Args:
            nu_g: Smoothness parameter for function g
            nu_h: Smoothness parameter for function h
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (X, Y) tensors
        """
        self.logger.info(f'Generating compositional data with ν_g={nu_g}, ν_h={nu_h}')

        g_seed = None if seed is None else int(2 * seed)
        h_seed = None if seed is None else int(2 * seed + 1)
        
        # Step 1: Generate intermediate representation Z = g(X)
        kernel_g = MaternKernel(self.X, nu_g, self.device)
        Z = kernel_g.sample(
            (self.d_intermediate,),
            seed=g_seed
        )  # Shape: [n_samples, d_intermediate]
        
        # Step 2: Generate output Y = h(Z)
        kernel_h = MaternKernel(Z, nu_h, self.device)
        Y = kernel_h.sample(
            (self.d_output,),
            seed=h_seed
        )  # Shape: [n_samples, d_output]
        
        return self.X, Y
    
    def generate_parameter_sweep(self, 
                               nu_values: np.ndarray,
                               save_path: str) -> None:
        """
        Generate datasets for full parameter sweep.
        
        Args:
            nu_values: Array of ν values to sweep over
            save_path: Directory to save generated datasets
        """
        import os
        import pickle
        
        os.makedirs(save_path, exist_ok=True)
        
        for nu_g in nu_values:
            for nu_h in nu_values:
                X, Y = self.generate_compositional_data(nu_g, nu_h)
                
                # Save with consistent naming convention
                data_dict = {
                    'X': X.cpu(),
                    'Y': Y.cpu(),
                    'nu_g': nu_g,
                    'nu_h': nu_h,
                    'metadata': {
                        'd_input': self.d_input,
                        'd_intermediate': self.d_intermediate,
                        'd_output': self.d_output,
                        'n_samples': self.n_samples
                    }
                }
                
                filename = f'compositional_data_nu_g_{nu_g}_nu_h_{nu_h}.pkl'
                filepath = os.path.join(save_path, filename)
                
                with open(filepath, 'wb') as f:
                    pickle.dump(data_dict, f)
