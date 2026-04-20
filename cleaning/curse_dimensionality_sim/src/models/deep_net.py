"""
Deep Neural Network Implementation

Standard deep network for comparison with accordion networks
in compositionality learning research.
"""

import torch
import torch.nn as nn
from typing import List, Dict
from .base_net import BaseNet
from math import prod


class DeepNet(BaseNet):
    """
    Standard deep neural network with uniform hidden layers.
    
    Used as baseline comparison to study how architectural choices
    affect the ability to learn compositional representations.
    """
    
    def __init__(self, 
                 widths: List[int],
                 **kwargs):
        """
        Initialize deep network.
        
        Args:
            widths: Layer widths [d_input, hidden_width, ..., d_output]
            **kwargs: Additional arguments for BaseNet
        """
        super().__init__(widths, **kwargs)
        
        # Create linear layers
        self.linears = nn.ModuleList([
            nn.Linear(w_in, w_out) 
            for w_in, w_out in zip(widths[:-1], widths[1:])
        ])
        
        self._initialize_weights()
        
        # Move to device
        self.to(self.device_type)
    
    def _initialize_weights(self):
        """Use PyTorch's default initialization (Kaiming/He uniform)."""
        # PyTorch's nn.Linear already uses Kaiming uniform initialization by default
        # which is optimal for ReLU networks: init.kaiming_uniform_(weight, a=math.sqrt(5))
        pass  # Let PyTorch handle it
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through deep network.
        
        Args:
            x: Input tensor of shape [batch_size, d_input]
            
        Returns:
            Output tensor of shape [batch_size, d_output]
        """
        # Clear previous activations
        self.activations = [x]
        self.pre_activations = [x]
        
        z = x
        for i, layer in enumerate(self.linears):
            z = layer(z)
            self.pre_activations.append(z)
            
            # Apply nonlinearity except for final layer
            if i < len(self.linears) - 1:
                z = self.nonlin(z)
            
            self.activations.append(z)
            
        return z
    
    def compute_complexity_bounds(self, X_train: torch.Tensor) -> Dict[str, float]:
        """
        Compute complexity bounds for deep networks.
        
        Implements various generalization bounds from literature:
        - Neyshabur et al. (2015)
        - Bartlett et al. (2017) 
        - Neyshabur et al. (2018)
        - Ledent et al. (2021)
        - Our bound (2024)
        """
        # Get network statistics
        norms = self.compute_weight_norms()
        lips = self.compute_lipschitz_constants()
        ranks = self.compute_ranks()
        stable_ranks = self.compute_stable_ranks()
        
        n_samples = X_train.shape[0]
        input_scale = torch.linalg.norm(X_train, dim=1).max().item()
        
        # Neyshabur et al. (2015): Product of spectral norms
        bound_neyshabur15 = prod(lips) * input_scale
        
        # Bartlett et al. (2017): Frobenius norm bound
        bound_bartlett17 = sum([norm ** 2 for norm in norms]) * input_scale
        
        # Neyshabur et al. (2018): Spectral complexity
        bound_neyshabur18 = prod(lips) * sum(norms) * input_scale
        
        # Ledent et al. (2021): Path norm bound
        path_norm = sum([norm * lip for norm, lip in zip(norms, lips)])
        bound_ledent21 = path_norm * input_scale
        
        # Our bound (2024): Complexity with rank considerations
        def _compute_our_bound(dims):
            lip_product = prod(lips)
            complexity_sum = sum([
                n / l * (d_in + d_out) ** 0.5 
                for n, l, d_in, d_out in zip(norms, lips, dims[:-1], dims[1:])
            ])
            return lip_product * complexity_sum / (n_samples ** 0.5)
        
        bound_ours_standard = _compute_our_bound(self.widths)
        
        return {
            'neyshabur_2015': bound_neyshabur15,
            'bartlett_2017': bound_bartlett17, 
            'neyshabur_2018': bound_neyshabur18,
            'ledent_2021': bound_ledent21,
            'ours_standard_rank': bound_ours_standard,
            'lipschitz_product': prod(lips),
            'total_norm': self.compute_total_norm(),
            'depth': len(self.linears),
            'width': max(self.widths[1:-1]) if len(self.widths) > 2 else self.widths[1]
        }