"""
Accordion Network (AccNet) Implementation

AccNets use alternating expansion and contraction layers to learn
compositional representations, as described in the paper on how DNNs
break the curse of dimensionality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict
from math import prod
from .base_net import BaseNet


class AccordionNet(BaseNet):
    """
    Accordion Network with alternating expansion/contraction structure.
    
    Architecture alternates between:
    - Expansion layers (in_linears): increase dimensionality
    - Contraction layers (out_linears): reduce dimensionality
    
    This structure encourages learning of compositional representations
    f = f_L ∘ ... ∘ f_1 where each f_i operates on intermediate representations.
    """
    
    def __init__(self, 
                 widths: List[int],
                 L: int = 5,
                 **kwargs):
        """
        Initialize Accordion Network.
        
        Args:
            widths: [d_input, d_full, d_mid, d_output] defining accordion structure
            L: Number of accordion layers (expansion-contraction pairs)
            **kwargs: Additional arguments for BaseNet
        """
        super().__init__(widths, **kwargs)
        
        self.L = L
        self.d_input = widths[0]
        self.d_full = widths[1] if len(widths) > 1 else 500
        self.d_mid = widths[2] if len(widths) > 2 else 100
        self.d_output = widths[-1]
        
        # Create expansion layers (in_linears)
        self.in_linears = nn.ModuleList()
        self.in_linears.append(nn.Linear(self.d_input, self.d_full))
        for _ in range(L - 1):
            self.in_linears.append(nn.Linear(self.d_mid, self.d_full))
            
        # Create contraction layers (out_linears)  
        self.out_linears = nn.ModuleList()
        for _ in range(L - 1):
            self.out_linears.append(nn.Linear(self.d_full, self.d_mid))
        self.out_linears.append(nn.Linear(self.d_full, self.d_output))
        
        # Initialize weights
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
        Forward pass through accordion network.
        
        Args:
            x: Input tensor of shape [batch_size, d_input]
            
        Returns:
            Output tensor of shape [batch_size, d_output]
        """
        # Clear previous activations
        self.activations = [x]
        self.pre_activations = [x]
        
        z = x
        for i in range(self.L):
            # Expansion phase
            z_expanded = self.in_linears[i](z)
            self.pre_activations.append(z_expanded)
            z_expanded = self.nonlin(z_expanded)
            self.activations.append(z_expanded)
            
            # Contraction phase
            z = self.out_linears[i](z_expanded)
            if i < self.L - 1:  # Apply nonlinearity except for final layer
                self.pre_activations.append(z)
                z = self.nonlin(z)
                self.activations.append(z)
            else:
                self.pre_activations.append(z)
                self.activations.append(z)
                
        return z
    
    def compute_accordion_norms(self) -> List[float]:
        """
        Compute norms specific to accordion structure.
        
        Combines norms from expansion and contraction layers at each level.
        """
        norms = []
        for i in range(len(self.in_linears)):
            in_norm = torch.norm(self.in_linears[i].weight, p='fro') ** 2
            out_norm = torch.norm(self.out_linears[i].weight, p='fro') ** 2
            combined_norm = (0.5 * in_norm + 0.5 * out_norm).item()
            norms.append(combined_norm)
        return norms
    
    def compute_accordion_lipschitz(self) -> List[float]:
        """
        Compute Lipschitz constants for accordion layer pairs.
        
        For each accordion layer, computes the product of expansion and
        contraction Lipschitz constants.
        """
        lipschitz_constants = []
        for i in range(len(self.in_linears)):
            lip_in = torch.linalg.matrix_norm(self.in_linears[i].weight, ord=2)
            lip_out = torch.linalg.matrix_norm(self.out_linears[i].weight, ord=2)
            lipschitz_constants.append((lip_in * lip_out).item())
        return lipschitz_constants
    
    def compute_complexity_bounds(self, X_train: torch.Tensor) -> Dict[str, float]:
        """
        Compute complexity bounds for accordion networks.
        
        Args:
            X_train: Training data for bound computation
            
        Returns:
            Dictionary containing various complexity measures
        """
        # Get network statistics
        norms = self.compute_accordion_norms()
        lips = self.compute_accordion_lipschitz()
        ranks = self.compute_ranks()
        stable_ranks = self.compute_stable_ranks()
        
        # Compute effective dimensions for each layer pair
        effective_dims = []
        for i in range(len(norms)):
            if i == 0:
                d_in, d_out = self.d_input, self.d_mid if i < len(norms) - 1 else self.d_output
            else:
                d_in, d_out = self.d_mid, self.d_mid if i < len(norms) - 1 else self.d_output
            effective_dims.append((d_in, d_out))
        
        # Standard complexity bound: ∏L_i * Σ(||W_i||_F / L_i * √(d_in + d_out))
        def _compute_bound(norms, lips, dims):
            lip_product = prod(lips)
            complexity_sum = sum([
                n / l * (d_in + d_out) ** 0.5 
                for n, l, (d_in, d_out) in zip(norms, lips, dims)
            ])
            return lip_product * complexity_sum
        
        # Compute bounds with different rank measures
        bound_standard = _compute_bound(norms, lips, [(r, r) for r in ranks])
        bound_stable = _compute_bound(norms, lips, [(r, r) for r in stable_ranks])
        
        # Normalize by dataset size and input magnitude
        rho = 1.0  # regularization constant
        input_scale = torch.linalg.norm(X_train, dim=1).max().item()
        sqrt_n = X_train.shape[0] ** 0.5
        
        normalization = rho * input_scale / sqrt_n
        
        return {
            'complexity_standard': normalization * bound_standard,
            'complexity_stable_rank': normalization * bound_stable,
            'lipschitz_product': prod(lips),
            'total_norm': self.compute_total_norm(),
            'effective_depth': len(norms),
            'norms': norms,
            'lipschitz_constants': lips
        }
    
    def get_layer_representations(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Extract intermediate representations from all layers.
        
        Args:
            x: Input tensor
            
        Returns:
            Dictionary mapping layer names to their activations
        """
        representations = {}
        
        # Forward pass to populate activations
        _ = self.forward(x)
        
        for i, (pre_act, post_act) in enumerate(zip(self.pre_activations, self.activations)):
            representations[f'layer_{i}_pre'] = pre_act
            representations[f'layer_{i}_post'] = post_act
            
        return representations