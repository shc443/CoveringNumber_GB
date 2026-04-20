"""
Shallow Neural Network Implementation

Single hidden layer network used as baseline in compositionality research.
"""

import torch
import torch.nn as nn
from typing import List, Dict
from .base_net import BaseNet


class ShallowNet(BaseNet):
    """
    Shallow neural network with single hidden layer.
    
    Used to demonstrate limitations of shallow architectures
    in learning compositional representations compared to deep networks.
    """
    
    def __init__(self, 
                 widths: List[int],
                 **kwargs):
        """
        Initialize shallow network.
        
        Args:
            widths: Layer widths [d_input, hidden_width, d_output]
            **kwargs: Additional arguments for BaseNet
        """
        if len(widths) != 3:
            raise ValueError("Shallow network requires exactly 3 widths: [input, hidden, output]")
            
        super().__init__(widths, **kwargs)
        
        self.hidden_layer = nn.Linear(widths[0], widths[1])
        self.output_layer = nn.Linear(widths[1], widths[2])
        
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
        Forward pass through shallow network.
        
        Args:
            x: Input tensor of shape [batch_size, d_input]
            
        Returns:
            Output tensor of shape [batch_size, d_output]
        """
        # Clear previous activations
        self.activations = [x]
        self.pre_activations = [x]
        
        # Hidden layer
        z_hidden = self.hidden_layer(x)
        self.pre_activations.append(z_hidden)
        z_hidden = self.nonlin(z_hidden)
        self.activations.append(z_hidden)
        
        # Output layer (no nonlinearity)
        z_output = self.output_layer(z_hidden)
        self.pre_activations.append(z_output)
        self.activations.append(z_output)
        
        return z_output
    
    def compute_complexity_bounds(self, X_train: torch.Tensor) -> Dict[str, float]:
        """
        Compute complexity bounds for shallow networks.
        
        Shallow networks have simpler complexity analysis due to
        their limited depth.
        """
        # Get network statistics
        norm_hidden = torch.norm(self.hidden_layer.weight, p='fro').item()
        norm_output = torch.norm(self.output_layer.weight, p='fro').item()
        
        lip_hidden = torch.linalg.matrix_norm(self.hidden_layer.weight, ord=2).item()
        lip_output = torch.linalg.matrix_norm(self.output_layer.weight, ord=2).item()
        
        n_samples = X_train.shape[0]
        input_scale = torch.linalg.norm(X_train, dim=1).max().item()
        
        # For shallow networks, most bounds simplify significantly
        lipschitz_product = lip_hidden * lip_output
        total_frobenius = norm_hidden + norm_output
        
        # Rademacher-based bound for shallow networks
        bound_rademacher = (lipschitz_product * input_scale) / (n_samples ** 0.5)
        
        # Frobenius-based bound
        bound_frobenius = (total_frobenius * input_scale) / (n_samples ** 0.5)
        
        return {
            'rademacher_bound': bound_rademacher,
            'frobenius_bound': bound_frobenius,
            'lipschitz_product': lipschitz_product,
            'total_norm': norm_hidden ** 2 + norm_output ** 2,
            'hidden_width': self.widths[1],
            'depth': 2
        }