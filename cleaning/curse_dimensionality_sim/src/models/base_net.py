"""
Base Neural Network Architecture for Compositionality Research

Abstract base class defining the interface for all network architectures
used in studying how DNNs break the curse of dimensionality.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple, Dict, Any
from math import prod


class BaseNet(nn.Module, ABC):
    """
    Abstract base class for all network architectures in compositionality research.
    
    Defines unified interface for training, complexity analysis, and evaluation
    across different architectures (Deep, Accordion, Shallow).
    """
    
    def __init__(self, 
                 widths: List[int], 
                 nonlin: callable = F.relu,
                 loss_type: str = 'L2',
                 test_loss_type: str = 'L2',
                 prof_jacot: bool = False,
                 reduction: str = 'mean',
                 device: str = 'cuda'):
        """
        Initialize base network.
        
        Args:
            widths: Layer widths [d_in, hidden_widths..., d_out]
            nonlin: Activation function
            loss_type: Training loss ('L1', 'L2', 'fro', 'nuc')
            test_loss_type: Test loss type
            prof_jacot: Use Prof. Jacot's L1 definition
            reduction: Loss reduction method ('mean', 'sum')
            device: Device for computation ('cuda' or 'cpu')
        """
        super().__init__()
        self.widths = widths
        self.nonlin = nonlin
        self.loss_type = loss_type
        self.test_loss_type = test_loss_type
        self.prof_jacot = prof_jacot
        self.reduction = reduction
        self.device_type = device
        
        # Track activations for analysis
        self.activations = []
        self.pre_activations = []
        
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through network."""
        pass
    
    def compute_loss(self, y_pred: torch.Tensor, y_true: torch.Tensor, 
                    loss_type: str = None) -> torch.Tensor:
        """
        Unified loss computation supporting multiple loss types.
        
        Args:
            y_pred: Predicted outputs
            y_true: True outputs  
            loss_type: Override default loss type
            
        Returns:
            Computed loss value
        """
        loss_type = loss_type or self.loss_type
        
        if loss_type == 'L2':
            return nn.MSELoss(reduction=self.reduction)(y_pred, y_true)
        elif loss_type == 'L1' and not self.prof_jacot:
            return nn.L1Loss(reduction=self.reduction)(y_pred, y_true)
        elif loss_type == 'L1' and self.prof_jacot:
            # Prof. Jacot's L1 definition
            return (torch.sum((y_pred - y_true) ** 2, axis=0) ** 0.5).mean()
        elif loss_type == 'fro':
            return torch.norm(y_pred - y_true, p='fro')
        elif loss_type == 'nuc':
            return torch.norm(y_pred - y_true, p='nuc')
        else:
            return torch.norm(y_pred - y_true, p='nuc')
    
    def compute_weight_norms(self) -> List[float]:
        """Compute Frobenius norms of weight matrices."""
        return [torch.norm(layer.weight, p='fro').item() 
                for layer in self.modules() if isinstance(layer, nn.Linear)]
    
    def compute_lipschitz_constants(self) -> List[float]:
        """Compute layer-wise Lipschitz constants (spectral norms)."""
        return [torch.linalg.matrix_norm(layer.weight, ord=2).item()
                for layer in self.modules() if isinstance(layer, nn.Linear)]
    
    def compute_ranks(self, atol: float = 0.1, rtol: float = 0.1) -> List[int]:
        """Compute ranks of weight matrices."""
        ranks = []
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                rank = torch.linalg.matrix_rank(layer.weight, atol=atol, rtol=rtol)
                ranks.append(rank.item())
        return ranks
    
    def compute_stable_ranks(self) -> List[float]:
        """
        Compute stable ranks: ||W||_F^2 / ||W||_2^2
        
        Stable rank provides a continuous measure of effective rank.
        """
        stable_ranks = []
        for layer in self.modules():
            if isinstance(layer, nn.Linear):
                fro_norm_sq = torch.norm(layer.weight, p='fro') ** 2
                spectral_norm_sq = torch.linalg.matrix_norm(layer.weight, ord=2) ** 2
                stable_rank = (fro_norm_sq / spectral_norm_sq).item()
                stable_ranks.append(stable_rank)
        return stable_ranks
    
    def compute_total_norm(self) -> float:
        """Compute total parameter norm."""
        return sum([(p ** 2).sum().item() for p in self.parameters()])
    
    def get_depth(self) -> int:
        """Get network depth."""
        return len(self.widths) - 1
    
    def analyze_nonlinearity_impact(self) -> np.ndarray:
        """
        Analyze impact of nonlinear activations on representations.
        
        Computes ratio of change due to nonlinearity vs linear transformation.
        """
        if not self.activations or not self.pre_activations:
            raise ValueError("Forward pass required before nonlinearity analysis")
            
        impacts = np.zeros(len(self.activations))
        for i, (post_nl, pre_nl) in enumerate(zip(self.activations, self.pre_activations)):
            if i == 0:  # Skip input layer
                continue
            change = ((post_nl - pre_nl) ** 2).sum().item()
            magnitude = (pre_nl ** 2).sum().item()
            impacts[i] = change / magnitude if magnitude > 0 else 0
            
        return impacts
    
    @abstractmethod
    def compute_complexity_bounds(self, X_train: torch.Tensor) -> Dict[str, float]:
        """
        Compute various complexity bounds for generalization analysis.
        
        Different network architectures may implement different bounds
        based on their specific structure.
        """
        pass