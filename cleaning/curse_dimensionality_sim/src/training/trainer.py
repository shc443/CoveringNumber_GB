"""
Training Framework for Compositionality Research

Unified training pipeline supporting multi-stage training with
different learning rates and regularization schedules.
"""

import copy
import torch
import torch.optim as optim
from typing import Dict, Optional, Tuple, Any
import logging
from ..models.base_net import BaseNet


class CompositionTrainer:
    """
    Multi-stage trainer for studying compositionality learning.
    
    Implements the training schedule used in the research:
    1. Initial high learning rate phase
    2. Medium learning rate with light regularization
    3. Fine-tuning with stronger regularization
    """
    
    def __init__(self, 
                 model: BaseNet,
                 device: str = 'cuda'):
        """
        Initialize trainer.
        
        Args:
            model: Neural network to train
            device: Device for computation
        """
        self.model = model.to(device)
        self.device = device
        self.optimizer = None
        
        self.logger = logging.getLogger(__name__)
        
        # Training history
        self.train_losses = []
        self.test_losses = []
        self.complexity_history = []
        
    def train_stage(self,
                   X_train: torch.Tensor,
                   Y_train: torch.Tensor,
                   X_test: torch.Tensor,
                   Y_test: torch.Tensor,
                   lr: float,
                   weight_decay: float = 0.0,
                   epochs: int = 1200,
                   num_batches: int = 5,
                   log_interval: int = 100) -> Tuple[float, float]:
        """
        Train for one stage with specified hyperparameters.
        
        Args:
            X_train, Y_train: Training data
            X_test, Y_test: Test data  
            lr: Learning rate
            weight_decay: L2 regularization strength
            epochs: Number of training epochs
            num_batches: Number of mini-batches per epoch
            log_interval: Logging frequency
            
        Returns:
            Tuple of (final_train_loss, final_test_loss)
        """
        # Setup optimizer for this stage
        if self.optimizer is None:
            self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        else:
            # Update learning rate and weight decay
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
                param_group['weight_decay'] = weight_decay
        
        batch_size = X_train.shape[0] // num_batches
        
        stage_train_losses = []
        stage_test_losses = []
        
        for epoch in range(epochs):
            epoch_loss = 0.0
            
            for batch_idx in range(num_batches):
                start_idx = batch_idx * batch_size
                end_idx = start_idx + batch_size
                
                X_batch = X_train[start_idx:end_idx]
                Y_batch = Y_train[start_idx:end_idx]
                
                # Forward pass
                self.optimizer.zero_grad()
                Y_pred = self.model(X_batch)
                loss = self.model.compute_loss(Y_pred, Y_batch)
                
                # Backward pass
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            epoch_loss /= num_batches
            stage_train_losses.append(epoch_loss)
            
            # Evaluate on test set
            if epoch % log_interval == 0 or epoch == epochs - 1:
                with torch.no_grad():
                    Y_test_pred = self.model(X_test)
                    test_loss = self.model.compute_loss(
                        Y_test_pred, Y_test, self.model.test_loss_type
                    ).item()
                    stage_test_losses.append(test_loss)
                    
                    self.logger.info(
                        f'Epoch {epoch:4d}: Train Loss = {epoch_loss:.6f}, '
                        f'Test Loss = {test_loss:.6f}'
                    )
        
        self.train_losses.extend(stage_train_losses)
        self.test_losses.extend(stage_test_losses)
        
        return stage_train_losses[-1], stage_test_losses[-1]
    
    def train_full_schedule(self,
                          X_train: torch.Tensor,
                          Y_train: torch.Tensor,
                          X_test: torch.Tensor,
                          Y_test: torch.Tensor,
                          lr_scale: float = 1.0,
                          base_weight_decay: float = 0.0,
                          stage_config: Optional[Dict[str, Dict[str, Any]]] = None) -> Dict[str, float]:
        """
        Execute full multi-stage training schedule.
        
        Follows the training protocol from the compositionality research:
        1. High LR, no regularization
        2. Medium LR, light regularization  
        3. Low LR, stronger regularization
        
        Args:
            X_train, Y_train: Training data
            X_test, Y_test: Test data
            lr_scale: Global learning rate scaling factor
            base_weight_decay: Base weight decay (scaled for each stage)
            stage_config: Optional stage config with keys stage_1/2/3 and
                per-stage keys lr, weight_decay, epochs
            
        Returns:
            Dictionary with final metrics
        """
        self.logger.info("Starting multi-stage training schedule")
        
        default_stages = {
            'stage_1': {'lr': 1.5e-3, 'weight_decay': 0.0, 'epochs': 1200},
            'stage_2': {'lr': 4.0e-4, 'weight_decay': 2.0e-3, 'epochs': 1200},
            'stage_3': {'lr': 1.0e-4, 'weight_decay': 5.0e-3, 'epochs': 1200}
        }

        resolved_stages = copy.deepcopy(default_stages)
        if stage_config:
            for stage_name, stage_values in stage_config.items():
                if stage_name in resolved_stages:
                    resolved_stages[stage_name].update(stage_values)

        # Stage 1: Initial training with high learning rate
        self.logger.info("Stage 1: High learning rate, no regularization")
        stage_1 = resolved_stages['stage_1']
        train_loss_1, test_loss_1 = self.train_stage(
            X_train, Y_train, X_test, Y_test,
            lr=stage_1['lr'] * lr_scale,
            weight_decay=base_weight_decay + stage_1['weight_decay'],
            epochs=int(stage_1['epochs'])
        )
        
        # Stage 2: Medium learning rate with light regularization
        self.logger.info("Stage 2: Medium learning rate, light regularization")
        stage_2 = resolved_stages['stage_2']
        train_loss_2, test_loss_2 = self.train_stage(
            X_train, Y_train, X_test, Y_test,
            lr=stage_2['lr'] * lr_scale,
            weight_decay=base_weight_decay + stage_2['weight_decay'],
            epochs=int(stage_2['epochs'])
        )
        
        # Stage 3: Fine-tuning with stronger regularization
        self.logger.info("Stage 3: Fine-tuning with stronger regularization")
        stage_3 = resolved_stages['stage_3']
        train_loss_3, test_loss_3 = self.train_stage(
            X_train, Y_train, X_test, Y_test,
            lr=stage_3['lr'] * lr_scale,
            weight_decay=base_weight_decay + stage_3['weight_decay'],
            epochs=int(stage_3['epochs'])
        )
        
        # Compute final complexity measures
        complexity_metrics = self.model.compute_complexity_bounds(X_train)
        self.complexity_history.append(complexity_metrics)
        
        return {
            'final_train_loss': train_loss_3,
            'final_test_loss': test_loss_3,
            'total_norm': self.model.compute_total_norm() / self.model.get_depth(),
            **complexity_metrics
        }
    
    def save_model(self, filepath: str) -> None:
        """Save trained model state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'train_losses': self.train_losses,
            'test_losses': self.test_losses,
            'complexity_history': self.complexity_history
        }, filepath)
        
    def load_model(self, filepath: str) -> None:
        """Load trained model state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.test_losses = checkpoint.get('test_losses', [])
        self.complexity_history = checkpoint.get('complexity_history', [])
