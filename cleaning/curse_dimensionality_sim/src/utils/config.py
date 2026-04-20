"""
Configuration Management Utilities

Handles loading and validation of experiment configurations.
"""

import yaml
import torch
import torch.nn.functional as F
from typing import Dict, Any
import os


class ConfigManager:
    """Manages experiment configuration loading and validation."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict[str, Any]:
        """
        Load experiment configuration from YAML file.
        
        Args:
            config_path: Path to configuration file
            
        Returns:
            Configuration dictionary
        """
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
            
        # Validate configuration
        ConfigManager._validate_config(config)
        
        # Process special values
        config = ConfigManager._process_config(config)
        
        return config
    
    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> None:
        """Validate configuration structure and values."""
        required_sections = ['data', 'models', 'training', 'sweep', 'output']
        
        for section in required_sections:
            if section not in config:
                raise ValueError(f"Missing required configuration section: {section}")
        
        # Validate data configuration
        data_config = config['data']
        required_data_keys = ['d_input', 'd_intermediate', 'd_output', 'n_samples']
        for key in required_data_keys:
            if key not in data_config:
                raise ValueError(f"Missing required data parameter: {key}")
                
        # Validate model configurations
        models_config = config['models']
        for arch_name, arch_config in models_config.items():
            if 'nonlin' not in arch_config:
                raise ValueError(f"Missing nonlin for architecture: {arch_name}")
        
        # Validate training configuration
        training_config = config['training']
        required_training_keys = ['test_size', 'num_trials', 'loss_type']
        for key in required_training_keys:
            if key not in training_config:
                raise ValueError(f"Missing required training parameter: {key}")
    
    @staticmethod
    def _process_config(config: Dict[str, Any]) -> Dict[str, Any]:
        """Process configuration to convert string values to actual objects."""
        # Convert activation function names to actual functions
        activation_map = {
            'relu': F.relu,
            'leaky_relu': F.leaky_relu,
            'tanh': F.tanh,
            'sigmoid': F.sigmoid,
            'gelu': F.gelu
        }
        
        for arch_name, arch_config in config['models'].items():
            if 'nonlin' in arch_config:
                nonlin_name = arch_config['nonlin']
                if nonlin_name in activation_map:
                    arch_config['nonlin'] = activation_map[nonlin_name]
                else:
                    raise ValueError(f"Unknown activation function: {nonlin_name}")
        
        # Set device
        if 'hardware' in config and 'device' in config['hardware']:
            device = config['hardware']['device']
            if device == 'cuda' and not torch.cuda.is_available():
                print("CUDA not available, falling back to CPU")
                config['hardware']['device'] = 'cpu'
        
        return config
    
    @staticmethod
    def create_default_config(save_path: str) -> None:
        """Create a default configuration file."""
        # This would create the YAML content as a template
        # Implementation omitted for brevity
        pass