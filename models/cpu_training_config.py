"""CPU Training Configuration Optimization

This module provides optimized training configurations for CPU-based training,
including batch size optimization, learning rate scheduling, and early stopping
for efficient training on limited computational resources.
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, CosineAnnealingLR
import json
from pathlib import Path
from typing import Dict, Any, Optional, Tuple


class CPUTrainingConfig:
    """Configuration class for CPU-optimized training."""
    
    def __init__(self, dataset_size: int, available_memory_gb: float = 8.0):
        """
        Initialize CPU training configuration.
        
        Parameters:
        -----------
        dataset_size : int
            Number of samples in the dataset
        available_memory_gb : float
            Available system memory in GB
        """
        self.dataset_size = dataset_size
        self.available_memory_gb = available_memory_gb
        
        # Base configuration
        self.base_config = {
            'device': 'cpu',
            'mixed_precision': False,  # Not supported on CPU
            'pin_memory': False,       # Not beneficial for CPU
            'num_workers': 0,          # CPU training works best with 0 workers
        }
        
        # Generate optimized configuration
        self.config = self._generate_optimized_config()
    
    def _generate_optimized_config(self) -> Dict[str, Any]:
        """Generate optimized configuration based on dataset size and resources."""
        config = self.base_config.copy()
        
        # Optimize batch size based on dataset size and memory
        config['batch_size'] = self._optimize_batch_size()
        
        # Optimize learning rate based on batch size
        config['learning_rate'] = self._optimize_learning_rate(config['batch_size'])
        
        # Set number of epochs based on dataset size
        config['num_epochs'] = self._optimize_num_epochs()
        
        # Configure early stopping
        config['early_stopping'] = self._configure_early_stopping()
        
        # Configure learning rate scheduling
        config['lr_scheduler'] = self._configure_lr_scheduler()
        
        # Configure gradient clipping
        config['gradient_clipping'] = {
            'enabled': True,
            'max_norm': 1.0
        }
        
        # Configure checkpointing
        config['checkpointing'] = self._configure_checkpointing()
        
        return config
    
    def _optimize_batch_size(self) -> int:
        """Optimize batch size for CPU training."""
        # CPU training benefits from smaller batch sizes
        if self.dataset_size <= 10:
            return 2
        elif self.dataset_size <= 50:
            return 4
        elif self.dataset_size <= 200:
            return 8
        else:
            return 16
    
    def _optimize_learning_rate(self, batch_size: int) -> float:
        """Optimize learning rate based on batch size."""
        # Scale learning rate with batch size (linear scaling rule)
        base_lr = 1e-3
        base_batch_size = 8
        
        # For smaller batch sizes, use slightly higher learning rates
        if batch_size <= 4:
            return base_lr * 1.5
        else:
            return base_lr * (batch_size / base_batch_size)
    
    def _optimize_num_epochs(self) -> int:
        """Optimize number of epochs based on dataset size."""
        # Smaller datasets need more epochs to converge
        if self.dataset_size <= 10:
            return 50
        elif self.dataset_size <= 50:
            return 30
        elif self.dataset_size <= 200:
            return 20
        else:
            return 15
    
    def _configure_early_stopping(self) -> Dict[str, Any]:
        """Configure early stopping parameters."""
        return {
            'enabled': True,
            'patience': max(5, self.dataset_size // 10),  # Adaptive patience
            'min_delta': 1e-4,
            'restore_best_weights': True,
            'monitor': 'val_loss',
            'mode': 'min'
        }
    
    def _configure_lr_scheduler(self) -> Dict[str, Any]:
        """Configure learning rate scheduler."""
        return {
            'type': 'ReduceLROnPlateau',
            'params': {
                'mode': 'min',
                'factor': 0.5,
                'patience': max(3, self.dataset_size // 20),
                'verbose': True,
                'min_lr': 1e-6
            }
        }
    
    def _configure_checkpointing(self) -> Dict[str, Any]:
        """Configure checkpointing strategy."""
        # More frequent checkpointing for smaller datasets
        num_epochs = self._optimize_num_epochs()
        save_frequency = max(2, num_epochs // 5)
        
        return {
            'save_every': save_frequency,
            'save_best_only': True,
            'save_last': True,
            'monitor': 'val_loss',
            'mode': 'min'
        }
    
    def get_config(self) -> Dict[str, Any]:
        """Get the complete configuration."""
        return self.config.copy()
    
    def save_config(self, filepath: str) -> None:
        """Save configuration to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.config, f, indent=2)
    
    def print_config(self) -> None:
        """Print configuration in a readable format."""
        print("🔧 CPU Training Configuration:")
        print("=" * 50)
        
        print(f"Dataset Size: {self.dataset_size}")
        print(f"Available Memory: {self.available_memory_gb} GB")
        print()
        
        print("Training Parameters:")
        print(f"  Batch Size: {self.config['batch_size']}")
        print(f"  Learning Rate: {self.config['learning_rate']:.6f}")
        print(f"  Number of Epochs: {self.config['num_epochs']}")
        print(f"  Device: {self.config['device']}")
        print()
        
        print("Early Stopping:")
        es = self.config['early_stopping']
        print(f"  Enabled: {es['enabled']}")
        print(f"  Patience: {es['patience']}")
        print(f"  Min Delta: {es['min_delta']}")
        print()
        
        print("Learning Rate Scheduler:")
        lr_sched = self.config['lr_scheduler']
        print(f"  Type: {lr_sched['type']}")
        print(f"  Patience: {lr_sched['params']['patience']}")
        print(f"  Factor: {lr_sched['params']['factor']}")
        print()
        
        print("Checkpointing:")
        cp = self.config['checkpointing']
        print(f"  Save Every: {cp['save_every']} epochs")
        print(f"  Save Best Only: {cp['save_best_only']}")
        print()
        
        print("=" * 50)


class EarlyStopping:
    """Early stopping utility for training."""
    
    def __init__(self, patience: int = 7, min_delta: float = 0, 
                 restore_best_weights: bool = True, verbose: bool = True):
        """
        Initialize early stopping.
        
        Parameters:
        -----------
        patience : int
            Number of epochs to wait before stopping
        min_delta : float
            Minimum change to qualify as improvement
        restore_best_weights : bool
            Whether to restore best weights when stopping
        verbose : bool
            Whether to print messages
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.verbose = verbose
        
        self.best_loss = float('inf')
        self.counter = 0
        self.best_weights = None
        self.early_stop = False
    
    def __call__(self, val_loss: float, model: torch.nn.Module) -> bool:
        """
        Check if training should stop early.
        
        Parameters:
        -----------
        val_loss : float
            Current validation loss
        model : torch.nn.Module
            Model to potentially save weights from
        
        Returns:
        --------
        bool
            Whether to stop training
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
            if self.verbose:
                print(f"Early stopping: validation loss improved to {val_loss:.6f}")
        else:
            self.counter += 1
            if self.verbose:
                print(f"Early stopping: no improvement for {self.counter}/{self.patience} epochs")
            
            if self.counter >= self.patience:
                self.early_stop = True
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                    if self.verbose:
                        print("Early stopping: restored best weights")
                return True
        
        return False


def create_cpu_optimizer(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Create optimizer optimized for CPU training."""
    return optim.Adam(
        model.parameters(),
        lr=config['learning_rate'],
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-5  # Light regularization
    )


def create_cpu_scheduler(optimizer: torch.optim.Optimizer, 
                        config: Dict[str, Any]) -> torch.optim.lr_scheduler._LRScheduler:
    """Create learning rate scheduler optimized for CPU training."""
    scheduler_config = config['lr_scheduler']
    
    if scheduler_config['type'] == 'ReduceLROnPlateau':
        return ReduceLROnPlateau(optimizer, **scheduler_config['params'])
    elif scheduler_config['type'] == 'StepLR':
        return StepLR(optimizer, **scheduler_config['params'])
    elif scheduler_config['type'] == 'CosineAnnealingLR':
        return CosineAnnealingLR(optimizer, **scheduler_config['params'])
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_config['type']}")


def get_cpu_training_recommendations(dataset_size: int, 
                                   available_memory_gb: float = 8.0) -> Dict[str, Any]:
    """Get training recommendations for CPU-based training."""
    config_generator = CPUTrainingConfig(dataset_size, available_memory_gb)
    config = config_generator.get_config()
    
    recommendations = {
        'config': config,
        'tips': [
            "Use smaller batch sizes (2-8) for better CPU performance",
            "Enable gradient clipping to prevent exploding gradients",
            "Use early stopping to prevent overfitting on small datasets",
            "Monitor training closely due to limited computational resources",
            "Consider data augmentation if dataset is very small",
            "Save checkpoints frequently in case of interruption"
        ],
        'expected_performance': {
            'training_time_multiplier': '10-50x slower than GPU',
            'memory_usage': 'Lower than GPU training',
            'convergence': 'May require more epochs for small datasets'
        }
    }
    
    return recommendations


def create_training_config_template(output_dir: str = 'outputs') -> str:
    """Create a training configuration template file."""
    template = {
        'small_dataset': get_cpu_training_recommendations(15),
        'medium_dataset': get_cpu_training_recommendations(100),
        'large_dataset': get_cpu_training_recommendations(500),
        'usage_instructions': [
            "1. Choose configuration based on your dataset size",
            "2. Adjust batch_size based on available memory",
            "3. Modify learning_rate if training is unstable",
            "4. Increase patience for early stopping if dataset is noisy",
            "5. Monitor training curves and adjust as needed"
        ]
    }
    
    output_path = Path(output_dir) / 'cpu_training_config_templates.json'
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(template, f, indent=2)
    
    return str(output_path)


if __name__ == "__main__":
    # Example usage
    print("CPU Training Configuration Generator")
    print("=" * 50)
    
    # Test with different dataset sizes
    for size in [10, 50, 200]:
        print(f"\nConfiguration for dataset size: {size}")
        config_gen = CPUTrainingConfig(size)
        config_gen.print_config()
    
    # Create template file
    template_path = create_training_config_template()
    print(f"\nTraining configuration templates saved to: {template_path}")