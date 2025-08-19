"""Enhanced Trainer for CPU Training Validation

This module provides an enhanced trainer class with CPU optimizations,
early stopping, advanced monitoring, and comprehensive logging capabilities.
"""

import os
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from cpu_training_config import EarlyStopping, create_cpu_optimizer, create_cpu_scheduler


class EnhancedTrainer:
    """Enhanced trainer with CPU optimizations and advanced monitoring."""
    
    def __init__(self, model: nn.Module, loss_fn: nn.Module, 
                 train_loader: DataLoader, val_loader: Optional[DataLoader] = None,
                 config: Optional[Dict[str, Any]] = None, 
                 checkpoint_dir: str = 'checkpoints',
                 log_dir: str = 'logs'):
        """
        Initialize enhanced trainer.
        
        Parameters:
        -----------
        model : nn.Module
            Model to train
        loss_fn : nn.Module
            Loss function
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader, optional
            Validation data loader
        config : Dict[str, Any], optional
            Training configuration
        checkpoint_dir : str
            Directory to save checkpoints
        log_dir : str
            Directory to save logs
        """
        self.model = model
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Default configuration
        self.config = config or {
            'device': 'cpu',
            'learning_rate': 1e-3,
            'num_epochs': 20,
            'early_stopping': {'enabled': True, 'patience': 5, 'min_delta': 1e-4},
            'gradient_clipping': {'enabled': True, 'max_norm': 1.0},
            'lr_scheduler': {'type': 'ReduceLROnPlateau', 'params': {'patience': 3, 'factor': 0.5}}
        }
        
        self.device = self.config['device']
        self.model = self.model.to(self.device)
        
        # Create optimizer and scheduler
        self.optimizer = create_cpu_optimizer(self.model, self.config)
        self.scheduler = create_cpu_scheduler(self.optimizer, self.config)
        
        # Early stopping
        if self.config['early_stopping']['enabled']:
            self.early_stopping = EarlyStopping(
                patience=self.config['early_stopping']['patience'],
                min_delta=self.config['early_stopping']['min_delta'],
                verbose=True
            )
        else:
            self.early_stopping = None
        
        # Directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Training history
        self.history = {
            'train_losses': [],
            'val_losses': [],
            'learning_rates': [],
            'epoch_times': [],
            'batch_times': [],
            'memory_usage': [],
            'gradient_norms': []
        }
        
        self.best_val_loss = float('inf')
        self.start_time = None
        self.current_epoch = 0
        
        # Logging
        self.log_file = self.log_dir / f'training_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.txt'
        self._log("Enhanced Trainer initialized")
        self._log(f"Configuration: {json.dumps(self.config, indent=2)}")
    
    def _log(self, message: str) -> None:
        """Log message to file and console."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"[{timestamp}] {message}"
        
        print(log_message)
        
        with open(self.log_file, 'a') as f:
            f.write(log_message + "\n")
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB."""
        if torch.cuda.is_available() and self.device != 'cpu':
            return torch.cuda.memory_allocated() / 1024 / 1024
        else:
            # For CPU, we'll track process memory (simplified)
            import psutil
            process = psutil.Process(os.getpid())
            return process.memory_info().rss / 1024 / 1024
    
    def _get_gradient_norm(self) -> float:
        """Calculate gradient norm."""
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** (1. / 2)
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch with detailed monitoring."""
        self.model.train()
        epoch_start_time = time.time()
        
        total_loss = 0.0
        num_batches = 0
        batch_times = []
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {self.current_epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            batch_start_time = time.time()
            
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output, latent = self.model(batch)
            
            # Calculate loss
            loss_dict = self.loss_fn(output, batch, latent)
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config['gradient_clipping']['enabled']:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 
                    self.config['gradient_clipping']['max_norm']
                )
            
            # Track gradient norm
            grad_norm = self._get_gradient_norm()
            
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            num_batches += 1
            
            batch_time = time.time() - batch_start_time
            batch_times.append(batch_time)
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/num_batches:.4f}',
                'Grad Norm': f'{grad_norm:.4f}',
                'Batch Time': f'{batch_time:.3f}s'
            })
        
        epoch_time = time.time() - epoch_start_time
        avg_loss = total_loss / num_batches
        avg_batch_time = np.mean(batch_times)
        
        # Store metrics
        self.history['epoch_times'].append(epoch_time)
        self.history['batch_times'].extend(batch_times)
        self.history['gradient_norms'].append(grad_norm)
        
        return {
            'loss': avg_loss,
            'epoch_time': epoch_time,
            'avg_batch_time': avg_batch_time,
            'gradient_norm': grad_norm
        }
    
    def validate(self) -> Optional[Dict[str, float]]:
        """Validate the model with detailed metrics."""
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        val_start_time = time.time()
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                batch = batch.to(self.device)
                
                # Forward pass
                output, latent = self.model(batch)
                
                # Calculate loss
                loss_dict = self.loss_fn(output, batch, latent)
                loss = loss_dict['total_loss']
                
                total_loss += loss.item()
                num_batches += 1
        
        val_time = time.time() - val_start_time
        avg_loss = total_loss / num_batches
        
        return {
            'loss': avg_loss,
            'val_time': val_time
        }
    
    def save_checkpoint(self, epoch: int, metrics: Dict[str, float], 
                       is_best: bool = False) -> None:
        """Save comprehensive checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'history': self.history,
            'metrics': metrics,
            'best_val_loss': self.best_val_loss,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            self._log(f"New best model saved with validation loss: {metrics.get('val_loss', 'N/A'):.4f}")
        
        self._log(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str) -> Dict[str, Any]:
        """Load checkpoint and restore training state."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.history = checkpoint.get('history', self.history)
        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.current_epoch = checkpoint['epoch']
        
        self._log(f"Checkpoint loaded from: {checkpoint_path}")
        return checkpoint
    
    def train(self, num_epochs: Optional[int] = None, save_every: int = 5) -> Dict[str, Any]:
        """Main training loop with comprehensive monitoring."""
        if num_epochs is None:
            num_epochs = self.config['num_epochs']
        
        self._log(f"Starting training for {num_epochs} epochs")
        self._log(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        self._log(f"Training on device: {self.device}")
        
        self.start_time = time.time()
        training_interrupted = False
        
        try:
            for epoch in range(1, num_epochs + 1):
                self.current_epoch = epoch
                
                self._log(f"\nEpoch {epoch}/{num_epochs}")
                self._log("-" * 50)
                
                # Train
                train_metrics = self.train_epoch()
                self.history['train_losses'].append(train_metrics['loss'])
                
                # Validate
                val_metrics = self.validate()
                if val_metrics is not None:
                    self.history['val_losses'].append(val_metrics['loss'])
                    
                    # Learning rate scheduling
                    if hasattr(self.scheduler, 'step'):
                        if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                            self.scheduler.step(val_metrics['loss'])
                        else:
                            self.scheduler.step()
                    
                    # Check if best model
                    is_best = val_metrics['loss'] < self.best_val_loss
                    if is_best:
                        self.best_val_loss = val_metrics['loss']
                else:
                    is_best = False
                
                # Track learning rate
                current_lr = self.optimizer.param_groups[0]['lr']
                self.history['learning_rates'].append(current_lr)
                
                # Track memory usage
                memory_usage = self._get_memory_usage()
                self.history['memory_usage'].append(memory_usage)
                
                # Log epoch summary
                self._log(f"Train Loss: {train_metrics['loss']:.6f}")
                if val_metrics:
                    self._log(f"Val Loss: {val_metrics['loss']:.6f}")
                self._log(f"Learning Rate: {current_lr:.8f}")
                self._log(f"Epoch Time: {train_metrics['epoch_time']:.2f}s")
                self._log(f"Memory Usage: {memory_usage:.1f} MB")
                
                # Early stopping
                if self.early_stopping and val_metrics:
                    if self.early_stopping(val_metrics['loss'], self.model):
                        self._log("Early stopping triggered")
                        break
                
                # Save checkpoint
                if epoch % save_every == 0 or is_best:
                    combined_metrics = {**train_metrics}
                    if val_metrics:
                        combined_metrics.update({f'val_{k}': v for k, v in val_metrics.items()})
                    self.save_checkpoint(epoch, combined_metrics, is_best)
        
        except KeyboardInterrupt:
            self._log("Training interrupted by user")
            training_interrupted = True
        
        except Exception as e:
            self._log(f"Training failed with error: {str(e)}")
            training_interrupted = True
            raise
        
        finally:
            total_time = time.time() - self.start_time
            self._log(f"\nTraining completed in {total_time/3600:.2f} hours")
            
            # Generate final report
            final_report = self._generate_training_report(total_time, training_interrupted)
            
            # Save training history
            self._save_training_history()
            
            # Plot training curves
            self._plot_training_curves()
            
            return final_report
    
    def _generate_training_report(self, total_time: float, interrupted: bool) -> Dict[str, Any]:
        """Generate comprehensive training report."""
        report = {
            'training_completed': not interrupted,
            'total_time_hours': total_time / 3600,
            'epochs_completed': len(self.history['train_losses']),
            'final_train_loss': self.history['train_losses'][-1] if self.history['train_losses'] else None,
            'final_val_loss': self.history['val_losses'][-1] if self.history['val_losses'] else None,
            'best_val_loss': self.best_val_loss if self.best_val_loss != float('inf') else None,
            'avg_epoch_time': np.mean(self.history['epoch_times']) if self.history['epoch_times'] else None,
            'avg_batch_time': np.mean(self.history['batch_times']) if self.history['batch_times'] else None,
            'peak_memory_usage': max(self.history['memory_usage']) if self.history['memory_usage'] else None,
            'config': self.config,
            'convergence_analysis': self._analyze_convergence()
        }
        
        # Save report
        report_path = self.log_dir / f'training_report_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        self._log(f"Training report saved: {report_path}")
        return report
    
    def _analyze_convergence(self) -> Dict[str, Any]:
        """Analyze training convergence."""
        if len(self.history['train_losses']) < 2:
            return {'status': 'insufficient_data'}
        
        train_losses = self.history['train_losses']
        
        # Calculate loss reduction
        initial_loss = train_losses[0]
        final_loss = train_losses[-1]
        loss_reduction = (initial_loss - final_loss) / initial_loss
        
        # Check for convergence (loss reduction > 10% and stable)
        converged = loss_reduction > 0.1
        
        # Check for overfitting
        overfitting = False
        if self.history['val_losses'] and len(self.history['val_losses']) > 5:
            val_losses = self.history['val_losses']
            # Simple overfitting check: validation loss increasing while training loss decreasing
            recent_val_trend = np.polyfit(range(len(val_losses[-5:])), val_losses[-5:], 1)[0]
            recent_train_trend = np.polyfit(range(len(train_losses[-5:])), train_losses[-5:], 1)[0]
            overfitting = recent_val_trend > 0 and recent_train_trend < 0
        
        return {
            'status': 'converged' if converged else 'not_converged',
            'loss_reduction_percent': loss_reduction * 100,
            'overfitting_detected': overfitting,
            'final_gradient_norm': self.history['gradient_norms'][-1] if self.history['gradient_norms'] else None
        }
    
    def _save_training_history(self) -> None:
        """Save detailed training history."""
        history_path = self.log_dir / f'training_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_history = {}
        for key, value in self.history.items():
            if isinstance(value, list):
                serializable_history[key] = [float(x) if isinstance(x, (np.floating, np.integer)) else x for x in value]
            else:
                serializable_history[key] = value
        
        with open(history_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
        
        self._log(f"Training history saved: {history_path}")
    
    def _plot_training_curves(self) -> None:
        """Create comprehensive training visualization."""
        if not self.history['train_losses']:
            return
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle('Enhanced Training Analysis', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(self.history['train_losses']) + 1)
        
        # Plot 1: Loss curves
        axes[0, 0].plot(epochs, self.history['train_losses'], 'b-', label='Training', linewidth=2)
        if self.history['val_losses']:
            axes[0, 0].plot(epochs, self.history['val_losses'], 'r-', label='Validation', linewidth=2)
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Learning rate
        if self.history['learning_rates']:
            axes[0, 1].plot(epochs, self.history['learning_rates'], 'g-', linewidth=2)
            axes[0, 1].set_title('Learning Rate Schedule')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('Learning Rate')
            axes[0, 1].set_yscale('log')
            axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Memory usage
        if self.history['memory_usage']:
            axes[0, 2].plot(epochs, self.history['memory_usage'], 'm-', linewidth=2)
            axes[0, 2].set_title('Memory Usage')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Memory (MB)')
            axes[0, 2].grid(True, alpha=0.3)
        
        # Plot 4: Epoch times
        if self.history['epoch_times']:
            axes[1, 0].plot(epochs, self.history['epoch_times'], 'c-', linewidth=2)
            axes[1, 0].set_title('Epoch Training Time')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Time (seconds)')
            axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 5: Gradient norms
        if self.history['gradient_norms']:
            axes[1, 1].plot(epochs, self.history['gradient_norms'], 'orange', linewidth=2)
            axes[1, 1].set_title('Gradient Norms')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Gradient Norm')
            axes[1, 1].grid(True, alpha=0.3)
        
        # Plot 6: Batch time distribution
        if self.history['batch_times']:
            axes[1, 2].hist(self.history['batch_times'], bins=30, alpha=0.7, color='purple')
            axes[1, 2].set_title('Batch Time Distribution')
            axes[1, 2].set_xlabel('Time (seconds)')
            axes[1, 2].set_ylabel('Frequency')
            axes[1, 2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.checkpoint_dir / 'enhanced_training_analysis.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        self._log(f"Training analysis plots saved: {plot_path}")


if __name__ == "__main__":
    print("Enhanced Trainer for CPU Training Validation")
    print("This module provides advanced training capabilities with comprehensive monitoring.")