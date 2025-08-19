"""Training Script for Speech Translation Model

This script handles the training loop, data loading, and model checkpointing
for the multilingual speech translation system.
"""

import os
import sys
import time
import argparse
import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from models.encoder_decoder import create_model, count_parameters
from utils.framing import create_feature_matrix_advanced
from utils.denoise import preprocess_audio_complete

import librosa
import soundfile as sf


class AudioDataset(Dataset):
    """
    Dataset class for loading and preprocessing audio files.
    """
    
    def __init__(self, data_dir, max_files=None, frame_length_ms=20, 
                 hop_length_ms=10, n_features=441, sr=44100):
        """
        Initialize dataset.
        
        Parameters:
        -----------
        data_dir : str
            Directory containing audio files
        max_files : int, optional
            Maximum number of files to load
        frame_length_ms : float
            Frame length in milliseconds
        hop_length_ms : float
            Hop length in milliseconds
        n_features : int
            Number of features per frame
        sr : int
            Target sample rate
        """
        self.data_dir = Path(data_dir)
        self.frame_length_ms = frame_length_ms
        self.hop_length_ms = hop_length_ms
        self.n_features = n_features
        self.sr = sr
        
        # Find all audio files
        self.audio_files = list(self.data_dir.glob('*.wav'))
        if max_files:
            self.audio_files = self.audio_files[:max_files]
        
        print(f"Found {len(self.audio_files)} audio files")
        
        # Precompute feature matrices to avoid repeated computation
        self.feature_matrices = []
        self._precompute_features()
    
    def _precompute_features(self):
        """
        Precompute feature matrices for all audio files.
        """
        print("Precomputing feature matrices...")
        
        for audio_file in tqdm(self.audio_files, desc="Processing audio files"):
            try:
                # Load and preprocess audio
                y, sr = librosa.load(audio_file, sr=self.sr)
                y_processed = preprocess_audio_complete(y, sr)
                
                # Extract features
                feature_result = create_feature_matrix_advanced(
                    y_processed, sr, 
                    frame_length_ms=self.frame_length_ms,
                    hop_length_ms=self.hop_length_ms,
                    n_features=self.n_features,
                    include_spectral=False,
                    include_mfcc=False
                )
                
                feature_matrix = feature_result['feature_matrix']
                self.feature_matrices.append(feature_matrix)
                
            except Exception as e:
                print(f"Error processing {audio_file}: {e}")
                continue
        
        print(f"Successfully processed {len(self.feature_matrices)} files")
    
    def __len__(self):
        return len(self.feature_matrices)
    
    def __getitem__(self, idx):
        """
        Get a single item from the dataset.
        
        Returns:
        --------
        torch.Tensor
            Feature matrix as tensor
        """
        feature_matrix = self.feature_matrices[idx]
        
        # Convert to tensor
        tensor = torch.FloatTensor(feature_matrix)
        
        return tensor


def collate_fn(batch):
    """
    Custom collate function to handle variable-length sequences.
    
    Parameters:
    -----------
    batch : list
        List of tensors with potentially different sequence lengths
    
    Returns:
    --------
    torch.Tensor
        Padded batch tensor
    """
    # Find maximum sequence length in batch
    max_len = max(tensor.shape[0] for tensor in batch)
    
    # Pad sequences to max length
    padded_batch = []
    for tensor in batch:
        seq_len, n_features = tensor.shape
        if seq_len < max_len:
            # Pad with zeros
            padding = torch.zeros(max_len - seq_len, n_features)
            padded_tensor = torch.cat([tensor, padding], dim=0)
        else:
            padded_tensor = tensor
        
        padded_batch.append(padded_tensor)
    
    # Stack into batch
    batch_tensor = torch.stack(padded_batch)
    
    return batch_tensor


class Trainer:
    """
    Training class for the speech translation model.
    """
    
    def __init__(self, model, loss_fn, train_loader, val_loader=None, 
                 device='cuda', learning_rate=1e-3, checkpoint_dir='checkpoints'):
        """
        Initialize trainer.
        
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
        device : str
            Device to use for training
        learning_rate : float
            Learning rate
        checkpoint_dir : str
            Directory to save checkpoints
        """
        self.model = model.to(device)
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Optimizer
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
        
        # Checkpoint directory
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.best_val_loss = float('inf')
        
    def train_epoch(self):
        """
        Train for one epoch.
        
        Returns:
        --------
        float
            Average training loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training")
        
        for batch in progress_bar:
            batch = batch.to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            output, latent = self.model(batch)
            
            # Calculate loss (autoencoder: input = target)
            loss_dict = self.loss_fn(output, batch, latent)
            loss = loss_dict['total_loss']
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # Update statistics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{total_loss/num_batches:.4f}'
            })
        
        return total_loss / num_batches
    
    def validate(self):
        """
        Validate the model.
        
        Returns:
        --------
        float
            Average validation loss
        """
        if self.val_loader is None:
            return None
        
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
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
        
        return total_loss / num_batches
    
    def save_checkpoint(self, epoch, val_loss=None, is_best=False):
        """
        Save model checkpoint.
        
        Parameters:
        -----------
        epoch : int
            Current epoch
        val_loss : float, optional
            Validation loss
        is_best : bool
            Whether this is the best model so far
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_loss': val_loss,
            'best_val_loss': self.best_val_loss
        }
        
        # Save regular checkpoint
        checkpoint_path = self.checkpoint_dir / f'checkpoint_epoch_{epoch}.pt'
        torch.save(checkpoint, checkpoint_path)
        
        # Save best model
        if is_best:
            best_path = self.checkpoint_dir / 'best_model.pt'
            torch.save(checkpoint, best_path)
            print(f"New best model saved with validation loss: {val_loss:.4f}")
    
    def train(self, num_epochs, save_every=5):
        """
        Main training loop.
        
        Parameters:
        -----------
        num_epochs : int
            Number of epochs to train
        save_every : int
            Save checkpoint every N epochs
        """
        print(f"Starting training for {num_epochs} epochs...")
        print(f"Model has {count_parameters(self.model):,} trainable parameters")
        print(f"Training on device: {self.device}")
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            print(f"\nEpoch {epoch}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss = self.train_epoch()
            self.train_losses.append(train_loss)
            
            # Validate
            val_loss = self.validate()
            if val_loss is not None:
                self.val_losses.append(val_loss)
                self.scheduler.step(val_loss)
                
                # Check if best model
                is_best = val_loss < self.best_val_loss
                if is_best:
                    self.best_val_loss = val_loss
            else:
                is_best = False
            
            # Print epoch summary
            print(f"Train Loss: {train_loss:.4f}")
            if val_loss is not None:
                print(f"Val Loss: {val_loss:.4f}")
            print(f"Learning Rate: {self.optimizer.param_groups[0]['lr']:.6f}")
            
            # Save checkpoint
            if epoch % save_every == 0 or is_best:
                self.save_checkpoint(epoch, val_loss, is_best)
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time/3600:.2f} hours")
        
        # Plot training curves
        self.plot_training_curves()
    
    def plot_training_curves(self):
        """
        Plot training and validation loss curves.
        """
        plt.figure(figsize=(10, 6))
        
        epochs = range(1, len(self.train_losses) + 1)
        plt.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        
        if self.val_losses:
            plt.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        
        # Save plot
        plot_path = self.checkpoint_dir / 'training_curves.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"Training curves saved to {plot_path}")


def main():
    """
    Main training function.
    """
    parser = argparse.ArgumentParser(description='Train Speech Translation Model')
    parser.add_argument('--data_dir', type=str, required=True,
                       help='Directory containing audio files')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=8,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3,
                       help='Learning rate')
    parser.add_argument('--max_files', type=int, default=None,
                       help='Maximum number of files to use')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create dataset
    print("Loading dataset...")
    dataset = AudioDataset(
        data_dir=args.data_dir,
        max_files=args.max_files
    )
    
    # Split dataset (80% train, 20% validation)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        collate_fn=collate_fn,
        num_workers=2
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn,
        num_workers=2
    ) if val_size > 0 else None
    
    # Create model
    print("Creating model...")
    model, loss_fn = create_model()
    
    # Create trainer
    trainer = Trainer(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        val_loader=val_loader,
        device=device,
        learning_rate=args.learning_rate,
        checkpoint_dir=args.checkpoint_dir
    )
    
    # Start training
    trainer.train(num_epochs=args.epochs)
    
    print("Training completed!")


if __name__ == "__main__":
    main()