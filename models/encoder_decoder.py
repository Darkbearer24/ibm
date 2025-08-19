"""Encoder-Decoder Model Architecture

This module contains the neural network architecture for the multilingual
speech translation system using PyTorch.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional


class AudioEncoder(nn.Module):
    """
    Encoder network that compresses 441-dimensional audio features
    to 100-dimensional latent representation.
    """
    
    def __init__(self, input_dim=441, hidden_dim=256, latent_dim=100, 
                 num_layers=2, dropout=0.2, bidirectional=True):
        """
        Initialize the encoder.
        
        Parameters:
        -----------
        input_dim : int
            Input feature dimension (default: 441)
        hidden_dim : int
            Hidden layer dimension (default: 256)
        latent_dim : int
            Latent representation dimension (default: 100)
        num_layers : int
            Number of LSTM layers (default: 2)
        dropout : float
            Dropout rate (default: 0.2)
        bidirectional : bool
            Whether to use bidirectional LSTM (default: True)
        """
        super(AudioEncoder, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Input projection layer
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output dimension after LSTM
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Latent projection layers
        self.latent_projection = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, latent_dim),
            nn.Tanh()  # Bounded output
        )
        
    def forward(self, x):
        """
        Forward pass through encoder.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_dim)
        
        Returns:
        --------
        torch.Tensor
            Latent representation of shape (batch_size, sequence_length, latent_dim)
        """
        batch_size, seq_len, _ = x.shape
        
        # Project input to hidden dimension
        x = self.input_projection(x)  # (batch_size, seq_len, hidden_dim)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out: (batch_size, seq_len, hidden_dim * num_directions)
        
        # Project to latent space
        latent = self.latent_projection(lstm_out)
        # latent: (batch_size, seq_len, latent_dim)
        
        return latent


class AudioDecoder(nn.Module):
    """
    Decoder network that reconstructs 441-dimensional audio features
    from 100-dimensional latent representation.
    """
    
    def __init__(self, latent_dim=100, hidden_dim=256, output_dim=441, 
                 num_layers=2, dropout=0.2, bidirectional=True):
        """
        Initialize the decoder.
        
        Parameters:
        -----------
        latent_dim : int
            Latent representation dimension (default: 100)
        hidden_dim : int
            Hidden layer dimension (default: 256)
        output_dim : int
            Output feature dimension (default: 441)
        num_layers : int
            Number of LSTM layers (default: 2)
        dropout : float
            Dropout rate (default: 0.2)
        bidirectional : bool
            Whether to use bidirectional LSTM (default: True)
        """
        super(AudioDecoder, self).__init__()
        
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        # Latent projection layer
        self.latent_projection = nn.Linear(latent_dim, hidden_dim)
        
        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True
        )
        
        # Output dimension after LSTM
        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        
        # Output projection layers
        self.output_projection = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, latent):
        """
        Forward pass through decoder.
        
        Parameters:
        -----------
        latent : torch.Tensor
            Latent tensor of shape (batch_size, sequence_length, latent_dim)
        
        Returns:
        --------
        torch.Tensor
            Reconstructed features of shape (batch_size, sequence_length, output_dim)
        """
        batch_size, seq_len, _ = latent.shape
        
        # Project latent to hidden dimension
        x = self.latent_projection(latent)  # (batch_size, seq_len, hidden_dim)
        
        # LSTM processing
        lstm_out, (hidden, cell) = self.lstm(x)
        # lstm_out: (batch_size, seq_len, hidden_dim * num_directions)
        
        # Project to output space
        output = self.output_projection(lstm_out)
        # output: (batch_size, seq_len, output_dim)
        
        return output


class SpeechTranslationModel(nn.Module):
    """
    Complete encoder-decoder model for speech translation.
    """
    
    def __init__(self, input_dim=441, latent_dim=100, output_dim=441,
                 encoder_hidden_dim=256, decoder_hidden_dim=256,
                 num_layers=2, dropout=0.2, bidirectional=True):
        """
        Initialize the complete model.
        
        Parameters:
        -----------
        input_dim : int
            Input feature dimension
        latent_dim : int
            Latent representation dimension
        output_dim : int
            Output feature dimension
        encoder_hidden_dim : int
            Encoder hidden dimension
        decoder_hidden_dim : int
            Decoder hidden dimension
        num_layers : int
            Number of LSTM layers
        dropout : float
            Dropout rate
        bidirectional : bool
            Whether to use bidirectional LSTM
        """
        super(SpeechTranslationModel, self).__init__()
        
        self.encoder = AudioEncoder(
            input_dim=input_dim,
            hidden_dim=encoder_hidden_dim,
            latent_dim=latent_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
        self.decoder = AudioDecoder(
            latent_dim=latent_dim,
            hidden_dim=decoder_hidden_dim,
            output_dim=output_dim,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=bidirectional
        )
        
    def forward(self, x):
        """
        Forward pass through the complete model.
        
        Parameters:
        -----------
        x : torch.Tensor
            Input tensor of shape (batch_size, sequence_length, input_dim)
        
        Returns:
        --------
        tuple
            (reconstructed_output, latent_representation)
        """
        # Encode input to latent space
        latent = self.encoder(x)
        
        # Decode latent to output space
        output = self.decoder(latent)
        
        return output, latent
    
    def encode(self, x):
        """
        Encode input to latent representation only.
        """
        return self.encoder(x)
    
    def decode(self, latent):
        """
        Decode latent representation to output only.
        """
        return self.decoder(latent)


class SpeechTranslationLoss(nn.Module):
    """
    Custom loss function for speech translation model.
    """
    
    def __init__(self, reconstruction_weight=1.0, latent_reg_weight=0.01):
        """
        Initialize loss function.
        
        Parameters:
        -----------
        reconstruction_weight : float
            Weight for reconstruction loss
        latent_reg_weight : float
            Weight for latent regularization
        """
        super(SpeechTranslationLoss, self).__init__()
        self.reconstruction_weight = reconstruction_weight
        self.latent_reg_weight = latent_reg_weight
        self.mse_loss = nn.MSELoss()
        
    def forward(self, output, target, latent=None):
        """
        Calculate total loss.
        
        Parameters:
        -----------
        output : torch.Tensor
            Model output
        target : torch.Tensor
            Target values
        latent : torch.Tensor, optional
            Latent representation for regularization
        
        Returns:
        --------
        dict
            Dictionary containing total loss and component losses
        """
        # Reconstruction loss (MSE)
        reconstruction_loss = self.mse_loss(output, target)
        
        total_loss = self.reconstruction_weight * reconstruction_loss
        
        loss_dict = {
            'total_loss': total_loss,
            'reconstruction_loss': reconstruction_loss
        }
        
        # Latent regularization (optional)
        if latent is not None and self.latent_reg_weight > 0:
            latent_reg = torch.mean(torch.norm(latent, dim=-1))
            total_loss += self.latent_reg_weight * latent_reg
            loss_dict['latent_regularization'] = latent_reg
            loss_dict['total_loss'] = total_loss
        
        return loss_dict


def create_model(config=None):
    """
    Factory function to create model with default or custom configuration.
    
    Parameters:
    -----------
    config : dict, optional
        Model configuration dictionary
    
    Returns:
    --------
    tuple
        (model, loss_function)
    """
    if config is None:
        config = {
            'input_dim': 441,
            'latent_dim': 100,
            'output_dim': 441,
            'encoder_hidden_dim': 256,
            'decoder_hidden_dim': 256,
            'num_layers': 2,
            'dropout': 0.2,
            'bidirectional': True
        }
    
    model = SpeechTranslationModel(**config)
    loss_fn = SpeechTranslationLoss()
    
    return model, loss_fn


def count_parameters(model):
    """
    Count the number of trainable parameters in the model.
    
    Parameters:
    -----------
    model : nn.Module
        PyTorch model
    
    Returns:
    --------
    int
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test the model
    print("Testing Speech Translation Model...")
    
    # Create model
    model, loss_fn = create_model()
    print(f"Model created with {count_parameters(model):,} trainable parameters")
    
    # Test with dummy data
    batch_size = 4
    sequence_length = 100
    input_dim = 441
    
    # Create dummy input
    dummy_input = torch.randn(batch_size, sequence_length, input_dim)
    print(f"Input shape: {dummy_input.shape}")
    
    # Forward pass
    with torch.no_grad():
        output, latent = model(dummy_input)
        print(f"Output shape: {output.shape}")
        print(f"Latent shape: {latent.shape}")
        
        # Test loss
        loss_dict = loss_fn(output, dummy_input, latent)
        print(f"Loss: {loss_dict['total_loss'].item():.4f}")
    
    print("Model test completed successfully!")
    
    # Print model architecture
    print("\nModel Architecture:")
    print(model)