"""Audio Reconstruction Utilities for Speech Translation System

This module provides robust audio reconstruction capabilities using overlap-add
method for converting predicted feature matrices back to audio waveforms.

Author: IBM Internship Project
Date: Sprint 6 - Signal Reconstruction & Evaluation
"""

import numpy as np
import librosa
import scipy.signal
from typing import Dict, Tuple, Optional, Union
import warnings


def reconstruct_audio_overlap_add(feature_matrix: np.ndarray, 
                                 sr: int = 44100,
                                 frame_length_ms: float = 20.0,
                                 hop_length_ms: float = 10.0,
                                 window_type: str = 'hann',
                                 normalize: bool = True,
                                 feature_type: str = 'raw') -> np.ndarray:
    """
    Enhanced audio reconstruction from feature matrix using overlap-add method.
    
    Parameters:
    -----------
    feature_matrix : np.ndarray
        Feature matrix with shape (n_frames, n_features)
    sr : int
        Sample rate (default: 44100)
    frame_length_ms : float
        Frame length in milliseconds (default: 20.0)
    hop_length_ms : float
        Hop length in milliseconds (default: 10.0)
    window_type : str
        Window type: 'hann', 'hamming', 'blackman', 'rectangular' (default: 'hann')
    normalize : bool
        Whether to normalize output audio (default: True)
    feature_type : str
        Type of features: 'raw', 'spectral', 'mfcc' (default: 'raw')
    
    Returns:
    --------
    np.ndarray
        Reconstructed audio signal
    """
    if feature_matrix.ndim != 2:
        raise ValueError(f"Feature matrix must be 2D, got shape {feature_matrix.shape}")
    
    n_frames, n_features = feature_matrix.shape
    
    # Convert ms to samples
    frame_length = int(frame_length_ms * sr / 1000)
    hop_length = int(hop_length_ms * sr / 1000)
    
    # Calculate output length
    output_length = (n_frames - 1) * hop_length + frame_length
    reconstructed = np.zeros(output_length, dtype=np.float32)
    
    # Create analysis window
    window = _create_window(frame_length, window_type)
    
    # Handle different feature types
    if feature_type == 'raw':
        frames = _extract_raw_frames(feature_matrix, frame_length)
    elif feature_type == 'spectral':
        frames = _reconstruct_from_spectral(feature_matrix, frame_length)
    elif feature_type == 'mfcc':
        frames = _reconstruct_from_mfcc(feature_matrix, frame_length, sr)
    else:
        warnings.warn(f"Unknown feature type '{feature_type}', using raw reconstruction")
        frames = _extract_raw_frames(feature_matrix, frame_length)
    
    # Overlap-add reconstruction with proper windowing
    window_sum = np.zeros(output_length)
    
    for i in range(n_frames):
        start_idx = i * hop_length
        end_idx = start_idx + frame_length
        
        if end_idx > output_length:
            # Handle edge case
            valid_length = output_length - start_idx
            windowed_frame = frames[i][:valid_length] * window[:valid_length]
            reconstructed[start_idx:output_length] += windowed_frame
            window_sum[start_idx:output_length] += window[:valid_length]
        else:
            windowed_frame = frames[i] * window
            reconstructed[start_idx:end_idx] += windowed_frame
            window_sum[start_idx:end_idx] += window
    
    # Normalize by window sum to avoid amplitude scaling issues
    nonzero_mask = window_sum > 1e-8
    reconstructed[nonzero_mask] /= window_sum[nonzero_mask]
    
    # Optional normalization
    if normalize:
        reconstructed = _normalize_audio(reconstructed)
    
    return reconstructed


def reconstruct_with_quality_metrics(feature_matrix: np.ndarray,
                                    original_audio: Optional[np.ndarray] = None,
                                    sr: int = 44100,
                                    **kwargs) -> Dict[str, Union[np.ndarray, float]]:
    """
    Reconstruct audio and compute quality metrics if original is provided.
    
    Parameters:
    -----------
    feature_matrix : np.ndarray
        Feature matrix to reconstruct
    original_audio : np.ndarray, optional
        Original audio for comparison
    sr : int
        Sample rate
    **kwargs
        Additional arguments for reconstruction
    
    Returns:
    --------
    Dict containing:
        - 'reconstructed': reconstructed audio
        - 'mse': mean squared error (if original provided)
        - 'snr': signal-to-noise ratio (if original provided)
        - 'correlation': correlation coefficient (if original provided)
    """
    reconstructed = reconstruct_audio_overlap_add(feature_matrix, sr=sr, **kwargs)
    
    result = {'reconstructed': reconstructed}
    
    if original_audio is not None:
        # Align lengths
        min_len = min(len(reconstructed), len(original_audio))
        recon_aligned = reconstructed[:min_len]
        orig_aligned = original_audio[:min_len]
        
        # Compute metrics
        result['mse'] = np.mean((recon_aligned - orig_aligned) ** 2)
        result['snr'] = _compute_snr(orig_aligned, recon_aligned)
        result['correlation'] = np.corrcoef(orig_aligned, recon_aligned)[0, 1]
    
    return result


def batch_reconstruct(feature_matrices: list,
                     sr: int = 44100,
                     **kwargs) -> list:
    """
    Reconstruct multiple feature matrices in batch.
    
    Parameters:
    -----------
    feature_matrices : list
        List of feature matrices
    sr : int
        Sample rate
    **kwargs
        Additional arguments for reconstruction
    
    Returns:
    --------
    list
        List of reconstructed audio arrays
    """
    reconstructed_list = []
    
    for i, feature_matrix in enumerate(feature_matrices):
        try:
            reconstructed = reconstruct_audio_overlap_add(feature_matrix, sr=sr, **kwargs)
            reconstructed_list.append(reconstructed)
        except Exception as e:
            warnings.warn(f"Failed to reconstruct sample {i}: {str(e)}")
            reconstructed_list.append(np.array([]))
    
    return reconstructed_list


# Helper functions
def _create_window(frame_length: int, window_type: str) -> np.ndarray:
    """Create analysis window."""
    if window_type == 'hann':
        return np.hanning(frame_length)
    elif window_type == 'hamming':
        return np.hamming(frame_length)
    elif window_type == 'blackman':
        return np.blackman(frame_length)
    elif window_type == 'rectangular':
        return np.ones(frame_length)
    else:
        warnings.warn(f"Unknown window type '{window_type}', using Hann window")
        return np.hanning(frame_length)


def _extract_raw_frames(feature_matrix: np.ndarray, frame_length: int) -> np.ndarray:
    """Extract raw audio frames from feature matrix."""
    n_frames, n_features = feature_matrix.shape
    frames = np.zeros((n_frames, frame_length))
    
    for i in range(n_frames):
        if n_features >= frame_length:
            frames[i] = feature_matrix[i, :frame_length]
        else:
            frames[i, :n_features] = feature_matrix[i]
    
    return frames


def _reconstruct_from_spectral(feature_matrix: np.ndarray, frame_length: int) -> np.ndarray:
    """Reconstruct from spectral features (placeholder for future implementation)."""
    warnings.warn("Spectral reconstruction not fully implemented, using raw method")
    return _extract_raw_frames(feature_matrix, frame_length)


def _reconstruct_from_mfcc(feature_matrix: np.ndarray, frame_length: int, sr: int) -> np.ndarray:
    """Reconstruct from MFCC features (placeholder for future implementation)."""
    warnings.warn("MFCC reconstruction not fully implemented, using raw method")
    return _extract_raw_frames(feature_matrix, frame_length)


def _normalize_audio(audio: np.ndarray, target_db: float = -20.0) -> np.ndarray:
    """Normalize audio to target dB level."""
    if len(audio) == 0:
        return audio
    
    # RMS normalization
    rms = np.sqrt(np.mean(audio ** 2))
    if rms > 0:
        target_rms = 10 ** (target_db / 20)
        audio = audio * (target_rms / rms)
    
    # Clip to prevent overflow
    audio = np.clip(audio, -1.0, 1.0)
    
    return audio


def _compute_snr(original: np.ndarray, reconstructed: np.ndarray) -> float:
    """Compute Signal-to-Noise Ratio."""
    if len(original) == 0 or len(reconstructed) == 0:
        return float('-inf')
    
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - reconstructed) ** 2)
    
    if noise_power == 0:
        return float('inf')
    
    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db


if __name__ == "__main__":
    # Test reconstruction utilities
    print("Audio Reconstruction Utilities loaded successfully!")
    print("Available functions:")
    print("- reconstruct_audio_overlap_add()")
    print("- reconstruct_with_quality_metrics()")
    print("- batch_reconstruct()")
    
    # Test with dummy data
    print("\nTesting with dummy feature matrix...")
    n_frames, n_features = 100, 441
    dummy_features = np.random.randn(n_frames, n_features) * 0.1
    
    reconstructed = reconstruct_audio_overlap_add(dummy_features)
    print(f"Input shape: {dummy_features.shape}")
    print(f"Reconstructed audio length: {len(reconstructed)} samples")
    print(f"Duration: {len(reconstructed) / 44100:.2f} seconds")
    print("Reconstruction test completed successfully!")