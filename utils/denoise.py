"""Audio Denoising Utilities

This module contains functions for audio denoising and preprocessing
used in the multilingual speech translation project.
"""

import numpy as np
import librosa
from scipy import signal
import soundfile as sf


def adaptive_denoise(y, sr, noise_factor=0.1, alpha=2.0, beta=0.01):
    """
    Apply adaptive denoising using spectral subtraction method.
    
    Parameters:
    -----------
    y : np.ndarray
        Input audio signal
    sr : int
        Sample rate
    noise_factor : float
        Factor for noise estimation (default: 0.1)
    alpha : float
        Over-subtraction factor (default: 2.0)
    beta : float
        Spectral floor factor (default: 0.01)
    
    Returns:
    --------
    np.ndarray
        Denoised audio signal
    """
    # Compute STFT
    stft = librosa.stft(y, hop_length=512, n_fft=2048)
    magnitude = np.abs(stft)
    phase = np.angle(stft)
    
    # Estimate noise from first few frames (assuming initial silence/noise)
    noise_frames = min(5, magnitude.shape[1] // 4)
    noise_spectrum = np.mean(magnitude[:, :noise_frames], axis=1, keepdims=True)
    
    # Spectral subtraction
    clean_magnitude = magnitude - alpha * noise_spectrum
    
    # Apply spectral floor to prevent over-subtraction
    clean_magnitude = np.maximum(clean_magnitude, beta * magnitude)
    
    # Reconstruct signal
    clean_stft = clean_magnitude * np.exp(1j * phase)
    clean_audio = librosa.istft(clean_stft, hop_length=512)
    
    return clean_audio


def wiener_filter(y, sr, noise_power_ratio=0.1):
    """
    Apply Wiener filtering for noise reduction.
    
    Parameters:
    -----------
    y : np.ndarray
        Input audio signal
    sr : int
        Sample rate
    noise_power_ratio : float
        Estimated noise to signal power ratio
    
    Returns:
    --------
    np.ndarray
        Filtered audio signal
    """
    # Compute power spectral density
    f, psd = signal.welch(y, sr, nperseg=1024)
    
    # Estimate noise power (assume it's a fraction of signal power)
    signal_power = np.mean(psd)
    noise_power = noise_power_ratio * signal_power
    
    # Wiener filter transfer function
    H = signal_power / (signal_power + noise_power)
    
    # Apply filter in frequency domain
    Y = np.fft.fft(y)
    Y_filtered = Y * H
    y_filtered = np.real(np.fft.ifft(Y_filtered))
    
    return y_filtered


def normalize_audio(y, target_db=-20, method='rms'):
    """
    Normalize audio to target dB level.
    
    Parameters:
    -----------
    y : np.ndarray
        Input audio signal
    target_db : float
        Target dB level (default: -20)
    method : str
        Normalization method ('rms' or 'peak')
    
    Returns:
    --------
    np.ndarray
        Normalized audio signal
    """
    if method == 'rms':
        # RMS normalization
        rms = np.sqrt(np.mean(y**2))
        if rms > 0:
            target_rms = 10**(target_db/20)
            normalized = y * (target_rms / rms)
        else:
            normalized = y
    elif method == 'peak':
        # Peak normalization
        peak = np.max(np.abs(y))
        if peak > 0:
            target_peak = 10**(target_db/20)
            normalized = y * (target_peak / peak)
        else:
            normalized = y
    else:
        raise ValueError("Method must be 'rms' or 'peak'")
    
    # Clip to prevent clipping
    normalized = np.clip(normalized, -1.0, 1.0)
    
    return normalized


def remove_silence(y, sr, top_db=20, frame_length=2048, hop_length=512):
    """
    Remove silence from audio signal.
    
    Parameters:
    -----------
    y : np.ndarray
        Input audio signal
    sr : int
        Sample rate
    top_db : float
        Threshold for silence detection (default: 20)
    frame_length : int
        Frame length for analysis
    hop_length : int
        Hop length for analysis
    
    Returns:
    --------
    np.ndarray
        Audio signal with silence removed
    """
    # Trim silence from beginning and end
    y_trimmed, _ = librosa.effects.trim(y, top_db=top_db, 
                                       frame_length=frame_length, 
                                       hop_length=hop_length)
    return y_trimmed


def preprocess_audio_complete(y, sr, denoise_method='spectral_subtraction', 
                            normalize_method='rms', target_db=-20, 
                            remove_silence_flag=True):
    """
    Complete audio preprocessing pipeline.
    
    Parameters:
    -----------
    y : np.ndarray
        Input audio signal
    sr : int
        Sample rate
    denoise_method : str
        Denoising method ('spectral_subtraction' or 'wiener')
    normalize_method : str
        Normalization method ('rms' or 'peak')
    target_db : float
        Target dB level for normalization
    remove_silence_flag : bool
        Whether to remove silence
    
    Returns:
    --------
    np.ndarray
        Processed audio signal
    """
    # Remove silence if requested
    if remove_silence_flag:
        y = remove_silence(y, sr)
    
    # Apply denoising
    if denoise_method == 'spectral_subtraction':
        y_denoised = adaptive_denoise(y, sr)
    elif denoise_method == 'wiener':
        y_denoised = wiener_filter(y, sr)
    else:
        y_denoised = y  # No denoising
    
    # Normalize audio
    y_normalized = normalize_audio(y_denoised, target_db, normalize_method)
    
    return y_normalized


def calculate_snr(signal, noise):
    """
    Calculate Signal-to-Noise Ratio.
    
    Parameters:
    -----------
    signal : np.ndarray
        Clean signal
    noise : np.ndarray
        Noise signal
    
    Returns:
    --------
    float
        SNR in dB
    """
    signal_power = np.mean(signal**2)
    noise_power = np.mean(noise**2)
    
    if noise_power > 0:
        snr = 10 * np.log10(signal_power / noise_power)
    else:
        snr = float('inf')
    
    return snr


def save_audio(y, sr, filepath, format='WAV', subtype='PCM_16'):
    """
    Save audio to file.
    
    Parameters:
    -----------
    y : np.ndarray
        Audio signal
    sr : int
        Sample rate
    filepath : str
        Output file path
    format : str
        Audio format (default: 'WAV')
    subtype : str
        Audio subtype (default: 'PCM_16')
    """
    sf.write(filepath, y, sr, format=format, subtype=subtype)


if __name__ == "__main__":
    # Example usage
    print("Audio denoising utilities loaded successfully!")
    print("Available functions:")
    print("- adaptive_denoise()")
    print("- wiener_filter()")
    print("- normalize_audio()")
    print("- remove_silence()")
    print("- preprocess_audio_complete()")
    print("- calculate_snr()")
    print("- save_audio()")