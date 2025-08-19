"""Audio Framing and Feature Extraction Utilities

This module contains functions for audio framing and feature matrix generation
used in the multilingual speech translation project.
"""

import numpy as np
import librosa
from scipy import signal
import matplotlib.pyplot as plt


def frame_audio(y, sr, frame_length_ms=20, hop_length_ms=10):
    """
    Frame audio signal into overlapping windows.
    
    Parameters:
    -----------
    y : np.ndarray
        Input audio signal
    sr : int
        Sample rate
    frame_length_ms : float
        Frame length in milliseconds (default: 20ms)
    hop_length_ms : float
        Hop length in milliseconds (default: 10ms)
    
    Returns:
    --------
    np.ndarray
        Framed audio with shape (n_frames, frame_length)
    """
    # Convert ms to samples
    frame_length = int(frame_length_ms * sr / 1000)
    hop_length = int(hop_length_ms * sr / 1000)
    
    # Calculate number of frames
    n_frames = 1 + (len(y) - frame_length) // hop_length
    
    # Create frame matrix
    frames = np.zeros((n_frames, frame_length))
    
    for i in range(n_frames):
        start_idx = i * hop_length
        end_idx = start_idx + frame_length
        frames[i] = y[start_idx:end_idx]
    
    return frames


def apply_window(frames, window_type='hann'):
    """
    Apply windowing function to frames.
    
    Parameters:
    -----------
    frames : np.ndarray
        Input frames with shape (n_frames, frame_length)
    window_type : str
        Window type ('hann', 'hamming', 'blackman', 'rectangular')
    
    Returns:
    --------
    np.ndarray
        Windowed frames
    """
    frame_length = frames.shape[1]
    
    if window_type == 'hann':
        window = np.hanning(frame_length)
    elif window_type == 'hamming':
        window = np.hamming(frame_length)
    elif window_type == 'blackman':
        window = np.blackman(frame_length)
    elif window_type == 'rectangular':
        window = np.ones(frame_length)
    else:
        raise ValueError(f"Unknown window type: {window_type}")
    
    # Apply window to each frame
    windowed_frames = frames * window[np.newaxis, :]
    
    return windowed_frames


def extract_features_per_frame(frames, sr, n_features=441):
    """
    Extract features from each frame to create feature matrix.
    
    Parameters:
    -----------
    frames : np.ndarray
        Input frames with shape (n_frames, frame_length)
    sr : int
        Sample rate
    n_features : int
        Target number of features per frame (default: 441)
    
    Returns:
    --------
    np.ndarray
        Feature matrix with shape (n_frames, n_features)
    """
    n_frames, frame_length = frames.shape
    feature_matrix = np.zeros((n_frames, n_features))
    
    for i, frame in enumerate(frames):
        # Extract multiple types of features
        features = []
        
        # 1. Raw audio samples (first part)
        if frame_length >= n_features:
            # Downsample if frame is longer than target features
            raw_features = signal.resample(frame, n_features)
        else:
            # Pad if frame is shorter
            raw_features = np.pad(frame, (0, n_features - frame_length), 'constant')
        
        features.extend(raw_features)
        
        # Truncate or pad to exact n_features
        if len(features) > n_features:
            features = features[:n_features]
        elif len(features) < n_features:
            features.extend([0.0] * (n_features - len(features)))
        
        feature_matrix[i] = features
    
    return feature_matrix


def extract_spectral_features(frames, sr, n_fft=512):
    """
    Extract spectral features from frames.
    
    Parameters:
    -----------
    frames : np.ndarray
        Input frames with shape (n_frames, frame_length)
    sr : int
        Sample rate
    n_fft : int
        FFT size
    
    Returns:
    --------
    dict
        Dictionary containing various spectral features
    """
    n_frames = frames.shape[0]
    
    # Initialize feature arrays
    spectral_centroids = np.zeros(n_frames)
    spectral_rolloffs = np.zeros(n_frames)
    spectral_bandwidths = np.zeros(n_frames)
    zero_crossing_rates = np.zeros(n_frames)
    
    for i, frame in enumerate(frames):
        # Spectral centroid
        spectral_centroids[i] = librosa.feature.spectral_centroid(y=frame, sr=sr)[0, 0]
        
        # Spectral rolloff
        spectral_rolloffs[i] = librosa.feature.spectral_rolloff(y=frame, sr=sr)[0, 0]
        
        # Spectral bandwidth
        spectral_bandwidths[i] = librosa.feature.spectral_bandwidth(y=frame, sr=sr)[0, 0]
        
        # Zero crossing rate
        zero_crossing_rates[i] = librosa.feature.zero_crossing_rate(frame)[0, 0]
    
    return {
        'spectral_centroid': spectral_centroids,
        'spectral_rolloff': spectral_rolloffs,
        'spectral_bandwidth': spectral_bandwidths,
        'zero_crossing_rate': zero_crossing_rates
    }


def extract_mfcc_features(frames, sr, n_mfcc=13):
    """
    Extract MFCC features from frames.
    
    Parameters:
    -----------
    frames : np.ndarray
        Input frames
    sr : int
        Sample rate
    n_mfcc : int
        Number of MFCC coefficients
    
    Returns:
    --------
    np.ndarray
        MFCC features with shape (n_frames, n_mfcc)
    """
    n_frames = frames.shape[0]
    mfcc_features = np.zeros((n_frames, n_mfcc))
    
    for i, frame in enumerate(frames):
        # Extract MFCC
        mfcc = librosa.feature.mfcc(y=frame, sr=sr, n_mfcc=n_mfcc)
        mfcc_features[i] = mfcc[:, 0]  # Take first frame of MFCC
    
    return mfcc_features


def create_feature_matrix_advanced(y, sr, frame_length_ms=20, hop_length_ms=10, 
                                 n_features=441, include_spectral=True, 
                                 include_mfcc=True, window_type='hann'):
    """
    Create comprehensive feature matrix from audio signal.
    
    Parameters:
    -----------
    y : np.ndarray
        Input audio signal
    sr : int
        Sample rate
    frame_length_ms : float
        Frame length in milliseconds
    hop_length_ms : float
        Hop length in milliseconds
    n_features : int
        Target number of features per frame
    include_spectral : bool
        Whether to include spectral features
    include_mfcc : bool
        Whether to include MFCC features
    window_type : str
        Window type for framing
    
    Returns:
    --------
    dict
        Dictionary containing feature matrix and metadata
    """
    # Frame the audio
    frames = frame_audio(y, sr, frame_length_ms, hop_length_ms)
    
    # Apply windowing
    windowed_frames = apply_window(frames, window_type)
    
    # Extract basic features (raw audio)
    feature_matrix = extract_features_per_frame(windowed_frames, sr, n_features)
    
    result = {
        'feature_matrix': feature_matrix,
        'frames': frames,
        'windowed_frames': windowed_frames,
        'n_frames': frames.shape[0],
        'frame_length': frames.shape[1],
        'sr': sr
    }
    
    # Add spectral features if requested
    if include_spectral:
        spectral_features = extract_spectral_features(windowed_frames, sr)
        result['spectral_features'] = spectral_features
    
    # Add MFCC features if requested
    if include_mfcc:
        mfcc_features = extract_mfcc_features(windowed_frames, sr)
        result['mfcc_features'] = mfcc_features
    
    return result


def reconstruct_audio_overlap_add(feature_matrix, sr, frame_length_ms=20, 
                                hop_length_ms=10, window_type='hann'):
    """
    Reconstruct audio from feature matrix using overlap-add method.
    
    Parameters:
    -----------
    feature_matrix : np.ndarray
        Feature matrix with shape (n_frames, n_features)
    sr : int
        Sample rate
    frame_length_ms : float
        Frame length in milliseconds
    hop_length_ms : float
        Hop length in milliseconds
    window_type : str
        Window type used during analysis
    
    Returns:
    --------
    np.ndarray
        Reconstructed audio signal
    """
    n_frames, n_features = feature_matrix.shape
    
    # Convert ms to samples
    frame_length = int(frame_length_ms * sr / 1000)
    hop_length = int(hop_length_ms * sr / 1000)
    
    # Calculate output length
    output_length = (n_frames - 1) * hop_length + frame_length
    reconstructed = np.zeros(output_length)
    
    # Create window for overlap-add
    if window_type == 'hann':
        window = np.hanning(frame_length)
    elif window_type == 'hamming':
        window = np.hamming(frame_length)
    elif window_type == 'blackman':
        window = np.blackman(frame_length)
    else:
        window = np.ones(frame_length)
    
    # Overlap-add reconstruction
    for i in range(n_frames):
        start_idx = i * hop_length
        end_idx = start_idx + frame_length
        
        # Get frame from feature matrix (assuming first n_features are raw audio)
        frame = feature_matrix[i, :frame_length] if n_features >= frame_length else feature_matrix[i, :]
        
        # Pad or truncate frame to correct length
        if len(frame) < frame_length:
            frame = np.pad(frame, (0, frame_length - len(frame)), 'constant')
        elif len(frame) > frame_length:
            frame = frame[:frame_length]
        
        # Apply window and add to output
        windowed_frame = frame * window
        reconstructed[start_idx:end_idx] += windowed_frame
    
    return reconstructed


def visualize_feature_matrix(feature_matrix, sr, frame_length_ms=20, 
                            hop_length_ms=10, title='Feature Matrix'):
    """
    Visualize feature matrix as a spectrogram-like plot.
    
    Parameters:
    -----------
    feature_matrix : np.ndarray
        Feature matrix to visualize
    sr : int
        Sample rate
    frame_length_ms : float
        Frame length in milliseconds
    hop_length_ms : float
        Hop length in milliseconds
    title : str
        Plot title
    """
    plt.figure(figsize=(12, 6))
    
    # Create time axis
    n_frames = feature_matrix.shape[0]
    time_frames = np.arange(n_frames) * hop_length_ms / 1000  # Convert to seconds
    
    # Plot feature matrix
    plt.imshow(feature_matrix.T, aspect='auto', origin='lower', 
               extent=[0, time_frames[-1], 0, feature_matrix.shape[1]])
    plt.colorbar(label='Feature Value')
    plt.xlabel('Time (s)')
    plt.ylabel('Feature Index')
    plt.title(title)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    # Example usage
    print("Audio framing utilities loaded successfully!")
    print("Available functions:")
    print("- frame_audio()")
    print("- apply_window()")
    print("- extract_features_per_frame()")
    print("- extract_spectral_features()")
    print("- extract_mfcc_features()")
    print("- create_feature_matrix_advanced()")
    print("- reconstruct_audio_overlap_add()")
    print("- visualize_feature_matrix()")
    
    # Test with dummy data
    print("\nTesting with dummy audio...")
    sr = 44100
    duration = 1.0  # 1 second
    t = np.linspace(0, duration, int(sr * duration))
    y_test = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    
    # Test framing
    frames = frame_audio(y_test, sr)
    print(f"Original audio length: {len(y_test)} samples")
    print(f"Number of frames: {frames.shape[0]}")
    print(f"Frame length: {frames.shape[1]} samples")
    
    # Test feature extraction
    feature_result = create_feature_matrix_advanced(y_test, sr)
    print(f"Feature matrix shape: {feature_result['feature_matrix'].shape}")
    print("Framing utilities test completed successfully!")