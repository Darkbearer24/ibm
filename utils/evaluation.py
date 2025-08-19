"""Evaluation Utilities for Speech Translation System

This module provides comprehensive evaluation metrics and analysis tools
for assessing the quality of audio reconstruction and translation.

Author: IBM Internship Project
Date: Sprint 6 - Signal Reconstruction & Evaluation
"""

import numpy as np
import librosa
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal
from scipy.stats import pearsonr
from typing import Dict, List, Tuple, Optional, Union
import warnings
from pathlib import Path
import json


class AudioEvaluator:
    """Comprehensive audio evaluation class."""
    
    def __init__(self, sr: int = 44100):
        self.sr = sr
        self.metrics_history = []
    
    def evaluate_reconstruction(self, 
                              original: np.ndarray, 
                              reconstructed: np.ndarray,
                              detailed: bool = True) -> Dict[str, float]:
        """
        Comprehensive evaluation of audio reconstruction quality.
        
        Parameters:
        -----------
        original : np.ndarray
            Original audio signal
        reconstructed : np.ndarray
            Reconstructed audio signal
        detailed : bool
            Whether to compute detailed metrics
        
        Returns:
        --------
        Dict[str, float]
            Dictionary of evaluation metrics
        """
        # Align lengths
        min_len = min(len(original), len(reconstructed))
        orig = original[:min_len]
        recon = reconstructed[:min_len]
        
        if len(orig) == 0:
            return {'error': 'Empty audio signals'}
        
        metrics = {}
        
        # Basic metrics
        metrics['mse'] = self._compute_mse(orig, recon)
        metrics['rmse'] = np.sqrt(metrics['mse'])
        metrics['mae'] = self._compute_mae(orig, recon)
        metrics['snr_db'] = self._compute_snr(orig, recon)
        metrics['correlation'] = self._compute_correlation(orig, recon)
        
        if detailed:
            # Advanced metrics
            metrics['spectral_distance'] = self._compute_spectral_distance(orig, recon)
            metrics['mfcc_distance'] = self._compute_mfcc_distance(orig, recon)
            metrics['zero_crossing_rate_diff'] = self._compute_zcr_difference(orig, recon)
            metrics['energy_ratio'] = self._compute_energy_ratio(orig, recon)
            metrics['dynamic_range_ratio'] = self._compute_dynamic_range_ratio(orig, recon)
            
            # Perceptual metrics
            metrics['spectral_centroid_diff'] = self._compute_spectral_centroid_diff(orig, recon)
            metrics['spectral_rolloff_diff'] = self._compute_spectral_rolloff_diff(orig, recon)
        
        # Store in history
        self.metrics_history.append(metrics.copy())
        
        return metrics
    
    def evaluate_batch(self, 
                      original_list: List[np.ndarray], 
                      reconstructed_list: List[np.ndarray]) -> Dict[str, Union[float, List[float]]]:
        """
        Evaluate a batch of audio reconstructions.
        
        Parameters:
        -----------
        original_list : List[np.ndarray]
            List of original audio signals
        reconstructed_list : List[np.ndarray]
            List of reconstructed audio signals
        
        Returns:
        --------
        Dict containing aggregate statistics
        """
        if len(original_list) != len(reconstructed_list):
            raise ValueError("Original and reconstructed lists must have same length")
        
        batch_metrics = []
        
        for orig, recon in zip(original_list, reconstructed_list):
            try:
                metrics = self.evaluate_reconstruction(orig, recon, detailed=False)
                if 'error' not in metrics:
                    batch_metrics.append(metrics)
            except Exception as e:
                warnings.warn(f"Failed to evaluate sample: {str(e)}")
        
        if not batch_metrics:
            return {'error': 'No valid evaluations'}
        
        # Aggregate statistics
        result = {}
        for key in batch_metrics[0].keys():
            values = [m[key] for m in batch_metrics if key in m and not np.isnan(m[key])]
            if values:
                result[f'{key}_mean'] = np.mean(values)
                result[f'{key}_std'] = np.std(values)
                result[f'{key}_min'] = np.min(values)
                result[f'{key}_max'] = np.max(values)
                result[f'{key}_values'] = values
        
        result['n_samples'] = len(batch_metrics)
        
        return result
    
    def plot_evaluation_summary(self, 
                               metrics: Dict[str, float], 
                               save_path: Optional[str] = None) -> plt.Figure:
        """
        Create visualization of evaluation metrics.
        
        Parameters:
        -----------
        metrics : Dict[str, float]
            Evaluation metrics
        save_path : str, optional
            Path to save the plot
        
        Returns:
        --------
        plt.Figure
            The created figure
        """
        # Filter numeric metrics
        numeric_metrics = {k: v for k, v in metrics.items() 
                          if isinstance(v, (int, float)) and not np.isnan(v)}
        
        if not numeric_metrics:
            raise ValueError("No numeric metrics to plot")
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle('Audio Reconstruction Evaluation Summary', fontsize=16)
        
        # Metrics bar plot
        ax1 = axes[0, 0]
        keys = list(numeric_metrics.keys())
        values = list(numeric_metrics.values())
        bars = ax1.bar(range(len(keys)), values)
        ax1.set_xticks(range(len(keys)))
        ax1.set_xticklabels(keys, rotation=45, ha='right')
        ax1.set_title('Evaluation Metrics')
        ax1.grid(True, alpha=0.3)
        
        # Color bars based on metric type
        for i, (key, bar) in enumerate(zip(keys, bars)):
            if 'snr' in key.lower() or 'correlation' in key.lower():
                bar.set_color('green' if values[i] > 0 else 'red')
            elif 'mse' in key.lower() or 'mae' in key.lower() or 'distance' in key.lower():
                bar.set_color('red' if values[i] > 0.1 else 'orange' if values[i] > 0.01 else 'green')
            else:
                bar.set_color('blue')
        
        # SNR vs Correlation scatter
        ax2 = axes[0, 1]
        if 'snr_db' in numeric_metrics and 'correlation' in numeric_metrics:
            ax2.scatter(numeric_metrics['correlation'], numeric_metrics['snr_db'], 
                       s=100, alpha=0.7, c='blue')
            ax2.set_xlabel('Correlation')
            ax2.set_ylabel('SNR (dB)')
            ax2.set_title('SNR vs Correlation')
            ax2.grid(True, alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'SNR/Correlation\ndata not available', 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title('SNR vs Correlation')
        
        # Error metrics comparison
        ax3 = axes[1, 0]
        error_metrics = {k: v for k, v in numeric_metrics.items() 
                        if any(err in k.lower() for err in ['mse', 'mae', 'rmse'])}
        if error_metrics:
            ax3.bar(error_metrics.keys(), error_metrics.values(), color='red', alpha=0.7)
            ax3.set_title('Error Metrics')
            ax3.set_ylabel('Error Value')
            ax3.tick_params(axis='x', rotation=45)
            ax3.grid(True, alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Error metrics\nnot available', 
                    ha='center', va='center', transform=ax3.transAxes)
            ax3.set_title('Error Metrics')
        
        # Quality score (composite)
        ax4 = axes[1, 1]
        quality_score = self._compute_quality_score(numeric_metrics)
        colors = ['red', 'orange', 'yellow', 'lightgreen', 'green']
        color_idx = min(int(quality_score * 5), 4)
        
        ax4.pie([quality_score, 1 - quality_score], 
               labels=['Quality', 'Room for Improvement'],
               colors=[colors[color_idx], 'lightgray'],
               startangle=90,
               counterclock=False)
        ax4.set_title(f'Overall Quality Score: {quality_score:.2f}')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def plot_waveform_comparison(self, 
                                original: np.ndarray, 
                                reconstructed: np.ndarray,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot waveform comparison between original and reconstructed audio.
        
        Parameters:
        -----------
        original : np.ndarray
            Original audio signal
        reconstructed : np.ndarray
            Reconstructed audio signal
        save_path : str, optional
            Path to save the plot
        
        Returns:
        --------
        plt.Figure
            The created figure
        """
        # Align lengths
        min_len = min(len(original), len(reconstructed))
        orig = original[:min_len]
        recon = reconstructed[:min_len]
        
        time = np.arange(len(orig)) / self.sr
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Original waveform
        axes[0].plot(time, orig, color='blue', alpha=0.7, linewidth=0.5)
        axes[0].set_title('Original Audio')
        axes[0].set_ylabel('Amplitude')
        axes[0].grid(True, alpha=0.3)
        
        # Reconstructed waveform
        axes[1].plot(time, recon, color='red', alpha=0.7, linewidth=0.5)
        axes[1].set_title('Reconstructed Audio')
        axes[1].set_ylabel('Amplitude')
        axes[1].grid(True, alpha=0.3)
        
        # Difference
        diff = orig - recon
        axes[2].plot(time, diff, color='green', alpha=0.7, linewidth=0.5)
        axes[2].set_title('Difference (Original - Reconstructed)')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Amplitude')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        return fig
    
    def save_evaluation_report(self, 
                              metrics: Dict[str, float], 
                              save_path: str,
                              additional_info: Optional[Dict] = None):
        """
        Save evaluation report to JSON file.
        
        Parameters:
        -----------
        metrics : Dict[str, float]
            Evaluation metrics
        save_path : str
            Path to save the report
        additional_info : Dict, optional
            Additional information to include
        """
        report = {
            'evaluation_metrics': metrics,
            'timestamp': str(np.datetime64('now')),
            'sample_rate': self.sr
        }
        
        if additional_info:
            report['additional_info'] = additional_info
        
        # Convert numpy types to Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        report = convert_numpy(report)
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
    
    # Private helper methods
    def _compute_mse(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Compute Mean Squared Error."""
        return float(np.mean((original - reconstructed) ** 2))
    
    def _compute_mae(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Compute Mean Absolute Error."""
        return float(np.mean(np.abs(original - reconstructed)))
    
    def _compute_snr(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Compute Signal-to-Noise Ratio in dB."""
        signal_power = np.mean(original ** 2)
        noise_power = np.mean((original - reconstructed) ** 2)
        
        if noise_power == 0:
            return float('inf')
        
        return float(10 * np.log10(signal_power / noise_power))
    
    def _compute_correlation(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Compute Pearson correlation coefficient."""
        try:
            corr, _ = pearsonr(original, reconstructed)
            return float(corr) if not np.isnan(corr) else 0.0
        except:
            return 0.0
    
    def _compute_spectral_distance(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Compute spectral distance between signals."""
        try:
            # Compute spectrograms
            _, _, spec_orig = signal.spectrogram(original, fs=self.sr)
            _, _, spec_recon = signal.spectrogram(reconstructed, fs=self.sr)
            
            # Align shapes
            min_shape = (min(spec_orig.shape[0], spec_recon.shape[0]),
                        min(spec_orig.shape[1], spec_recon.shape[1]))
            
            spec_orig = spec_orig[:min_shape[0], :min_shape[1]]
            spec_recon = spec_recon[:min_shape[0], :min_shape[1]]
            
            # Compute distance
            return float(np.mean((spec_orig - spec_recon) ** 2))
        except:
            return float('nan')
    
    def _compute_mfcc_distance(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Compute MFCC distance between signals."""
        try:
            mfcc_orig = librosa.feature.mfcc(y=original, sr=self.sr, n_mfcc=13)
            mfcc_recon = librosa.feature.mfcc(y=reconstructed, sr=self.sr, n_mfcc=13)
            
            # Align shapes
            min_frames = min(mfcc_orig.shape[1], mfcc_recon.shape[1])
            mfcc_orig = mfcc_orig[:, :min_frames]
            mfcc_recon = mfcc_recon[:, :min_frames]
            
            return float(np.mean((mfcc_orig - mfcc_recon) ** 2))
        except:
            return float('nan')
    
    def _compute_zcr_difference(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Compute difference in zero crossing rates."""
        try:
            zcr_orig = librosa.feature.zero_crossing_rate(original)[0]
            zcr_recon = librosa.feature.zero_crossing_rate(reconstructed)[0]
            
            min_len = min(len(zcr_orig), len(zcr_recon))
            return float(np.mean(np.abs(zcr_orig[:min_len] - zcr_recon[:min_len])))
        except:
            return float('nan')
    
    def _compute_energy_ratio(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Compute ratio of energy between signals."""
        energy_orig = np.sum(original ** 2)
        energy_recon = np.sum(reconstructed ** 2)
        
        if energy_orig == 0:
            return float('inf') if energy_recon > 0 else 1.0
        
        return float(energy_recon / energy_orig)
    
    def _compute_dynamic_range_ratio(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Compute ratio of dynamic ranges."""
        dr_orig = np.max(original) - np.min(original)
        dr_recon = np.max(reconstructed) - np.min(reconstructed)
        
        if dr_orig == 0:
            return float('inf') if dr_recon > 0 else 1.0
        
        return float(dr_recon / dr_orig)
    
    def _compute_spectral_centroid_diff(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Compute difference in spectral centroids."""
        try:
            centroid_orig = librosa.feature.spectral_centroid(y=original, sr=self.sr)[0]
            centroid_recon = librosa.feature.spectral_centroid(y=reconstructed, sr=self.sr)[0]
            
            min_len = min(len(centroid_orig), len(centroid_recon))
            return float(np.mean(np.abs(centroid_orig[:min_len] - centroid_recon[:min_len])))
        except:
            return float('nan')
    
    def _compute_spectral_rolloff_diff(self, original: np.ndarray, reconstructed: np.ndarray) -> float:
        """Compute difference in spectral rolloff."""
        try:
            rolloff_orig = librosa.feature.spectral_rolloff(y=original, sr=self.sr)[0]
            rolloff_recon = librosa.feature.spectral_rolloff(y=reconstructed, sr=self.sr)[0]
            
            min_len = min(len(rolloff_orig), len(rolloff_recon))
            return float(np.mean(np.abs(rolloff_orig[:min_len] - rolloff_recon[:min_len])))
        except:
            return float('nan')
    
    def _compute_quality_score(self, metrics: Dict[str, float]) -> float:
        """Compute composite quality score (0-1, higher is better)."""
        score = 0.0
        weight_sum = 0.0
        
        # SNR contribution (higher is better)
        if 'snr_db' in metrics and not np.isnan(metrics['snr_db']):
            snr_score = min(max(metrics['snr_db'] / 30.0, 0), 1)  # Normalize to 0-1
            score += snr_score * 0.3
            weight_sum += 0.3
        
        # Correlation contribution (higher is better)
        if 'correlation' in metrics and not np.isnan(metrics['correlation']):
            corr_score = max(metrics['correlation'], 0)  # Ensure positive
            score += corr_score * 0.3
            weight_sum += 0.3
        
        # MSE contribution (lower is better)
        if 'mse' in metrics and not np.isnan(metrics['mse']):
            mse_score = max(1 - min(metrics['mse'] * 10, 1), 0)  # Invert and normalize
            score += mse_score * 0.2
            weight_sum += 0.2
        
        # Energy ratio contribution (closer to 1 is better)
        if 'energy_ratio' in metrics and not np.isnan(metrics['energy_ratio']):
            energy_score = 1 - min(abs(metrics['energy_ratio'] - 1), 1)
            score += energy_score * 0.2
            weight_sum += 0.2
        
        return score / weight_sum if weight_sum > 0 else 0.0


if __name__ == "__main__":
    # Test evaluation utilities
    print("Audio Evaluation Utilities loaded successfully!")
    print("Available classes and functions:")
    print("- AudioEvaluator class")
    print("  - evaluate_reconstruction()")
    print("  - evaluate_batch()")
    print("  - plot_evaluation_summary()")
    print("  - plot_waveform_comparison()")
    print("  - save_evaluation_report()")
    
    # Test with dummy data
    print("\nTesting with dummy audio...")
    evaluator = AudioEvaluator()
    
    # Create test signals
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    original = np.sin(2 * np.pi * 440 * t)  # 440 Hz sine wave
    reconstructed = original + np.random.normal(0, 0.1, len(original))  # Add noise
    
    # Evaluate
    metrics = evaluator.evaluate_reconstruction(original, reconstructed)
    print(f"Test metrics: {list(metrics.keys())}")
    print(f"SNR: {metrics.get('snr_db', 'N/A'):.2f} dB")
    print(f"Correlation: {metrics.get('correlation', 'N/A'):.3f}")
    print("Evaluation test completed successfully!")