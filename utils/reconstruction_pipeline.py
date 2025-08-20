"""Reconstruction Pipeline for Speech Translation System

This module provides a comprehensive pipeline for processing model outputs
into reconstructed audio with quality evaluation and batch processing capabilities.

Author: IBM Internship Project
Date: Sprint 6 - Signal Reconstruction & Evaluation
"""

import numpy as np
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple
import json
import warnings
from datetime import datetime
import os

# Import our utilities
from .reconstruction import (
    reconstruct_audio_overlap_add,
    reconstruct_with_quality_metrics,
    batch_reconstruct
)
from .evaluation import AudioEvaluator


class ReconstructionPipeline:
    """Complete pipeline for audio reconstruction and evaluation."""
    
    def __init__(self, 
                 sr: int = 44100,
                 output_dir: str = "outputs/reconstructed",
                 save_intermediate: bool = True,
                 quality_threshold: float = 0.5):
        """
        Initialize the reconstruction pipeline.
        
        Parameters:
        -----------
        sr : int
            Sample rate for audio processing
        output_dir : str
            Directory to save reconstructed audio and reports
        save_intermediate : bool
            Whether to save intermediate processing results
        quality_threshold : float
            Minimum quality score for flagging low-quality reconstructions
        """
        self.sr = sr
        self.output_dir = Path(output_dir)
        self.save_intermediate = save_intermediate
        self.quality_threshold = quality_threshold
        
        # Create output directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "audio").mkdir(exist_ok=True)
        (self.output_dir / "reports").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        
        # Initialize evaluator
        self.evaluator = AudioEvaluator(sr=sr)
        
        # Processing history
        self.processing_history = []
    
    def process_single(self,
                      feature_matrix: np.ndarray,
                      original_audio: Optional[np.ndarray] = None,
                      output_name: str = "reconstructed",
                      **reconstruction_kwargs) -> Dict:
        """
        Process a single feature matrix into reconstructed audio.
        
        Parameters:
        -----------
        feature_matrix : np.ndarray
            Feature matrix to reconstruct (n_frames, n_features)
        original_audio : np.ndarray, optional
            Original audio for quality comparison
        output_name : str
            Name for output files
        **reconstruction_kwargs
            Additional arguments for reconstruction
        
        Returns:
        --------
        Dict containing reconstruction results and metrics
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"{output_name}_{timestamp}"
        
        print(f"Processing: {output_name}")
        print(f"Feature matrix shape: {feature_matrix.shape}")
        
        try:
            # Reconstruct audio
            if original_audio is not None:
                result = reconstruct_with_quality_metrics(
                    feature_matrix, 
                    original_audio=original_audio,
                    sr=self.sr,
                    **reconstruction_kwargs
                )
                reconstructed = result['reconstructed']
                basic_metrics = {k: v for k, v in result.items() if k != 'reconstructed'}
            else:
                reconstructed = reconstruct_audio_overlap_add(
                    feature_matrix,
                    sr=self.sr,
                    **reconstruction_kwargs
                )
                basic_metrics = {}
            
            # Detailed evaluation if original is available
            detailed_metrics = {}
            if original_audio is not None:
                detailed_metrics = self.evaluator.evaluate_reconstruction(
                    original_audio, reconstructed, detailed=True
                )
            
            # Combine metrics
            all_metrics = {**basic_metrics, **detailed_metrics}
            
            # Save reconstructed audio
            audio_path = self.output_dir / "audio" / f"{session_id}.wav"
            sf.write(audio_path, reconstructed, self.sr)
            
            # Generate quality score
            quality_score = self.evaluator._compute_quality_score(all_metrics) if all_metrics else 0.0
            
            # Create result dictionary
            result_dict = {
                'session_id': session_id,
                'timestamp': timestamp,
                'input_shape': feature_matrix.shape,
                'output_length': len(reconstructed),
                'output_duration': len(reconstructed) / self.sr,
                'audio_path': str(audio_path),
                'reconstructed_audio': reconstructed,  # Include the actual audio data
                'quality_score': quality_score,
                'metrics': all_metrics,
                'reconstruction_params': reconstruction_kwargs,
                'has_original': original_audio is not None
            }
            
            # Flag low quality
            if quality_score < self.quality_threshold:
                result_dict['quality_flag'] = 'LOW_QUALITY'
                warnings.warn(f"Low quality reconstruction detected: {quality_score:.3f}")
            
            # Save evaluation report
            if all_metrics:
                report_path = self.output_dir / "reports" / f"{session_id}_report.json"
                self.evaluator.save_evaluation_report(
                    all_metrics, 
                    str(report_path),
                    additional_info={
                        'session_id': session_id,
                        'input_shape': feature_matrix.shape,
                        'reconstruction_params': reconstruction_kwargs
                    }
                )
                result_dict['report_path'] = str(report_path)
            
            # Generate plots if original is available
            if original_audio is not None:
                # Evaluation summary plot
                try:
                    plot_path = self.output_dir / "plots" / f"{session_id}_evaluation.png"
                    self.evaluator.plot_evaluation_summary(all_metrics, str(plot_path))
                    result_dict['evaluation_plot'] = str(plot_path)
                except Exception as e:
                    warnings.warn(f"Failed to create evaluation plot: {e}")
                
                # Waveform comparison plot
                try:
                    waveform_path = self.output_dir / "plots" / f"{session_id}_waveform.png"
                    self.evaluator.plot_waveform_comparison(
                        original_audio, reconstructed, str(waveform_path)
                    )
                    result_dict['waveform_plot'] = str(waveform_path)
                except Exception as e:
                    warnings.warn(f"Failed to create waveform plot: {e}")
            
            # Add to processing history
            self.processing_history.append(result_dict)
            
            print(f"✓ Reconstruction completed: {session_id}")
            print(f"  Quality Score: {quality_score:.3f}")
            print(f"  Output Duration: {len(reconstructed) / self.sr:.2f}s")
            
            return result_dict
            
        except Exception as e:
            error_result = {
                'session_id': session_id,
                'timestamp': timestamp,
                'error': str(e),
                'input_shape': feature_matrix.shape,
                'status': 'FAILED'
            }
            self.processing_history.append(error_result)
            raise RuntimeError(f"Reconstruction failed for {output_name}: {e}")
    
    def process_batch(self,
                     feature_matrices: List[np.ndarray],
                     original_audios: Optional[List[np.ndarray]] = None,
                     output_names: Optional[List[str]] = None,
                     **reconstruction_kwargs) -> Dict:
        """
        Process multiple feature matrices in batch.
        
        Parameters:
        -----------
        feature_matrices : List[np.ndarray]
            List of feature matrices to reconstruct
        original_audios : List[np.ndarray], optional
            List of original audio signals for comparison
        output_names : List[str], optional
            Names for output files
        **reconstruction_kwargs
            Additional arguments for reconstruction
        
        Returns:
        --------
        Dict containing batch processing results
        """
        n_samples = len(feature_matrices)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        batch_id = f"batch_{timestamp}"
        
        print(f"\nProcessing batch: {batch_id}")
        print(f"Number of samples: {n_samples}")
        
        # Prepare inputs
        if original_audios is None:
            original_audios = [None] * n_samples
        elif len(original_audios) != n_samples:
            raise ValueError("Number of original audios must match feature matrices")
        
        if output_names is None:
            output_names = [f"sample_{i:03d}" for i in range(n_samples)]
        elif len(output_names) != n_samples:
            raise ValueError("Number of output names must match feature matrices")
        
        # Process each sample
        results = []
        successful_reconstructions = []
        failed_count = 0
        
        for i, (features, original, name) in enumerate(zip(feature_matrices, original_audios, output_names)):
            try:
                print(f"\nProcessing sample {i+1}/{n_samples}: {name}")
                result = self.process_single(
                    features, 
                    original_audio=original,
                    output_name=f"{batch_id}_{name}",
                    **reconstruction_kwargs
                )
                results.append(result)
                if original is not None:
                    successful_reconstructions.append((original, 
                                                     np.load(result['audio_path'].replace('.wav', '.npy')) 
                                                     if self.save_intermediate 
                                                     else librosa.load(result['audio_path'], sr=self.sr)[0]))
            except Exception as e:
                print(f"✗ Failed to process {name}: {e}")
                failed_count += 1
                results.append({
                    'session_id': f"{batch_id}_{name}",
                    'error': str(e),
                    'status': 'FAILED'
                })
        
        # Batch evaluation if we have successful reconstructions with originals
        batch_metrics = {}
        if successful_reconstructions:
            try:
                originals = [pair[0] for pair in successful_reconstructions]
                reconstructed = [pair[1] for pair in successful_reconstructions]
                batch_metrics = self.evaluator.evaluate_batch(originals, reconstructed)
            except Exception as e:
                warnings.warn(f"Batch evaluation failed: {e}")
        
        # Create batch summary
        batch_summary = {
            'batch_id': batch_id,
            'timestamp': timestamp,
            'total_samples': n_samples,
            'successful': n_samples - failed_count,
            'failed': failed_count,
            'success_rate': (n_samples - failed_count) / n_samples,
            'batch_metrics': batch_metrics,
            'individual_results': results,
            'reconstruction_params': reconstruction_kwargs
        }
        
        # Save batch report
        batch_report_path = self.output_dir / "reports" / f"{batch_id}_batch_report.json"
        with open(batch_report_path, 'w') as f:
            # Convert numpy types for JSON serialization
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
            
            json.dump(convert_numpy(batch_summary), f, indent=2)
        
        batch_summary['batch_report_path'] = str(batch_report_path)
        
        print(f"\n✓ Batch processing completed: {batch_id}")
        print(f"  Success Rate: {batch_summary['success_rate']:.1%}")
        print(f"  Successful: {batch_summary['successful']}/{n_samples}")
        
        return batch_summary
    
    def load_model_outputs(self, 
                          model_output_dir: str,
                          file_pattern: str = "*.npy") -> List[Tuple[str, np.ndarray]]:
        """
        Load model output files from directory.
        
        Parameters:
        -----------
        model_output_dir : str
            Directory containing model output files
        file_pattern : str
            File pattern to match
        
        Returns:
        --------
        List of (filename, feature_matrix) tuples
        """
        output_dir = Path(model_output_dir)
        if not output_dir.exists():
            raise FileNotFoundError(f"Model output directory not found: {model_output_dir}")
        
        files = list(output_dir.glob(file_pattern))
        if not files:
            raise FileNotFoundError(f"No files matching pattern '{file_pattern}' in {model_output_dir}")
        
        loaded_data = []
        for file_path in sorted(files):
            try:
                data = np.load(file_path)
                loaded_data.append((file_path.stem, data))
                print(f"Loaded: {file_path.name} - Shape: {data.shape}")
            except Exception as e:
                warnings.warn(f"Failed to load {file_path}: {e}")
        
        print(f"Successfully loaded {len(loaded_data)} model outputs")
        return loaded_data
    
    def get_processing_summary(self) -> Dict:
        """
        Get summary of all processing done in this session.
        
        Returns:
        --------
        Dict containing processing statistics
        """
        if not self.processing_history:
            return {'message': 'No processing history available'}
        
        successful = [r for r in self.processing_history if 'error' not in r]
        failed = [r for r in self.processing_history if 'error' in r]
        
        quality_scores = [r.get('quality_score', 0) for r in successful if 'quality_score' in r]
        
        summary = {
            'total_processed': len(self.processing_history),
            'successful': len(successful),
            'failed': len(failed),
            'success_rate': len(successful) / len(self.processing_history),
            'average_quality_score': np.mean(quality_scores) if quality_scores else 0,
            'output_directory': str(self.output_dir),
            'processing_history': self.processing_history
        }
        
        return summary


if __name__ == "__main__":
    # Test the reconstruction pipeline
    print("Reconstruction Pipeline loaded successfully!")
    print("Available classes:")
    print("- ReconstructionPipeline")
    print("  - process_single()")
    print("  - process_batch()")
    print("  - load_model_outputs()")
    print("  - get_processing_summary()")
    
    # Test with dummy data
    print("\nTesting with dummy data...")
    pipeline = ReconstructionPipeline(output_dir="test_outputs")
    
    # Create test feature matrix
    n_frames, n_features = 50, 441
    test_features = np.random.randn(n_frames, n_features) * 0.1
    
    # Create test original audio
    sr = 44100
    duration = 1.0
    t = np.linspace(0, duration, int(sr * duration))
    test_original = np.sin(2 * np.pi * 440 * t) * 0.5
    
    try:
        result = pipeline.process_single(
            test_features,
            original_audio=test_original,
            output_name="test_reconstruction"
        )
        print(f"✓ Test completed successfully!")
        print(f"  Quality Score: {result['quality_score']:.3f}")
        print(f"  Output saved to: {result['audio_path']}")
    except Exception as e:
        print(f"✗ Test failed: {e}")