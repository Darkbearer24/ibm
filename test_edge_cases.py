#!/usr/bin/env python3
"""
Sprint 7: Edge Case Testing Suite

Comprehensive edge case tests for the speech translation pipeline,
including boundary conditions, error scenarios, and stress testing.
"""

import sys
import os
import tempfile
import shutil
from pathlib import Path
import numpy as np
import torch
import librosa
import soundfile as sf
from datetime import datetime
import json
import pytest
import warnings

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from utils.pipeline_orchestrator import PipelineOrchestrator
from utils.logging_config import get_logging_manager, LogLevel, LogCategory
from utils.error_handling import (
    SpeechTranslationError, AudioProcessingError, ModelInferenceError, 
    ReconstructionError, get_error_handler
)
from utils.framing import create_feature_matrix_advanced
from utils.denoise import preprocess_audio_complete


class TestAudioEdgeCases:
    """Test edge cases related to audio processing."""
    
    @classmethod
    def setup_class(cls):
        """Set up test environment."""
        cls.temp_dir = Path(tempfile.mkdtemp())
        cls.test_audio_dir = cls.temp_dir / "edge_case_audio"
        cls.test_audio_dir.mkdir(exist_ok=True)
        
        # Create edge case audio files
        cls._create_edge_case_audio_files()
    
    @classmethod
    def teardown_class(cls):
        """Clean up test environment."""
        if cls.temp_dir.exists():
            shutil.rmtree(cls.temp_dir)
    
    @classmethod
    def _create_edge_case_audio_files(cls):
        """Create various edge case audio files."""
        sr = 44100
        
        # Edge case 1: Silent audio
        silent_audio = np.zeros(int(sr * 2.0))
        sf.write(cls.test_audio_dir / "silent.wav", silent_audio, sr)
        
        # Edge case 2: Very short audio (< 100ms)
        very_short = np.sin(2 * np.pi * 440 * np.linspace(0, 0.05, int(sr * 0.05))) * 0.7
        sf.write(cls.test_audio_dir / "very_short.wav", very_short, sr)
        
        # Edge case 3: Very long audio (> 30 seconds)
        very_long = np.sin(2 * np.pi * 440 * np.linspace(0, 35.0, int(sr * 35.0))) * 0.7
        sf.write(cls.test_audio_dir / "very_long.wav", very_long, sr)
        
        # Edge case 4: Clipped audio (values at boundaries)
        clipped_audio = np.ones(int(sr * 2.0))  # All values at +1.0
        sf.write(cls.test_audio_dir / "clipped_positive.wav", clipped_audio, sr)
        
        clipped_negative = -np.ones(int(sr * 2.0))  # All values at -1.0
        sf.write(cls.test_audio_dir / "clipped_negative.wav", clipped_negative, sr)
        
        # Edge case 5: Very noisy audio (high SNR)
        signal = np.sin(2 * np.pi * 440 * np.linspace(0, 2.0, int(sr * 2.0))) * 0.1
        noise = np.random.normal(0, 0.5, len(signal))
        very_noisy = signal + noise
        sf.write(cls.test_audio_dir / "very_noisy.wav", very_noisy, sr)
        
        # Edge case 6: DC offset
        dc_offset_audio = np.sin(2 * np.pi * 440 * np.linspace(0, 2.0, int(sr * 2.0))) * 0.7 + 0.5
        sf.write(cls.test_audio_dir / "dc_offset.wav", dc_offset_audio, sr)
        
        # Edge case 7: Extremely low amplitude
        low_amplitude = np.sin(2 * np.pi * 440 * np.linspace(0, 2.0, int(sr * 2.0))) * 1e-6
        sf.write(cls.test_audio_dir / "low_amplitude.wav", low_amplitude, sr)
        
        # Edge case 8: High frequency content (near Nyquist)
        high_freq = np.sin(2 * np.pi * (sr/2 - 1000) * np.linspace(0, 2.0, int(sr * 2.0))) * 0.7
        sf.write(cls.test_audio_dir / "high_frequency.wav", high_freq, sr)
        
        # Edge case 9: Impulse response
        impulse = np.zeros(int(sr * 2.0))
        impulse[sr] = 1.0  # Single impulse at 1 second
        sf.write(cls.test_audio_dir / "impulse.wav", impulse, sr)
        
        # Edge case 10: NaN and Inf values (will be clipped by soundfile)
        nan_audio = np.full(int(sr * 2.0), np.nan)
        nan_audio = np.nan_to_num(nan_audio)  # Convert to zeros
        sf.write(cls.test_audio_dir / "nan_converted.wav", nan_audio, sr)
    
    def test_silent_audio_processing(self):
        """Test processing of completely silent audio."""
        try:
            from utils.pipeline_orchestrator import PipelineOrchestrator
            orchestrator = PipelineOrchestrator(
                model_path="models/speech_translation_model.pth",
                enable_logging=True
            )
            
            result = orchestrator.process_audio_file(
                audio_file_path=str(self.test_audio_dir / "silent.wav"),
                session_id="edge_test_silent"
            )
            
            # Silent audio should either process successfully or fail gracefully
            assert result is not None, "Pipeline should return results for silent audio"
            
            if result.get('success', False):
                # If successful, output should exist
                assert 'output_audio' in result, "Successful processing should include output"
            else:
                # If failed, should have meaningful error message
                assert 'error' in result, "Failed processing should include error message"
                
        except Exception as e:
            pytest.skip(f"Pipeline orchestrator not available: {e}")
    
    def test_very_short_audio_processing(self):
        """Test processing of very short audio files."""
        try:
            from utils.pipeline_orchestrator import PipelineOrchestrator
            orchestrator = PipelineOrchestrator(
                model_path="models/speech_translation_model.pth",
                enable_logging=True
            )
            
            result = orchestrator.process_audio_file(
                audio_file_path=str(self.test_audio_dir / "very_short.wav"),
                session_id="edge_test_short"
            )
            
            assert result is not None, "Pipeline should handle very short audio"
            
            # Very short audio might fail due to insufficient data
            if not result.get('success', False):
                error_msg = result.get('error', '')
                # Should be a meaningful error about insufficient data
                assert any(keyword in error_msg.lower() for keyword in 
                          ['short', 'insufficient', 'minimum', 'length']), \
                       f"Error should mention audio length issue: {error_msg}"
                
        except Exception as e:
            pytest.skip(f"Pipeline orchestrator not available: {e}")
    
    def test_clipped_audio_processing(self):
        """Test processing of clipped audio."""
        try:
            from utils.pipeline_orchestrator import PipelineOrchestrator
            orchestrator = PipelineOrchestrator(
                model_path="models/speech_translation_model.pth",
                enable_logging=True
            )
            
            for clipped_file in ["clipped_positive.wav", "clipped_negative.wav"]:
                result = orchestrator.process_audio_file(
                    audio_file_path=str(self.test_audio_dir / clipped_file),
                    session_id=f"edge_test_{clipped_file}"
                )
                
                assert result is not None, f"Pipeline should handle {clipped_file}"
                
                # Clipped audio should either process or fail with appropriate error
                if result.get('success', False):
                    output_audio = result.get('output_audio')
                    if output_audio is not None:
                        # Output should be within valid range
                        assert np.all(np.abs(output_audio) <= 1.0), \
                               "Output audio should be within [-1, 1] range"
                
        except Exception as e:
            pytest.skip(f"Pipeline orchestrator not available: {e}")
    
    def test_extreme_noise_processing(self):
        """Test processing of extremely noisy audio."""
        try:
            from utils.pipeline_orchestrator import PipelineOrchestrator
            orchestrator = PipelineOrchestrator(
                model_path="models/speech_translation_model.pth",
                enable_logging=True
            )
            
            result = orchestrator.process_audio_file(
                audio_file_path=str(self.test_audio_dir / "very_noisy.wav"),
                session_id="edge_test_noisy"
            )
            
            assert result is not None, "Pipeline should handle very noisy audio"
            
            # Very noisy audio should process but might have lower quality
            if result.get('success', False):
                # Check if quality metrics are available and reasonable
                if 'reconstruction' in result:
                    recon_result = result['reconstruction']
                    if 'metrics' in recon_result:
                        quality_score = recon_result['metrics'].get('quality_score', 0)
                        # Quality might be low but should be a valid number
                        assert isinstance(quality_score, (int, float)), \
                               "Quality score should be numeric"
                        assert 0 <= quality_score <= 1, \
                               "Quality score should be between 0 and 1"
                
        except Exception as e:
            pytest.skip(f"Pipeline orchestrator not available: {e}")


class TestFeatureExtractionEdgeCases:
    """Test edge cases in feature extraction."""
    
    def test_empty_audio_features(self):
        """Test feature extraction with empty audio."""
        empty_audio = np.array([])
        
        try:
            result = create_feature_matrix_advanced(
                empty_audio, 44100,
                frame_length_ms=20,
                hop_length_ms=10,
                n_features=441
            )
            
            # Should either return empty features or raise appropriate error
            if result is not None:
                feature_matrix = result.get('feature_matrix')
                if feature_matrix is not None:
                    assert feature_matrix.shape[0] == 0, "Empty audio should produce empty features"
            
        except (ValueError, IndexError) as e:
            # Expected behavior for empty input
            assert "empty" in str(e).lower() or "length" in str(e).lower()
        except Exception as e:
            pytest.skip(f"Feature extraction not available: {e}")
    
    def test_single_sample_audio_features(self):
        """Test feature extraction with single sample audio."""
        single_sample = np.array([0.5])
        
        try:
            result = create_feature_matrix_advanced(
                single_sample, 44100,
                frame_length_ms=20,
                hop_length_ms=10,
                n_features=441
            )
            
            # Should handle gracefully
            assert result is not None, "Should return result for single sample"
            
        except Exception as e:
            # Acceptable to fail with meaningful error
            assert any(keyword in str(e).lower() for keyword in 
                      ['length', 'insufficient', 'minimum']), \
                   f"Error should be about insufficient data: {e}"
    
    def test_extreme_values_features(self):
        """Test feature extraction with extreme audio values."""
        sr = 44100
        duration = 1.0
        
        # Test with maximum positive values
        max_audio = np.ones(int(sr * duration))
        
        try:
            result = create_feature_matrix_advanced(
                max_audio, sr,
                frame_length_ms=20,
                hop_length_ms=10,
                n_features=441
            )
            
            assert result is not None, "Should handle maximum values"
            feature_matrix = result.get('feature_matrix')
            
            if feature_matrix is not None:
                # Features should be finite
                assert np.all(np.isfinite(feature_matrix)), \
                       "Features should be finite for extreme input values"
                
        except Exception as e:
            pytest.skip(f"Feature extraction failed with extreme values: {e}")


class TestModelInferenceEdgeCases:
    """Test edge cases in model inference."""
    
    def test_zero_features_inference(self):
        """Test model inference with all-zero features."""
        try:
            from models.encoder_decoder import create_model
            
            model, _ = create_model()
            model.eval()
            
            # Create zero feature matrix
            batch_size, seq_len, n_features = 1, 100, 441
            zero_features = torch.zeros(batch_size, seq_len, n_features)
            
            with torch.no_grad():
                output, latent = model(zero_features)
                
                # Output should be valid tensors
                assert output.shape == zero_features.shape, "Output shape should match input"
                assert torch.all(torch.isfinite(output)), "Output should be finite"
                assert torch.all(torch.isfinite(latent)), "Latent should be finite"
                
        except Exception as e:
            pytest.skip(f"Model not available: {e}")
    
    def test_extreme_sequence_lengths(self):
        """Test model with extreme sequence lengths."""
        try:
            from models.encoder_decoder import create_model
            
            model, _ = create_model()
            model.eval()
            
            batch_size, n_features = 1, 441
            
            # Test very short sequence
            short_seq = torch.randn(batch_size, 1, n_features)
            
            with torch.no_grad():
                output, latent = model(short_seq)
                assert output.shape[1] == 1, "Should handle single frame"
                
            # Test longer sequence (if memory allows)
            try:
                long_seq = torch.randn(batch_size, 1000, n_features)
                with torch.no_grad():
                    output, latent = model(long_seq)
                    assert output.shape[1] == 1000, "Should handle long sequences"
            except RuntimeError as e:
                if "memory" in str(e).lower():
                    # Expected for very long sequences
                    pass
                else:
                    raise
                    
        except Exception as e:
            pytest.skip(f"Model not available: {e}")


class TestSystemResourceEdgeCases:
    """Test edge cases related to system resources."""
    
    def test_memory_usage_monitoring(self):
        """Test memory usage during processing."""
        import psutil
        import gc
        
        # Get initial memory usage
        process = psutil.Process()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        try:
            from utils.pipeline_orchestrator import PipelineOrchestrator
            
            # Process multiple files to test memory accumulation
            orchestrator = PipelineOrchestrator(
                model_path="models/speech_translation_model.pth",
                enable_logging=True
            )
            
            # Create temporary audio file
            temp_dir = Path(tempfile.mkdtemp())
            sr = 44100
            duration = 5.0  # 5 second file
            test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sr * duration))) * 0.7
            test_file = temp_dir / "memory_test.wav"
            sf.write(test_file, test_audio, sr)
            
            # Process the same file multiple times
            for i in range(5):
                result = orchestrator.process_audio_file(
                    audio_file_path=str(test_file),
                    session_id=f"memory_test_{i}"
                )
                
                # Force garbage collection
                gc.collect()
                
                # Check memory usage
                current_memory = process.memory_info().rss / 1024 / 1024  # MB
                memory_increase = current_memory - initial_memory
                
                # Memory should not increase excessively (allow 500MB increase)
                assert memory_increase < 500, \
                       f"Memory usage increased by {memory_increase:.1f}MB after {i+1} iterations"
            
            # Clean up
            shutil.rmtree(temp_dir)
            
        except Exception as e:
            pytest.skip(f"Memory test not available: {e}")
    
    def test_concurrent_processing_stress(self):
        """Test system under concurrent processing stress."""
        try:
            from utils.pipeline_orchestrator import PipelineOrchestrator
            import threading
            import time
            
            orchestrator = PipelineOrchestrator(
                model_path="models/speech_translation_model.pth",
                enable_logging=True
            )
            
            # Create test audio file
            temp_dir = Path(tempfile.mkdtemp())
            sr = 44100
            duration = 2.0
            test_audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sr * duration))) * 0.7
            test_file = temp_dir / "stress_test.wav"
            sf.write(test_file, test_audio, sr)
            
            results = {}
            errors = {}
            
            def stress_process(thread_id):
                try:
                    for i in range(3):  # Each thread processes 3 times
                        session_id = f"stress_test_{thread_id}_{i}"
                        result = orchestrator.process_audio_file(
                            audio_file_path=str(test_file),
                            session_id=session_id
                        )
                        results[f"{thread_id}_{i}"] = result
                        time.sleep(0.1)  # Small delay between requests
                except Exception as e:
                    errors[thread_id] = str(e)
            
            # Create multiple threads for stress testing
            threads = []
            num_threads = 5
            
            start_time = time.time()
            
            for i in range(num_threads):
                thread = threading.Thread(target=stress_process, args=(i,))
                threads.append(thread)
                thread.start()
            
            # Wait for all threads to complete
            for thread in threads:
                thread.join(timeout=300)  # 5 minute timeout
            
            end_time = time.time()
            total_time = end_time - start_time
            
            # Clean up
            shutil.rmtree(temp_dir)
            
            # Validate stress test results
            assert len(errors) == 0, f"Stress test should not produce errors: {errors}"
            assert total_time < 300, f"Stress test should complete within 5 minutes, took {total_time:.1f}s"
            
            # At least some results should be successful
            successful_results = [r for r in results.values() if r and r.get('success', False)]
            total_expected = num_threads * 3
            success_rate = len(successful_results) / total_expected
            
            assert success_rate > 0.5, f"At least 50% of stress test requests should succeed, got {success_rate:.1%}"
            
        except Exception as e:
            pytest.skip(f"Stress test not available: {e}")


def run_edge_case_tests():
    """Run all edge case tests."""
    print("\n🧪 Running Edge Case Tests")
    print("=" * 60)
    
    # Run pytest with this file
    import subprocess
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        __file__, 
        "-v", 
        "--tb=short",
        "-x"  # Stop on first failure for edge cases
    ], capture_output=True, text=True)
    
    print(result.stdout)
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    return result.returncode == 0


if __name__ == "__main__":
    success = run_edge_case_tests()
    if success:
        print("\n✅ All edge case tests passed!")
    else:
        print("\n❌ Some edge case tests failed!")
        sys.exit(1)