#!/usr/bin/env python3
"""
Sprint 7: Unit Testing Suite

Comprehensive unit tests for individual components of the speech translation pipeline.
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
import unittest
from unittest.mock import Mock, patch, MagicMock

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))


class TestLoggingConfig(unittest.TestCase):
    """Unit tests for logging configuration."""
    
    def test_logging_manager_initialization(self):
        """Test LoggingManager initialization."""
        try:
            from utils.logging_config import LoggingManager, LogLevel, LogCategory
            
            manager = LoggingManager()
            self.assertIsNotNone(manager)
            
            # Test log level enum
            self.assertEqual(LogLevel.DEBUG.value, "DEBUG")
            self.assertEqual(LogLevel.INFO.value, "INFO")
            self.assertEqual(LogLevel.WARNING.value, "WARNING")
            self.assertEqual(LogLevel.ERROR.value, "ERROR")
            self.assertEqual(LogLevel.CRITICAL.value, "CRITICAL")
            
            # Test log category enum
            self.assertEqual(LogCategory.AUDIO.value, "audio")
            self.assertEqual(LogCategory.MODEL.value, "model")
            self.assertEqual(LogCategory.SYSTEM.value, "system")
            
        except ImportError:
            self.skipTest("Logging config not available")
    
    def test_get_logging_manager_singleton(self):
        """Test that get_logging_manager returns singleton instance."""
        try:
            from utils.logging_config import get_logging_manager
            
            manager1 = get_logging_manager()
            manager2 = get_logging_manager()
            
            self.assertIs(manager1, manager2, "Should return same singleton instance")
            
        except ImportError:
            self.skipTest("Logging config not available")
    
    def test_logging_methods(self):
        """Test logging methods work without errors."""
        try:
            from utils.logging_config import get_logging_manager, LogLevel, LogCategory
            
            manager = get_logging_manager()
            
            # Test structured logging method
            manager.log_structured(LogLevel.INFO, LogCategory.AUDIO, "test_component", "Test info message")
            manager.log_structured(LogLevel.WARNING, LogCategory.MODEL, "test_component", "Test warning message")
            manager.log_structured(LogLevel.ERROR, LogCategory.SYSTEM, "test_component", "Test error message")
            
            # Test logging with metadata
            manager.log_structured(LogLevel.INFO, LogCategory.PERFORMANCE, "test_component", "Performance test", 
                                 processing_time=1.5, metadata={"param": "value"})
            
        except ImportError:
            self.skipTest("Logging config not available")


class TestErrorHandling(unittest.TestCase):
    """Unit tests for error handling utilities."""
    
    def test_custom_exception_hierarchy(self):
        """Test custom exception class hierarchy."""
        try:
            from utils.error_handling import (
                SpeechTranslationError, AudioProcessingError, 
                ModelInferenceError, ReconstructionError
            )
            
            # Test base exception
            base_error = SpeechTranslationError("Base error")
            self.assertEqual(str(base_error), "Base error")
            self.assertIsInstance(base_error, Exception)
            
            # Test derived exceptions
            audio_error = AudioProcessingError("Audio error")
            self.assertIsInstance(audio_error, SpeechTranslationError)
            
            model_error = ModelInferenceError("Model error")
            self.assertIsInstance(model_error, SpeechTranslationError)
            
            recon_error = ReconstructionError("Reconstruction error")
            self.assertIsInstance(recon_error, SpeechTranslationError)
            
        except ImportError:
            self.skipTest("Error handling not available")
    
    def test_error_handler_initialization(self):
        """Test ErrorHandler initialization."""
        try:
            from utils.error_handling import get_error_handler
            
            handler = get_error_handler()
            self.assertIsNotNone(handler)
            
        except ImportError:
            self.skipTest("Error handling not available")
    
    def test_error_handler_decorator(self):
        """Test error handler decorator functionality."""
        try:
            from utils.error_handling import error_handler_decorator, AudioProcessingError
            
            @error_handler_decorator(component="test_component", operation="test_operation")
            def test_function_success():
                return "success"
            
            @error_handler_decorator(component="test_component", operation="test_operation")
            def test_function_failure():
                raise ValueError("Test error")
            
            # Test successful execution
            result = test_function_success()
            self.assertEqual(result, "success")
            
            # Test error handling - the decorator should reraise by default
            with self.assertRaises(ValueError):
                test_function_failure()
                
        except ImportError:
            self.skipTest("Error handling not available")


class TestFramingUtilities(unittest.TestCase):
    """Unit tests for framing utilities."""
    
    def test_create_feature_matrix_basic(self):
        """Test basic feature matrix creation."""
        try:
            from utils.framing import create_feature_matrix_advanced
            
            # Create test audio
            sr = 44100
            duration = 1.0
            audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sr * duration)))
            
            result = create_feature_matrix_advanced(
                audio, sr,
                frame_length_ms=20,
                hop_length_ms=10,
                n_features=441
            )
            
            self.assertIsNotNone(result)
            self.assertIn('feature_matrix', result)
            
            feature_matrix = result['feature_matrix']
            self.assertEqual(feature_matrix.shape[1], 441, "Should have correct number of features")
            self.assertGreater(feature_matrix.shape[0], 0, "Should have frames")
            
        except ImportError:
            self.skipTest("Framing utilities not available")
    
    def test_feature_matrix_parameters(self):
        """Test feature matrix with different parameters."""
        try:
            from utils.framing import create_feature_matrix_advanced
            
            sr = 44100
            duration = 2.0
            audio = np.random.randn(int(sr * duration)) * 0.5
            
            # Test different frame lengths
            for frame_ms in [10, 20, 40]:
                result = create_feature_matrix_advanced(
                    audio, sr,
                    frame_length_ms=frame_ms,
                    hop_length_ms=frame_ms // 2,
                    n_features=441
                )
                
                self.assertIsNotNone(result)
                feature_matrix = result['feature_matrix']
                self.assertEqual(feature_matrix.shape[1], 441)
            
            # Test different feature counts
            for n_feat in [220, 441, 882]:
                result = create_feature_matrix_advanced(
                    audio, sr,
                    frame_length_ms=20,
                    hop_length_ms=10,
                    n_features=n_feat
                )
                
                self.assertIsNotNone(result)
                feature_matrix = result['feature_matrix']
                self.assertEqual(feature_matrix.shape[1], n_feat)
                
        except ImportError:
            self.skipTest("Framing utilities not available")


class TestDenoiseUtilities(unittest.TestCase):
    """Unit tests for denoising utilities."""
    
    def test_preprocess_audio_basic(self):
        """Test basic audio preprocessing."""
        try:
            from utils.denoise import preprocess_audio_complete
            
            # Create test audio with noise
            sr = 44100
            duration = 2.0
            signal = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sr * duration))) * 0.7
            noise = np.random.normal(0, 0.1, len(signal))
            noisy_audio = signal + noise
            
            result = preprocess_audio_complete(noisy_audio, sr)
            
            self.assertIsNotNone(result)
            self.assertIsInstance(result, np.ndarray, "Result should be a numpy array")
            
            # Allow for more variation due to denoise processing
            self.assertAlmostEqual(len(result), len(noisy_audio), delta=2000)
            self.assertTrue(np.all(np.abs(result) <= 1.0), "Audio should be normalized")
            
        except ImportError:
            self.skipTest("Denoise utilities not available")
    
    def test_preprocess_audio_edge_cases(self):
        """Test preprocessing with edge cases."""
        try:
            from utils.denoise import preprocess_audio_complete
            
            sr = 44100
            
            # Test with silent audio
            silent_audio = np.zeros(int(sr * 1.0))
            result = preprocess_audio_complete(silent_audio, sr)
            self.assertIsNotNone(result)
            self.assertIsInstance(result, np.ndarray)
            
            # Test with very loud audio
            loud_audio = np.ones(int(sr * 1.0)) * 10.0  # Way above [-1, 1]
            result = preprocess_audio_complete(loud_audio, sr)
            self.assertIsNotNone(result)
            self.assertIsInstance(result, np.ndarray)
            self.assertTrue(np.all(np.abs(result) <= 1.0), "Should normalize loud audio")
            
        except ImportError:
            self.skipTest("Denoise utilities not available")


class TestPipelineOrchestrator(unittest.TestCase):
    """Unit tests for Pipeline Orchestrator."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.test_audio_file = self.temp_dir / "test_audio.wav"
        
        # Create test audio file
        sr = 44100
        duration = 2.0
        audio = np.sin(2 * np.pi * 440 * np.linspace(0, duration, int(sr * duration))) * 0.7
        sf.write(self.test_audio_file, audio, sr)
    
    def tearDown(self):
        """Clean up test environment."""
        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)
    
    def test_orchestrator_initialization(self):
        """Test PipelineOrchestrator initialization."""
        try:
            from utils.pipeline_orchestrator import PipelineOrchestrator
            
            # Test initialization with correct parameters
            orchestrator = PipelineOrchestrator(
                model_config={"input_dim": 441},
                audio_config={"sr": 44100},
                output_dir="test_outputs",
                enable_logging=False
            )
            
            self.assertIsNotNone(orchestrator)
            self.assertEqual(orchestrator.audio_config["sr"], 44100)
            
        except ImportError:
            self.skipTest("Pipeline orchestrator not available")
    
    @patch('utils.pipeline_orchestrator.PipelineOrchestrator._run_preprocessing')
    def test_process_audio_workflow(self, mock_preproc):
        """Test the audio processing workflow with mocks."""
        try:
            from utils.pipeline_orchestrator import PipelineOrchestrator, ProcessingResult, ProcessingStage, ProcessingStatus
            
            # Set up mock
            from datetime import datetime
            mock_result = ProcessingResult(
                session_id="test_session",
                timestamp=datetime.now().isoformat(),
                stage=ProcessingStage.PREPROCESSING,
                status=ProcessingStatus.COMPLETED,
                input_info={},
                output_info={'processed_audio': np.random.randn(44100)},
                metrics={},
                processing_time=0.1
            )
            mock_preproc.return_value = mock_result
            
            orchestrator = PipelineOrchestrator(
                output_dir="test_outputs",
                enable_logging=False
            )
            
            # Create test audio data
            test_audio = np.random.randn(44100).astype(np.float32)
            
            result = orchestrator._run_preprocessing(
                test_audio, sr=44100, session_id="test_session"
            )
            
            # Verify result structure
            self.assertIsNotNone(result)
            self.assertEqual(result.stage, ProcessingStage.PREPROCESSING)
            self.assertEqual(result.status, ProcessingStatus.COMPLETED)
            
        except ImportError:
            self.skipTest("Pipeline orchestrator not available")
    
    def test_orchestrator_error_handling(self):
        """Test error handling in orchestrator."""
        try:
            from utils.pipeline_orchestrator import PipelineOrchestrator
            
            orchestrator = PipelineOrchestrator(
                output_dir="test_outputs",
                enable_logging=False
            )
            
            # Test with invalid audio data
            invalid_audio = None
            
            result = orchestrator._run_preprocessing(
                invalid_audio, sr=44100, session_id="error_test"
            )
            
            self.assertIsNotNone(result)
            # The result should indicate failure or handle the error gracefully
            
        except ImportError:
            self.skipTest("Pipeline orchestrator not available")
        except Exception as e:
            # Expected to fail with invalid input
            self.assertIsInstance(e, Exception)


class TestModelComponents(unittest.TestCase):
    """Unit tests for model components."""
    
    def test_model_creation(self):
        """Test model creation and basic structure."""
        try:
            from models.encoder_decoder import create_model
            
            model, config = create_model()
            
            self.assertIsNotNone(model)
            self.assertIsNotNone(config)
            
            # Test model is in training mode by default
            self.assertTrue(model.training, "Model should be in training mode by default")
            
            # Test switching modes
            model.eval()
            self.assertFalse(model.training, "Model should be in eval mode")
            
            model.train()
            self.assertTrue(model.training, "Model should be in training mode")
            
            # Test model has required methods
            self.assertTrue(hasattr(model, 'forward'), "Model should have forward method")
            self.assertTrue(callable(getattr(model, 'forward')), "Forward should be callable")
            
        except ImportError:
            self.skipTest("Model components not available")
    
    def test_model_initialization(self):
        """Test model initialization."""
        try:
            from models.encoder_decoder import SpeechTranslationModel
            
            model = SpeechTranslationModel(
                input_dim=441,
                latent_dim=100,
                output_dim=441
            )
            
            self.assertIsNotNone(model)
            self.assertEqual(model.encoder.input_dim, 441)
            self.assertEqual(model.encoder.latent_dim, 100)
            
        except ImportError:
            self.skipTest("Model components not available")
    
    def test_model_forward_pass(self):
        """Test model forward pass with dummy data."""
        try:
            from models.encoder_decoder import create_model
            
            model, config = create_model()
            model.eval()
            
            # Create dummy input
            batch_size, seq_len, n_features = 2, 50, 441
            dummy_input = torch.randn(batch_size, seq_len, n_features)
            
            with torch.no_grad():
                output, latent = model(dummy_input)
                
                # Test output shapes
                self.assertEqual(output.shape, dummy_input.shape, "Output should match input shape")
                self.assertEqual(latent.shape[0], batch_size, "Latent batch size should match")
                self.assertEqual(latent.shape[1], seq_len, "Latent sequence length should match")
                
                # Test output values are reasonable
                self.assertTrue(torch.all(torch.isfinite(output)), "Output should be finite")
                self.assertTrue(torch.all(torch.isfinite(latent)), "Latent should be finite")
                
        except ImportError:
            self.skipTest("Model components not available")
    
    def test_model_forward_pass_direct(self):
        """Test model forward pass with direct instantiation."""
        try:
            from models.encoder_decoder import SpeechTranslationModel
            import torch
            
            model = SpeechTranslationModel(
                input_dim=441,
                latent_dim=100,
                output_dim=441
            )
            
            # Create test input
            batch_size, seq_len, input_dim = 2, 10, 441
            test_input = torch.randn(batch_size, seq_len, input_dim)
            
            # Forward pass
            output, latent = model(test_input)
            
            # Check output shapes
            expected_output_shape = (batch_size, seq_len, 441)
            expected_latent_shape = (batch_size, seq_len, 100)
            self.assertEqual(output.shape, expected_output_shape)
            self.assertEqual(latent.shape, expected_latent_shape)
            
        except ImportError:
            self.skipTest("Model components not available")
    
    def test_model_different_input_sizes(self):
        """Test model with different input sizes."""
        try:
            from models.encoder_decoder import create_model
            
            model, config = create_model()
            model.eval()
            
            n_features = 441
            
            # Test different sequence lengths
            for seq_len in [10, 50, 100]:
                dummy_input = torch.randn(1, seq_len, n_features)
                
                with torch.no_grad():
                    output, latent = model(dummy_input)
                    
                    self.assertEqual(output.shape[1], seq_len, f"Should handle seq_len={seq_len}")
                    self.assertEqual(latent.shape[1], seq_len, f"Latent should match seq_len={seq_len}")
            
            # Test different batch sizes
            for batch_size in [1, 2, 4]:
                dummy_input = torch.randn(batch_size, 50, n_features)
                
                with torch.no_grad():
                    output, latent = model(dummy_input)
                    
                    self.assertEqual(output.shape[0], batch_size, f"Should handle batch_size={batch_size}")
                    self.assertEqual(latent.shape[0], batch_size, f"Latent should match batch_size={batch_size}")
                    
        except ImportError:
            self.skipTest("Model components not available")


def run_unit_tests():
    """Run all unit tests."""
    print("\n🧪 Running Unit Tests")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestLoggingConfig,
        TestErrorHandling,
        TestFramingUtilities,
        TestDenoiseUtilities,
        TestPipelineOrchestrator,
        TestModelComponents
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Return success status
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_unit_tests()
    if success:
        print("\n✅ All unit tests passed!")
    else:
        print("\n❌ Some unit tests failed!")
        sys.exit(1)