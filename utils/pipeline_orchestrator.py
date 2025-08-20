"""Pipeline Orchestrator for Multilingual Speech Translation System

This module provides a comprehensive orchestrator class that manages the complete
end-to-end processing workflow from audio input to reconstructed output with
quality evaluation and error handling.

Author: IBM Internship Project
Date: Sprint 7 - System Integration
"""

import numpy as np
import torch
import librosa
import soundfile as sf
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import json
import warnings
from datetime import datetime
import logging
import traceback
from dataclasses import dataclass, asdict
from enum import Enum

# Import project modules
from models.encoder_decoder import SpeechTranslationModel
from utils.denoise import preprocess_audio_complete
from utils.framing import create_feature_matrix_advanced
from utils.reconstruction_pipeline import ReconstructionPipeline
from utils.evaluation import AudioEvaluator

# Import centralized logging and error handling
from utils.logging_config import get_logging_manager, LogLevel, LogCategory
from utils.error_handling import (
    get_error_handler, error_handler_decorator, AudioProcessingError,
    ModelInferenceError, ReconstructionError, ValidationError,
    ErrorContext, ErrorSeverity
)


class ProcessingStage(Enum):
    """Enumeration of processing stages."""
    PREPROCESSING = "preprocessing"
    FEATURE_EXTRACTION = "feature_extraction"
    MODEL_INFERENCE = "model_inference"
    RECONSTRUCTION = "reconstruction"
    EVALUATION = "evaluation"
    COMPLETE = "complete"


class ProcessingStatus(Enum):
    """Enumeration of processing status."""
    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class ProcessingResult:
    """Data class for storing processing results."""
    session_id: str
    timestamp: str
    stage: ProcessingStage
    status: ProcessingStatus
    input_info: Dict[str, Any]
    output_info: Dict[str, Any]
    metrics: Dict[str, float]
    processing_time: float
    error_message: Optional[str] = None
    warnings: List[str] = None

    def __post_init__(self):
        if self.warnings is None:
            self.warnings = []


class PipelineOrchestrator:
    """Comprehensive pipeline orchestrator for speech translation system."""
    
    def __init__(self,
                 model_config: Optional[Dict] = None,
                 audio_config: Optional[Dict] = None,
                 output_dir: str = "outputs/pipeline",
                 enable_logging: bool = True,
                 log_level: str = "INFO"):
        """
        Initialize the Pipeline Orchestrator.
        
        Parameters:
        -----------
        model_config : Dict, optional
            Model configuration parameters
        audio_config : Dict, optional
            Audio processing configuration
        output_dir : str
            Directory for saving outputs and logs
        enable_logging : bool
            Whether to enable detailed logging
        log_level : str
            Logging level (DEBUG, INFO, WARNING, ERROR)
        """
        # Default configurations
        self.model_config = model_config or {
            'input_dim': 441,
            'latent_dim': 100,
            'output_dim': 441,
            'encoder_hidden_dim': 256,
            'decoder_hidden_dim': 256,
            'num_layers': 2,
            'dropout': 0.2,
            'bidirectional': True
        }
        
        self.audio_config = audio_config or {
            'sr': 44100,
            'frame_length_ms': 20,
            'hop_length_ms': 10,
            'n_features': 441,
            'denoise_method': 'spectral_subtraction',
            'normalize_method': 'rms',
            'target_db': -20,
            'remove_silence_flag': True
        }
        
        # Setup directories
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "sessions").mkdir(exist_ok=True)
        (self.output_dir / "logs").mkdir(exist_ok=True)
        
        # Setup centralized logging and error handling
        self.enable_logging = enable_logging
        if enable_logging:
            from utils.logging_config import setup_logging
            setup_logging(log_dir=str(self.output_dir / "logs"), log_level=log_level)
            self.logging_manager = get_logging_manager()
            self.logger = self.logging_manager.get_logger('pipeline')
        else:
            self.logging_manager = None
            self.logger = None
        
        # Setup error handling
        self.error_handler = get_error_handler()
        
        # Initialize components
        self.model = None
        self.reconstruction_pipeline = None
        self.evaluator = None
        
        # Processing state
        self.current_session = None
        self.processing_history = []
        
        # Initialize components
        self._initialize_components()
    
    def _setup_logging(self, log_level: str):
        """Setup logging configuration."""
        log_file = self.output_dir / "logs" / f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        logging.basicConfig(
            level=getattr(logging, log_level.upper()),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler()
            ]
        )
        
        self.logger = logging.getLogger('PipelineOrchestrator')
        self.logger.info(f"Pipeline Orchestrator initialized with log level: {log_level}")
    
    def _initialize_components(self):
        """Initialize all pipeline components."""
        try:
            # Initialize model
            self.model = SpeechTranslationModel(**self.model_config)
            self._load_model_weights()
            
            # Initialize reconstruction pipeline
            self.reconstruction_pipeline = ReconstructionPipeline(
                sr=self.audio_config['sr'],
                output_dir=str(self.output_dir / "reconstruction"),
                save_intermediate=True
            )
            
            # Initialize evaluator
            self.evaluator = AudioEvaluator(sr=self.audio_config['sr'])
            
            if self.enable_logging:
                self.logger.info("All pipeline components initialized successfully")
                
        except Exception as e:
            error_msg = f"Failed to initialize pipeline components: {str(e)}"
            if self.enable_logging:
                self.logger.error(error_msg)
            raise RuntimeError(error_msg)
    
    def _load_model_weights(self):
        """Load trained model weights if available."""
        checkpoint_paths = [
            Path("test_checkpoints/cpu_validation/best_model.pt"),
            Path("checkpoints/best_model.pt"),
            Path("models/best_model.pt")
        ]
        
        for checkpoint_path in checkpoint_paths:
            if checkpoint_path.exists():
                try:
                    checkpoint = torch.load(checkpoint_path, map_location='cpu')
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self.model.load_state_dict(checkpoint)
                    
                    if self.enable_logging:
                        self.logger.info(f"Loaded model weights from: {checkpoint_path}")
                    return
                    
                except Exception as e:
                    if self.enable_logging:
                        self.logger.warning(f"Failed to load weights from {checkpoint_path}: {e}")
        
        if self.enable_logging:
            self.logger.warning("No trained model weights found. Using random initialization.")
        
        self.model.eval()
    
    def create_session(self, session_name: Optional[str] = None) -> str:
        """Create a new processing session."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = session_name or f"session_{timestamp}"
        
        self.current_session = {
            'session_id': session_id,
            'timestamp': timestamp,
            'stages': {},
            'results': [],
            'status': ProcessingStatus.PENDING
        }
        
        # Create session directory
        session_dir = self.output_dir / "sessions" / session_id
        session_dir.mkdir(exist_ok=True)
        
        if self.enable_logging:
            self.logger.info(f"Created new session: {session_id}")
        
        return session_id
    
    def process_audio_complete(self,
                             audio_input: Union[str, Path, np.ndarray],
                             sr: Optional[int] = None,
                             original_audio: Optional[np.ndarray] = None,
                             session_id: Optional[str] = None,
                             save_intermediate: bool = True) -> Dict[str, Any]:
        """
        Process audio through the complete pipeline.
        
        Parameters:
        -----------
        audio_input : Union[str, Path, np.ndarray]
            Audio file path or audio array
        sr : int, optional
            Sample rate (if audio_input is array)
        original_audio : np.ndarray, optional
            Original audio for comparison
        session_id : str, optional
            Session ID (creates new if None)
        save_intermediate : bool
            Whether to save intermediate results
        
        Returns:
        --------
        Dict containing complete processing results
        """
        # Create session if needed
        if session_id is None:
            session_id = self.create_session()
        
        start_time = datetime.now()
        
        try:
            # Load audio if needed
            if isinstance(audio_input, (str, Path)):
                audio_data, sample_rate = librosa.load(audio_input, sr=self.audio_config['sr'])
            else:
                audio_data = audio_input
                sample_rate = sr or self.audio_config['sr']
            
            # Stage 1: Preprocessing
            preprocessing_result = self._run_preprocessing(audio_data, sample_rate, session_id)
            
            # Stage 2: Feature Extraction
            feature_result = self._run_feature_extraction(
                preprocessing_result['output_info']['processed_audio'], 
                sample_rate, 
                session_id
            )
            
            # Stage 3: Model Inference
            inference_result = self._run_model_inference(
                feature_result['output_info']['feature_matrix'], 
                session_id
            )
            
            # Stage 4: Reconstruction
            reconstruction_result = self._run_reconstruction(
                inference_result['output_info']['reconstructed_features'],
                original_audio,
                session_id
            )
            
            # Stage 5: Evaluation (if original audio provided)
            evaluation_result = None
            if original_audio is not None:
                evaluation_result = self._run_evaluation(
                    original_audio,
                    reconstruction_result['output_info']['reconstructed_audio'],
                    session_id
                )
            
            # Compile complete results
            complete_result = self._compile_complete_result(
                session_id,
                [preprocessing_result, feature_result, inference_result, 
                 reconstruction_result, evaluation_result],
                start_time
            )
            
            # Save session results
            if save_intermediate:
                self._save_session_results(session_id, complete_result)
            
            # Update processing history
            self.processing_history.append(complete_result)
            
            if self.enable_logging:
                self.logger.info(f"Complete pipeline processing finished for session: {session_id}")
            
            return complete_result
            
        except Exception as e:
            error_msg = f"Pipeline processing failed: {str(e)}"
            if self.enable_logging:
                self.logger.error(f"{error_msg}\n{traceback.format_exc()}")
            
            # Create error result
            error_result = {
                'session_id': session_id,
                'status': ProcessingStatus.FAILED,
                'error': error_msg,
                'timestamp': datetime.now().isoformat(),
                'processing_time': (datetime.now() - start_time).total_seconds()
            }
            
            return error_result
    
    @error_handler_decorator(component="pipeline_orchestrator", operation="preprocessing")
    def _run_preprocessing(self, audio_data: np.ndarray, sr: int, session_id: str) -> ProcessingResult:
        """Run audio preprocessing stage."""
        stage_start = datetime.now()
        
        try:
            if self.enable_logging:
                self.logging_manager.log_structured(
                    LogLevel.INFO, LogCategory.PIPELINE, "preprocessing",
                    f"Starting preprocessing stage", session_id=session_id
                )
            
            # Validate input parameters
            if len(audio_data) == 0:
                raise AudioProcessingError(
                    "Empty audio data provided",
                    context=ErrorContext(
                        component="preprocessing",
                        operation="validate_input",
                        session_id=session_id
                    )
                )
            
            if sr <= 0:
                raise AudioProcessingError(
                    f"Invalid sample rate: {sr}",
                    context=ErrorContext(
                        component="preprocessing",
                        operation="validate_input",
                        session_id=session_id
                    )
                )
            
            processed_audio = preprocess_audio_complete(
                audio_data, sr,
                denoise_method=self.audio_config['denoise_method'],
                normalize_method=self.audio_config['normalize_method'],
                target_db=self.audio_config['target_db'],
                remove_silence_flag=self.audio_config['remove_silence_flag']
            )
            
            processing_time = (datetime.now() - stage_start).total_seconds()
            
            result = ProcessingResult(
                session_id=session_id,
                timestamp=datetime.now().isoformat(),
                stage=ProcessingStage.PREPROCESSING,
                status=ProcessingStatus.COMPLETED,
                input_info={
                    'original_length': len(audio_data),
                    'sample_rate': sr,
                    'duration': len(audio_data) / sr
                },
                output_info={
                    'processed_audio': processed_audio,
                    'processed_length': len(processed_audio),
                    'duration': len(processed_audio) / sr
                },
                metrics={
                    'length_ratio': len(processed_audio) / len(audio_data),
                    'rms_original': np.sqrt(np.mean(audio_data**2)),
                    'rms_processed': np.sqrt(np.mean(processed_audio**2))
                },
                processing_time=processing_time
            )
            
            if self.enable_logging:
                self.logging_manager.log_performance(
                    "preprocessing", "audio_preprocessing", processing_time, session_id
                )
            
            return result
            
        except AudioProcessingError:
            raise  # Re-raise our custom errors
        except Exception as e:
            processing_time = (datetime.now() - stage_start).total_seconds()
            
            # Convert to our custom error type
            audio_error = AudioProcessingError(
                f"Preprocessing failed: {str(e)}",
                context=ErrorContext(
                    component="preprocessing",
                    operation="preprocess_audio",
                    session_id=session_id,
                    input_data={'audio_length': len(audio_data), 'sample_rate': sr}
                ),
                original_error=e
            )
            
            return ProcessingResult(
                session_id=session_id,
                timestamp=datetime.now().isoformat(),
                stage=ProcessingStage.PREPROCESSING,
                status=ProcessingStatus.FAILED,
                input_info={'original_length': len(audio_data), 'sample_rate': sr},
                output_info={},
                metrics={},
                processing_time=processing_time,
                error_message=str(audio_error)
            )
    
    def _run_feature_extraction(self, audio_data: np.ndarray, sr: int, session_id: str) -> ProcessingResult:
        """Run feature extraction stage."""
        stage_start = datetime.now()
        
        try:
            if self.enable_logging:
                self.logger.info(f"[{session_id}] Starting feature extraction stage")
            
            feature_result = create_feature_matrix_advanced(
                audio_data, sr,
                frame_length_ms=self.audio_config['frame_length_ms'],
                hop_length_ms=self.audio_config['hop_length_ms'],
                n_features=self.audio_config['n_features'],
                include_spectral=False,
                include_mfcc=False
            )
            
            processing_time = (datetime.now() - stage_start).total_seconds()
            
            result = ProcessingResult(
                session_id=session_id,
                timestamp=datetime.now().isoformat(),
                stage=ProcessingStage.FEATURE_EXTRACTION,
                status=ProcessingStatus.COMPLETED,
                input_info={
                    'audio_length': len(audio_data),
                    'sample_rate': sr
                },
                output_info={
                    'feature_matrix': feature_result['feature_matrix'],
                    'matrix_shape': feature_result['feature_matrix'].shape,
                    'n_frames': feature_result['feature_matrix'].shape[0],
                    'n_features': feature_result['feature_matrix'].shape[1]
                },
                metrics={
                    'frames_per_second': feature_result['feature_matrix'].shape[0] / (len(audio_data) / sr),
                    'feature_density': np.mean(np.abs(feature_result['feature_matrix']))
                },
                processing_time=processing_time
            )
            
            if self.enable_logging:
                self.logger.info(f"[{session_id}] Feature extraction completed in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - stage_start).total_seconds()
            error_msg = f"Feature extraction failed: {str(e)}"
            
            if self.enable_logging:
                self.logger.error(f"[{session_id}] {error_msg}")
            
            return ProcessingResult(
                session_id=session_id,
                timestamp=datetime.now().isoformat(),
                stage=ProcessingStage.FEATURE_EXTRACTION,
                status=ProcessingStatus.FAILED,
                input_info={'audio_length': len(audio_data), 'sample_rate': sr},
                output_info={},
                metrics={},
                processing_time=processing_time,
                error_message=error_msg
            )
    
    @error_handler_decorator(component="pipeline_orchestrator", operation="model_inference")
    def _run_model_inference(self, feature_matrix: np.ndarray, session_id: str) -> ProcessingResult:
        """Run model inference stage."""
        stage_start = datetime.now()
        
        try:
            if self.enable_logging:
                self.logging_manager.log_structured(
                    LogLevel.INFO, LogCategory.MODEL, "model_inference",
                    f"Starting model inference stage", session_id=session_id
                )
            
            # Validate inputs
            if self.model is None:
                raise ModelInferenceError(
                    "Model not initialized",
                    context=ErrorContext(
                        component="model_inference",
                        operation="validate_model",
                        session_id=session_id
                    )
                )
            
            if feature_matrix.size == 0:
                raise ModelInferenceError(
                    "Empty feature matrix provided",
                    context=ErrorContext(
                        component="model_inference",
                        operation="validate_input",
                        session_id=session_id
                    )
                )
            
            # Convert to tensor and add batch dimension
            input_tensor = torch.FloatTensor(feature_matrix).unsqueeze(0)
            
            with torch.no_grad():
                self.model.eval()
                reconstructed, latent = self.model(input_tensor)
                
                # Remove batch dimension
                reconstructed_features = reconstructed.squeeze(0).numpy()
                latent_features = latent.squeeze(0).numpy()
            
            processing_time = (datetime.now() - stage_start).total_seconds()
            
            result = ProcessingResult(
                session_id=session_id,
                timestamp=datetime.now().isoformat(),
                stage=ProcessingStage.MODEL_INFERENCE,
                status=ProcessingStatus.COMPLETED,
                input_info={
                    'input_shape': feature_matrix.shape,
                    'n_frames': feature_matrix.shape[0],
                    'n_features': feature_matrix.shape[1]
                },
                output_info={
                    'reconstructed_features': reconstructed_features,
                    'latent_features': latent_features,
                    'output_shape': reconstructed_features.shape,
                    'latent_shape': latent_features.shape
                },
                metrics={
                    'reconstruction_mse': np.mean((feature_matrix - reconstructed_features)**2),
                    'latent_mean': np.mean(latent_features),
                    'latent_std': np.std(latent_features)
                },
                processing_time=processing_time
            )
            
            if self.enable_logging:
                self.logging_manager.log_performance(
                    "model_inference", "neural_network_inference", processing_time, session_id
                )
            
            return result
            
        except ModelInferenceError:
            raise  # Re-raise our custom errors
        except Exception as e:
            processing_time = (datetime.now() - stage_start).total_seconds()
            
            # Convert to our custom error type
            model_error = ModelInferenceError(
                f"Model inference failed: {str(e)}",
                context=ErrorContext(
                    component="model_inference",
                    operation="neural_network_forward",
                    session_id=session_id,
                    input_data={'feature_shape': feature_matrix.shape}
                ),
                original_error=e
            )
            
            return ProcessingResult(
                session_id=session_id,
                timestamp=datetime.now().isoformat(),
                stage=ProcessingStage.MODEL_INFERENCE,
                status=ProcessingStatus.FAILED,
                input_info={'input_shape': feature_matrix.shape},
                output_info={},
                metrics={},
                processing_time=processing_time,
                error_message=str(model_error)
            )
    
    @error_handler_decorator(component="pipeline_orchestrator", operation="reconstruction")
    def _run_reconstruction(self, feature_matrix: np.ndarray, original_audio: Optional[np.ndarray], session_id: str) -> ProcessingResult:
        """Run audio reconstruction stage."""
        stage_start = datetime.now()
        
        try:
            if self.enable_logging:
                self.logging_manager.log_structured(
                    LogLevel.INFO, LogCategory.RECONSTRUCTION, "audio_reconstruction",
                    f"Starting reconstruction stage", session_id=session_id
                )
            
            # Validate inputs
            if feature_matrix.size == 0:
                raise ReconstructionError(
                    "Empty feature matrix provided",
                    context=ErrorContext(
                        component="reconstruction",
                        operation="validate_input",
                        session_id=session_id
                    )
                )
            
            # Use reconstruction pipeline
            reconstruction_result = self.reconstruction_pipeline.process_single(
                feature_matrix,
                original_audio=original_audio,
                output_name=f"{session_id}_reconstruction",
                sr=self.audio_config['sr'],
                feature_type='raw'
            )
            
            processing_time = (datetime.now() - stage_start).total_seconds()
            
            result = ProcessingResult(
                session_id=session_id,
                timestamp=datetime.now().isoformat(),
                stage=ProcessingStage.RECONSTRUCTION,
                status=ProcessingStatus.COMPLETED,
                input_info={
                    'feature_shape': feature_matrix.shape,
                    'has_original': original_audio is not None
                },
                output_info={
                    'reconstructed_audio': reconstruction_result['reconstructed_audio'],
                    'audio_length': reconstruction_result['output_length'],
                    'duration': reconstruction_result['output_duration'],
                    'quality_score': reconstruction_result.get('quality_score', 0.0),
                    'audio_path': reconstruction_result.get('audio_path')
                },
                metrics=reconstruction_result.get('metrics', {}),
                processing_time=processing_time
            )
            
            if self.enable_logging:
                self.logging_manager.log_performance(
                    "reconstruction", "audio_synthesis", processing_time, session_id
                )
            
            return result
            
        except ReconstructionError:
            raise  # Re-raise our custom errors
        except Exception as e:
            processing_time = (datetime.now() - stage_start).total_seconds()
            
            # Convert to our custom error type
            reconstruction_error = ReconstructionError(
                f"Reconstruction failed: {str(e)}",
                context=ErrorContext(
                    component="reconstruction",
                    operation="audio_synthesis",
                    session_id=session_id,
                    input_data={
                        'feature_shape': feature_matrix.shape,
                        'has_original': original_audio is not None
                    }
                ),
                original_error=e
            )
            
            return ProcessingResult(
                session_id=session_id,
                timestamp=datetime.now().isoformat(),
                stage=ProcessingStage.RECONSTRUCTION,
                status=ProcessingStatus.FAILED,
                input_info={'feature_shape': feature_matrix.shape},
                output_info={},
                metrics={},
                processing_time=processing_time,
                error_message=str(reconstruction_error)
            )
    
    def _run_evaluation(self, original_audio: np.ndarray, reconstructed_audio: np.ndarray, session_id: str) -> ProcessingResult:
        """Run evaluation stage."""
        stage_start = datetime.now()
        
        try:
            if self.enable_logging:
                self.logger.info(f"[{session_id}] Starting evaluation stage")
            
            # Run detailed evaluation
            evaluation_metrics = self.evaluator.evaluate_reconstruction(
                original_audio, reconstructed_audio, detailed=True
            )
            
            processing_time = (datetime.now() - stage_start).total_seconds()
            
            result = ProcessingResult(
                session_id=session_id,
                timestamp=datetime.now().isoformat(),
                stage=ProcessingStage.EVALUATION,
                status=ProcessingStatus.COMPLETED,
                input_info={
                    'original_length': len(original_audio),
                    'reconstructed_length': len(reconstructed_audio)
                },
                output_info={
                    'evaluation_complete': True,
                    'quality_score': self.evaluator._compute_quality_score(evaluation_metrics)
                },
                metrics=evaluation_metrics,
                processing_time=processing_time
            )
            
            if self.enable_logging:
                self.logger.info(f"[{session_id}] Evaluation completed in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            processing_time = (datetime.now() - stage_start).total_seconds()
            error_msg = f"Evaluation failed: {str(e)}"
            
            if self.enable_logging:
                self.logger.error(f"[{session_id}] {error_msg}")
            
            return ProcessingResult(
                session_id=session_id,
                timestamp=datetime.now().isoformat(),
                stage=ProcessingStage.EVALUATION,
                status=ProcessingStatus.FAILED,
                input_info={
                    'original_length': len(original_audio),
                    'reconstructed_length': len(reconstructed_audio)
                },
                output_info={},
                metrics={},
                processing_time=processing_time,
                error_message=error_msg
            )
    
    def _compile_complete_result(self, session_id: str, stage_results: List[ProcessingResult], start_time: datetime) -> Dict[str, Any]:
        """Compile complete processing result from all stages."""
        total_time = (datetime.now() - start_time).total_seconds()
        
        # Filter out None results
        valid_results = [r for r in stage_results if r is not None]
        
        # Check overall status
        failed_stages = [r for r in valid_results if r.status == ProcessingStatus.FAILED]
        overall_status = ProcessingStatus.FAILED if failed_stages else ProcessingStatus.COMPLETED
        
        # Compile metrics
        all_metrics = {}
        for result in valid_results:
            if result.metrics:
                stage_name = result.stage.value
                all_metrics[stage_name] = result.metrics
        
        # Get final outputs
        final_outputs = {}
        for result in valid_results:
            if result.output_info:
                stage_name = result.stage.value
                final_outputs[stage_name] = result.output_info
        
        complete_result = {
            'session_id': session_id,
            'timestamp': datetime.now().isoformat(),
            'overall_status': overall_status.value,
            'total_processing_time': total_time,
            'stage_results': [asdict(r) for r in valid_results],
            'failed_stages': [r.stage.value for r in failed_stages],
            'metrics': all_metrics,
            'outputs': final_outputs,
            'summary': {
                'stages_completed': len([r for r in valid_results if r.status == ProcessingStatus.COMPLETED]),
                'stages_failed': len(failed_stages),
                'total_stages': len(valid_results),
                'success_rate': len([r for r in valid_results if r.status == ProcessingStatus.COMPLETED]) / len(valid_results) if valid_results else 0
            }
        }
        
        return complete_result
    
    def _save_session_results(self, session_id: str, results: Dict[str, Any]):
        """Save session results to file."""
        try:
            session_file = self.output_dir / "sessions" / session_id / "results.json"
            
            # Convert numpy arrays to lists for JSON serialization
            serializable_results = self._make_json_serializable(results)
            
            with open(session_file, 'w') as f:
                json.dump(serializable_results, f, indent=2)
            
            if self.enable_logging:
                self.logger.info(f"Session results saved to: {session_file}")
                
        except Exception as e:
            if self.enable_logging:
                self.logger.error(f"Failed to save session results: {e}")
    
    def _make_json_serializable(self, obj):
        """Convert numpy arrays and other non-serializable objects to JSON-compatible format."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return {
                '_type': 'numpy_array',
                'shape': obj.shape,
                'dtype': str(obj.dtype),
                'data': obj.tolist()
            }
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj
    
    def get_session_summary(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get summary of a specific session."""
        for result in self.processing_history:
            if result.get('session_id') == session_id:
                return result.get('summary')
        return None
    
    def get_processing_statistics(self) -> Dict[str, Any]:
        """Get overall processing statistics."""
        if not self.processing_history:
            return {'total_sessions': 0, 'message': 'No processing history available'}
        
        total_sessions = len(self.processing_history)
        successful_sessions = len([r for r in self.processing_history if r.get('overall_status') == 'completed'])
        
        avg_processing_time = np.mean([r.get('total_processing_time', 0) for r in self.processing_history])
        
        return {
            'total_sessions': total_sessions,
            'successful_sessions': successful_sessions,
            'success_rate': successful_sessions / total_sessions,
            'average_processing_time': avg_processing_time,
            'total_processing_time': sum([r.get('total_processing_time', 0) for r in self.processing_history])
        }
    
    def cleanup_session(self, session_id: str):
        """Clean up session files and data."""
        try:
            session_dir = self.output_dir / "sessions" / session_id
            if session_dir.exists():
                import shutil
                shutil.rmtree(session_dir)
                
            if self.enable_logging:
                self.logger.info(f"Cleaned up session: {session_id}")
                
        except Exception as e:
            if self.enable_logging:
                self.logger.error(f"Failed to cleanup session {session_id}: {e}")