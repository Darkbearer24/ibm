# Speech Translation System - API Documentation

## Overview

The Speech Translation System provides a comprehensive pipeline for processing audio files through advanced machine learning models. This documentation covers all public APIs, configuration options, and usage examples.

## Table of Contents

1. [Pipeline Orchestrator API](#pipeline-orchestrator-api)
2. [Audio Processing APIs](#audio-processing-apis)
3. [Model APIs](#model-apis)
4. [Utility APIs](#utility-apis)
5. [Error Handling](#error-handling)
6. [Configuration](#configuration)
7. [Examples](#examples)

---

## Pipeline Orchestrator API

### PipelineOrchestrator Class

The main orchestrator class that manages the complete speech translation pipeline.

#### Constructor

```python
PipelineOrchestrator(
    model_config: Optional[Dict] = None,
    audio_config: Optional[Dict] = None,
    output_dir: str = "outputs/pipeline",
    enable_logging: bool = True,
    log_level: str = "INFO"
)
```

**Parameters:**
- `model_config` (Dict, optional): Model configuration parameters
- `audio_config` (Dict, optional): Audio processing configuration
- `output_dir` (str): Directory for saving outputs and logs
- `enable_logging` (bool): Whether to enable detailed logging
- `log_level` (str): Logging level (DEBUG, INFO, WARNING, ERROR)

**Default Model Configuration:**
```python
{
    'input_dim': 441,
    'latent_dim': 100,
    'output_dim': 441,
    'encoder_hidden_dim': 256,
    'decoder_hidden_dim': 256,
    'num_layers': 2,
    'dropout': 0.2,
    'bidirectional': True
}
```

**Default Audio Configuration:**
```python
{
    'sr': 44100,
    'frame_length_ms': 20,
    'hop_length_ms': 10,
    'n_features': 441,
    'denoise_method': 'spectral_subtraction',
    'normalize_method': 'rms',
    'target_db': -20,
    'remove_silence_flag': True
}
```

#### Methods

##### process_audio_complete

```python
process_audio_complete(
    audio_input: Union[str, Path, np.ndarray],
    sr: Optional[int] = None,
    original_audio: Optional[np.ndarray] = None,
    session_id: Optional[str] = None,
    save_intermediate: bool = True
) -> Dict[str, Any]
```

Process audio through the complete pipeline.

**Parameters:**
- `audio_input`: Audio file path or audio array
- `sr`: Sample rate (required if audio_input is array)
- `original_audio`: Original audio for comparison
- `session_id`: Session ID (creates new if None)
- `save_intermediate`: Whether to save intermediate results

**Returns:**
```python
{
    'success': bool,
    'session_id': str,
    'processing_time': float,
    'preprocessing': {
        'execution_time': float,
        'memory_usage': float,
        'audio_length': float,
        'sample_rate': int
    },
    'inference': {
        'execution_time': float,
        'memory_usage': float,
        'latent_features': np.ndarray,
        'model_output': np.ndarray
    },
    'reconstruction': {
        'execution_time': float,
        'memory_usage': float,
        'reconstructed_audio': np.ndarray,
        'quality_score': float
    },
    'files': {
        'original_audio': str,
        'preprocessed_audio': str,
        'reconstructed_audio': str,
        'feature_matrix': str
    },
    'statistics': {
        'total_processing_time': float,
        'real_time_factor': float,
        'memory_peak': float,
        'quality_metrics': Dict
    }
}
```

##### create_session

```python
create_session() -> str
```

Create a new processing session with unique ID.

**Returns:** Session ID string

##### get_session_info

```python
get_session_info(session_id: str) -> Dict[str, Any]
```

Retrieve information about a processing session.

**Parameters:**
- `session_id`: Session identifier

**Returns:** Session information dictionary

---

## Audio Processing APIs

### Preprocessing Functions

#### preprocess_audio_complete

```python
from utils.denoise import preprocess_audio_complete

preprocess_audio_complete(
    audio: np.ndarray,
    sr: int,
    denoise_method: str = 'spectral_subtraction',
    normalize_method: str = 'rms',
    target_db: float = -20,
    remove_silence_flag: bool = True
) -> Tuple[np.ndarray, Dict[str, Any]]
```

Complete audio preprocessing pipeline.

**Parameters:**
- `audio`: Input audio array
- `sr`: Sample rate
- `denoise_method`: Denoising method ('spectral_subtraction', 'wiener', 'none')
- `normalize_method`: Normalization method ('rms', 'peak', 'lufs')
- `target_db`: Target dB level for normalization
- `remove_silence_flag`: Whether to remove silence

**Returns:**
- Preprocessed audio array
- Processing statistics dictionary

### Feature Extraction Functions

#### create_feature_matrix_advanced

```python
from utils.framing import create_feature_matrix_advanced

create_feature_matrix_advanced(
    audio: np.ndarray,
    sr: int,
    frame_length_ms: float = 20,
    hop_length_ms: float = 10,
    n_features: int = 441,
    feature_type: str = 'mfcc'
) -> Tuple[np.ndarray, Dict[str, Any]]
```

Create advanced feature matrix from audio.

**Parameters:**
- `audio`: Input audio array
- `sr`: Sample rate
- `frame_length_ms`: Frame length in milliseconds
- `hop_length_ms`: Hop length in milliseconds
- `n_features`: Number of features to extract
- `feature_type`: Type of features ('mfcc', 'mel', 'stft')

**Returns:**
- Feature matrix (frames × features)
- Feature extraction metadata

---

## Model APIs

### Encoder-Decoder Model

#### create_model

```python
from models.encoder_decoder import create_model

create_model(
    input_dim: int = 441,
    latent_dim: int = 100,
    output_dim: int = 441,
    encoder_hidden_dim: int = 256,
    decoder_hidden_dim: int = 256,
    num_layers: int = 2,
    dropout: float = 0.2,
    bidirectional: bool = True
) -> Tuple[torch.nn.Module, Dict[str, Any]]
```

Create encoder-decoder model with specified configuration.

**Parameters:**
- `input_dim`: Input feature dimension
- `latent_dim`: Latent space dimension
- `output_dim`: Output feature dimension
- `encoder_hidden_dim`: Encoder hidden layer dimension
- `decoder_hidden_dim`: Decoder hidden layer dimension
- `num_layers`: Number of LSTM layers
- `dropout`: Dropout probability
- `bidirectional`: Whether to use bidirectional LSTM

**Returns:**
- PyTorch model instance
- Model configuration dictionary

#### Model Forward Pass

```python
model = EncoderDecoderModel(...)
output, latent = model(input_features)
```

**Input:**
- `input_features`: Tensor of shape (batch_size, sequence_length, input_dim)

**Output:**
- `output`: Reconstructed features (batch_size, sequence_length, output_dim)
- `latent`: Latent representations (batch_size, sequence_length, latent_dim)

---

## Utility APIs

### Logging Configuration

#### LoggingManager

```python
from utils.logging_config import LoggingManager, get_logging_manager

# Get singleton instance
logger_manager = get_logging_manager()

# Log messages
logger_manager.log_info("Processing started", component="pipeline")
logger_manager.log_error("Processing failed", component="model", error=exception)
logger_manager.log_performance("inference_time", 0.5, component="model")
```

#### Available Log Methods

- `log_debug(message, component, **kwargs)`
- `log_info(message, component, **kwargs)`
- `log_warning(message, component, **kwargs)`
- `log_error(message, component, error=None, **kwargs)`
- `log_performance(metric_name, value, component, **kwargs)`
- `log_system_resource(resource_type, value, component, **kwargs)`

### Error Handling

#### Custom Exceptions

```python
from utils.error_handling import (
    SpeechTranslationError,
    AudioProcessingError,
    ModelInferenceError,
    ReconstructionError,
    ConfigurationError
)

# Usage
try:
    result = process_audio(audio_data)
except AudioProcessingError as e:
    print(f"Audio processing failed: {e}")
    print(f"Error code: {e.error_code}")
    print(f"Context: {e.context}")
```

#### Error Handler Decorator

```python
from utils.error_handling import error_handler_decorator

@error_handler_decorator
def my_function():
    # Function implementation
    pass
```

---

## Configuration

### Environment Variables

```bash
# Logging configuration
LOG_LEVEL=INFO
LOG_DIR=logs
ENABLE_PERFORMANCE_LOGGING=true

# Model configuration
MODEL_CACHE_DIR=models
DEVICE=cpu  # or cuda

# Audio processing
DEFAULT_SAMPLE_RATE=44100
MAX_AUDIO_LENGTH=300  # seconds
```

### Configuration Files

#### model_config.json
```json
{
    "input_dim": 441,
    "latent_dim": 100,
    "output_dim": 441,
    "encoder_hidden_dim": 256,
    "decoder_hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.2,
    "bidirectional": true
}
```

#### audio_config.json
```json
{
    "sr": 44100,
    "frame_length_ms": 20,
    "hop_length_ms": 10,
    "n_features": 441,
    "denoise_method": "spectral_subtraction",
    "normalize_method": "rms",
    "target_db": -20,
    "remove_silence_flag": true
}
```

---

## Examples

### Basic Usage

```python
from utils.pipeline_orchestrator import PipelineOrchestrator

# Initialize orchestrator
orchestrator = PipelineOrchestrator()

# Process audio file
result = orchestrator.process_audio_complete(
    audio_input="path/to/audio.wav",
    session_id="my_session"
)

if result['success']:
    print(f"Processing completed in {result['processing_time']:.2f}s")
    print(f"Quality score: {result['reconstruction']['quality_score']:.3f}")
else:
    print(f"Processing failed: {result.get('error', 'Unknown error')}")
```

### Custom Configuration

```python
# Custom model configuration
model_config = {
    'input_dim': 512,
    'latent_dim': 128,
    'encoder_hidden_dim': 512,
    'num_layers': 3
}

# Custom audio configuration
audio_config = {
    'sr': 48000,
    'denoise_method': 'wiener',
    'target_db': -15
}

# Initialize with custom config
orchestrator = PipelineOrchestrator(
    model_config=model_config,
    audio_config=audio_config,
    output_dir="custom_outputs"
)
```

### Batch Processing

```python
import os
from pathlib import Path

orchestrator = PipelineOrchestrator()
audio_files = list(Path("audio_directory").glob("*.wav"))

results = []
for audio_file in audio_files:
    result = orchestrator.process_audio_complete(
        audio_input=str(audio_file),
        session_id=f"batch_{audio_file.stem}"
    )
    results.append({
        'file': audio_file.name,
        'success': result['success'],
        'processing_time': result.get('processing_time', 0),
        'quality_score': result.get('reconstruction', {}).get('quality_score', 0)
    })

# Print summary
successful = sum(1 for r in results if r['success'])
print(f"Processed {successful}/{len(results)} files successfully")
```

### Error Handling Example

```python
from utils.error_handling import (
    AudioProcessingError,
    ModelInferenceError,
    ReconstructionError
)

try:
    result = orchestrator.process_audio_complete("audio.wav")
except AudioProcessingError as e:
    print(f"Audio processing failed: {e}")
    # Handle audio-specific errors
except ModelInferenceError as e:
    print(f"Model inference failed: {e}")
    # Handle model-specific errors
except ReconstructionError as e:
    print(f"Audio reconstruction failed: {e}")
    # Handle reconstruction-specific errors
except Exception as e:
    print(f"Unexpected error: {e}")
    # Handle other errors
```

### Performance Monitoring

```python
from utils.logging_config import get_logging_manager

logger = get_logging_manager()

# Enable performance logging
orchestrator = PipelineOrchestrator(enable_logging=True, log_level="DEBUG")

# Process with detailed logging
result = orchestrator.process_audio_complete("audio.wav")

# Check performance metrics
stats = result.get('statistics', {})
print(f"Real-time factor: {stats.get('real_time_factor', 0):.2f}x")
print(f"Peak memory: {stats.get('memory_peak', 0):.1f}MB")
```

---

## Performance Considerations

### Memory Usage

- The system typically uses 100-500MB of memory for processing
- Memory usage scales with audio length and model complexity
- Use `save_intermediate=False` to reduce memory footprint

### Processing Speed

- Average processing speed: 50-100x real-time on modern CPUs
- GPU acceleration available for model inference
- Concurrent processing supported for batch operations

### Optimization Tips

1. **Batch Processing**: Process multiple files in sequence to amortize initialization costs
2. **Memory Management**: Use appropriate audio chunk sizes for long files
3. **Model Caching**: Models are cached after first load for faster subsequent processing
4. **Configuration Tuning**: Adjust frame sizes and feature dimensions based on requirements

---

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all dependencies are installed (`pip install -r requirements.txt`)
2. **Audio Format Issues**: Supported formats: WAV, MP3, FLAC, M4A
3. **Memory Errors**: Reduce audio length or adjust model configuration
4. **Performance Issues**: Check system resources and consider GPU acceleration

### Debug Mode

```python
# Enable debug logging
orchestrator = PipelineOrchestrator(
    enable_logging=True,
    log_level="DEBUG"
)

# Check logs in output directory
# logs/pipeline_YYYYMMDD_HHMMSS.log
```

### Support

For additional support and examples, refer to:
- User Guide: `docs/User_Guide.md`
- Deployment Instructions: `docs/Deployment_Guide.md`
- Example notebooks: `notebooks/`