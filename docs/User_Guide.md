# Speech Translation System - User Guide

## Table of Contents

1. [Introduction](#introduction)
2. [Getting Started](#getting-started)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Using the Web Interface](#using-the-web-interface)
6. [Command Line Usage](#command-line-usage)
7. [Configuration](#configuration)
8. [Understanding Results](#understanding-results)
9. [Best Practices](#best-practices)
10. [Troubleshooting](#troubleshooting)
11. [Advanced Usage](#advanced-usage)

---

## Introduction

The Speech Translation System is an advanced machine learning pipeline designed to process and analyze audio files. The system provides:

- **Audio Preprocessing**: Noise reduction, normalization, and silence removal
- **Feature Extraction**: Advanced audio feature analysis
- **Model Inference**: Deep learning-based audio processing
- **Audio Reconstruction**: High-quality audio reconstruction
- **Quality Assessment**: Comprehensive quality metrics

### Key Features

✅ **Real-time Processing**: Process audio faster than real-time  
✅ **High Quality**: Advanced denoising and reconstruction algorithms  
✅ **User-friendly Interface**: Both web UI and command-line options  
✅ **Comprehensive Logging**: Detailed processing logs and metrics  
✅ **Batch Processing**: Handle multiple files efficiently  
✅ **Flexible Configuration**: Customizable processing parameters  

---

## Getting Started

### System Requirements

**Minimum Requirements:**
- Python 3.8 or higher
- 4GB RAM
- 2GB free disk space

**Recommended:**
- Python 3.9+
- 8GB+ RAM
- GPU with CUDA support (optional, for faster processing)
- SSD storage for better I/O performance

### Supported Audio Formats

- **WAV** (recommended for best quality)
- **MP3**
- **FLAC**
- **M4A**
- **OGG**

### Audio Specifications

- **Sample Rate**: 44.1kHz (recommended), supports 16kHz-48kHz
- **Bit Depth**: 16-bit or 24-bit
- **Channels**: Mono or Stereo (converted to mono for processing)
- **Duration**: Up to 5 minutes per file (longer files supported with chunking)

---

## Installation

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd speech-translation-system
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation

```bash
python -c "from utils.pipeline_orchestrator import PipelineOrchestrator; print('Installation successful!')"
```

---

## Quick Start

### Option 1: Web Interface (Recommended for Beginners)

1. **Start the Web Application**
   ```bash
   streamlit run app.py
   ```

2. **Open Your Browser**
   - Navigate to `http://localhost:8501`
   - The web interface will load automatically

3. **Upload and Process Audio**
   - Click "Browse files" to select your audio file
   - Click "Process Audio" to start processing
   - View results and download processed files

### Option 2: Python Script

```python
from utils.pipeline_orchestrator import PipelineOrchestrator

# Initialize the system
orchestrator = PipelineOrchestrator()

# Process your audio file
result = orchestrator.process_audio_complete(
    audio_input="path/to/your/audio.wav"
)

# Check results
if result['success']:
    print(f"Processing completed successfully!")
    print(f"Quality score: {result['reconstruction']['quality_score']:.3f}")
    print(f"Processing time: {result['processing_time']:.2f} seconds")
else:
    print(f"Processing failed: {result.get('error', 'Unknown error')}")
```

---

## Using the Web Interface

### Main Dashboard

The web interface provides an intuitive way to process audio files:

#### 1. File Upload Section
- **Drag and Drop**: Simply drag your audio file into the upload area
- **Browse Files**: Click to select files from your computer
- **File Validation**: The system automatically validates file format and size

#### 2. Processing Options
- **Session ID**: Optional identifier for tracking your processing session
- **Save Intermediate Files**: Choose whether to save preprocessing steps
- **Advanced Settings**: Access to detailed configuration options

#### 3. Processing Results

**Audio Player**: Listen to original and processed audio  
**Statistics Panel**: View detailed processing metrics  
**Quality Metrics**: Comprehensive quality assessment  
**Download Options**: Download processed files and reports  

#### 4. Real-time Monitoring
- **Progress Bar**: Visual processing progress
- **Live Logs**: Real-time processing status
- **Performance Metrics**: Memory usage and processing speed

### Advanced Web Features

#### Batch Processing
1. Upload multiple files using Ctrl+Click or Cmd+Click
2. Configure batch processing options
3. Monitor progress for each file
4. Download results as a ZIP archive

#### Configuration Presets
- **High Quality**: Best quality, slower processing
- **Balanced**: Good quality, moderate speed
- **Fast**: Lower quality, fastest processing
- **Custom**: Define your own parameters

---

## Command Line Usage

### Basic Processing

```bash
python -m utils.pipeline_orchestrator --input audio.wav --output results/
```

### Advanced Options

```bash
python -m utils.pipeline_orchestrator \
    --input audio.wav \
    --output results/ \
    --config custom_config.json \
    --session-id my_session \
    --log-level DEBUG \
    --save-intermediate
```

### Batch Processing

```bash
# Process all WAV files in a directory
python batch_process.py --input-dir audio_files/ --output-dir results/

# Process with custom configuration
python batch_process.py \
    --input-dir audio_files/ \
    --output-dir results/ \
    --config high_quality.json \
    --parallel 4
```

### Performance Benchmarking

```bash
# Run performance benchmarks
python benchmark_performance.py

# Generate detailed performance report
python benchmark_performance.py --detailed --output benchmark_report.json
```

---

## Configuration

### Configuration Files

The system uses JSON configuration files for customization:

#### Model Configuration (`model_config.json`)

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

#### Audio Configuration (`audio_config.json`)

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

### Configuration Options Explained

#### Audio Processing Options

**Denoising Methods:**
- `spectral_subtraction`: Good for stationary noise
- `wiener`: Better for non-stationary noise
- `none`: Skip denoising (fastest)

**Normalization Methods:**
- `rms`: Root Mean Square normalization (recommended)
- `peak`: Peak normalization
- `lufs`: Loudness Units relative to Full Scale

**Frame Parameters:**
- `frame_length_ms`: Analysis window size (10-50ms)
- `hop_length_ms`: Step size between frames (5-25ms)
- `n_features`: Number of features to extract (128-1024)

#### Model Parameters

**Architecture:**
- `input_dim`: Input feature dimension
- `latent_dim`: Compressed representation size
- `encoder_hidden_dim`: Encoder network size
- `decoder_hidden_dim`: Decoder network size
- `num_layers`: Number of LSTM layers (1-4)

**Training:**
- `dropout`: Regularization strength (0.0-0.5)
- `bidirectional`: Use bidirectional LSTM

### Environment Variables

```bash
# Set in your .env file or environment
LOG_LEVEL=INFO
LOG_DIR=logs
OUTPUT_DIR=outputs
MODEL_CACHE_DIR=models
MAX_AUDIO_LENGTH=300
DEVICE=cpu  # or cuda
```

---

## Understanding Results

### Processing Output Structure

After processing, you'll receive a comprehensive results dictionary:

```python
{
    'success': True,
    'session_id': 'session_20240101_120000',
    'processing_time': 2.45,
    
    'preprocessing': {
        'execution_time': 0.8,
        'memory_usage': 45.2,
        'audio_length': 5.0,
        'sample_rate': 44100,
        'noise_reduction_db': 12.3
    },
    
    'inference': {
        'execution_time': 1.2,
        'memory_usage': 120.5,
        'latent_features_shape': [500, 100],
        'model_confidence': 0.92
    },
    
    'reconstruction': {
        'execution_time': 0.45,
        'memory_usage': 67.8,
        'quality_score': 0.89,
        'snr_improvement': 8.5
    },
    
    'files': {
        'original_audio': 'outputs/session_xxx/original.wav',
        'preprocessed_audio': 'outputs/session_xxx/preprocessed.wav',
        'reconstructed_audio': 'outputs/session_xxx/reconstructed.wav',
        'feature_matrix': 'outputs/session_xxx/features.npy'
    },
    
    'statistics': {
        'total_processing_time': 2.45,
        'real_time_factor': 2.04,
        'memory_peak': 145.7,
        'quality_metrics': {
            'mse': 0.023,
            'psnr': 34.2,
            'ssim': 0.91
        }
    }
}
```

### Key Metrics Explained

#### Performance Metrics

- **Processing Time**: Total time to process the audio
- **Real-time Factor**: How much faster than real-time (>1.0 is good)
- **Memory Usage**: Peak memory consumption in MB

#### Quality Metrics

- **Quality Score**: Overall quality rating (0-1, higher is better)
- **SNR Improvement**: Signal-to-noise ratio improvement in dB
- **MSE**: Mean Squared Error (lower is better)
- **PSNR**: Peak Signal-to-Noise Ratio (higher is better)
- **SSIM**: Structural Similarity Index (0-1, higher is better)

#### Audio Characteristics

- **Audio Length**: Duration of processed audio in seconds
- **Sample Rate**: Audio sample rate in Hz
- **Noise Reduction**: Amount of noise reduced in dB

---

## Best Practices

### Audio Preparation

1. **Use High-Quality Source Audio**
   - Prefer WAV format over compressed formats
   - Use 44.1kHz sample rate when possible
   - Ensure adequate recording levels (avoid clipping)

2. **File Organization**
   - Keep original files as backups
   - Use descriptive filenames
   - Organize by project or date

3. **Preprocessing Considerations**
   - Enable noise reduction for noisy recordings
   - Use silence removal for speech content
   - Adjust normalization based on content type

### Performance Optimization

1. **System Resources**
   - Close unnecessary applications
   - Ensure adequate free disk space
   - Use SSD storage for better performance

2. **Batch Processing**
   - Process multiple files together
   - Use appropriate parallel processing settings
   - Monitor system resources during batch jobs

3. **Configuration Tuning**
   - Start with default settings
   - Adjust based on your specific needs
   - Test different configurations on sample files

### Quality Assurance

1. **Always Listen to Results**
   - Compare original and processed audio
   - Check for artifacts or distortions
   - Verify quality metrics make sense

2. **Monitor Processing Logs**
   - Check for warnings or errors
   - Review performance metrics
   - Identify potential issues early

3. **Validate Results**
   - Use quality metrics as guidelines
   - Trust your ears for final assessment
   - Keep notes on successful configurations

---

## Troubleshooting

### Common Issues and Solutions

#### Installation Problems

**Issue**: Import errors or missing dependencies
```bash
# Solution: Reinstall dependencies
pip install --upgrade -r requirements.txt

# For specific package issues:
pip install --force-reinstall package_name
```

**Issue**: Python version compatibility
```bash
# Check Python version
python --version

# Ensure Python 3.8+
# Consider using pyenv for version management
```

#### Audio Processing Issues

**Issue**: "Unsupported audio format"
- **Solution**: Convert to WAV format using audio editing software
- **Alternative**: Install additional audio codecs

**Issue**: "Audio file too large"
- **Solution**: Split large files into smaller chunks
- **Alternative**: Increase memory limits in configuration

**Issue**: Poor quality results
- **Solution**: Adjust denoising and normalization settings
- **Check**: Input audio quality and recording conditions

#### Performance Issues

**Issue**: Slow processing speed
- **Check**: System resources (CPU, memory)
- **Solution**: Reduce audio quality settings
- **Alternative**: Use GPU acceleration if available

**Issue**: High memory usage
- **Solution**: Process shorter audio segments
- **Alternative**: Reduce model complexity

#### Web Interface Issues

**Issue**: Cannot access web interface
```bash
# Check if Streamlit is running
streamlit run app.py --server.port 8501

# Try different port
streamlit run app.py --server.port 8502
```

**Issue**: Upload failures
- **Check**: File size limits
- **Solution**: Ensure stable internet connection
- **Alternative**: Use command-line interface

### Debug Mode

Enable detailed logging for troubleshooting:

```python
from utils.pipeline_orchestrator import PipelineOrchestrator

# Enable debug logging
orchestrator = PipelineOrchestrator(
    enable_logging=True,
    log_level="DEBUG"
)

# Process with detailed logs
result = orchestrator.process_audio_complete("audio.wav")

# Check log files in outputs/logs/
```

### Getting Help

1. **Check Log Files**: Look in `outputs/logs/` for detailed error messages
2. **Review Documentation**: Consult API documentation for detailed parameter descriptions
3. **Test with Sample Files**: Use provided sample audio files to isolate issues
4. **System Information**: Note your OS, Python version, and hardware specifications

---

## Advanced Usage

### Custom Model Training

```python
from models.encoder_decoder import create_model, train_model
from utils.data_loader import AudioDataLoader

# Create custom model
model, config = create_model(
    input_dim=512,
    latent_dim=128,
    encoder_hidden_dim=512
)

# Load training data
data_loader = AudioDataLoader("training_data/")

# Train model
trained_model = train_model(
    model=model,
    data_loader=data_loader,
    epochs=100,
    learning_rate=0.001
)
```

### Integration with Other Systems

```python
# REST API integration
import requests

response = requests.post(
    "http://localhost:8000/process",
    files={"audio": open("audio.wav", "rb")},
    data={"config": "high_quality"}
)

result = response.json()
```

### Automated Workflows

```python
# Automated processing pipeline
import os
import json
from pathlib import Path

def process_directory(input_dir, output_dir, config_file=None):
    """Process all audio files in a directory."""
    
    orchestrator = PipelineOrchestrator()
    
    audio_files = list(Path(input_dir).glob("*.wav"))
    results = []
    
    for audio_file in audio_files:
        print(f"Processing {audio_file.name}...")
        
        result = orchestrator.process_audio_complete(
            audio_input=str(audio_file),
            session_id=f"batch_{audio_file.stem}"
        )
        
        results.append({
            "file": audio_file.name,
            "success": result["success"],
            "quality_score": result.get("reconstruction", {}).get("quality_score", 0),
            "processing_time": result.get("processing_time", 0)
        })
    
    # Save batch results
    with open(Path(output_dir) / "batch_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

# Usage
results = process_directory("input_audio/", "output_results/")
print(f"Processed {len(results)} files")
```

### Performance Monitoring

```python
import time
import psutil
from utils.logging_config import get_logging_manager

class PerformanceMonitor:
    def __init__(self):
        self.logger = get_logging_manager()
        self.start_time = None
        self.start_memory = None
    
    def start_monitoring(self):
        self.start_time = time.time()
        self.start_memory = psutil.virtual_memory().used
    
    def log_performance(self, operation_name):
        end_time = time.time()
        end_memory = psutil.virtual_memory().used
        
        duration = end_time - self.start_time
        memory_delta = (end_memory - self.start_memory) / 1024 / 1024  # MB
        
        self.logger.log_performance(
            f"{operation_name}_duration", duration, "monitor"
        )
        self.logger.log_performance(
            f"{operation_name}_memory", memory_delta, "monitor"
        )

# Usage
monitor = PerformanceMonitor()
monitor.start_monitoring()

# Your processing code here
result = orchestrator.process_audio_complete("audio.wav")

monitor.log_performance("audio_processing")
```

---

## Conclusion

This user guide provides comprehensive information for using the Speech Translation System effectively. For additional technical details, refer to:

- **API Documentation**: `docs/API_Documentation.md`
- **Deployment Guide**: `docs/Deployment_Guide.md`
- **Example Notebooks**: `notebooks/`

Remember to:
- Start with default settings and adjust as needed
- Monitor performance and quality metrics
- Keep backups of original audio files
- Refer to logs for troubleshooting

Happy processing! 🎵