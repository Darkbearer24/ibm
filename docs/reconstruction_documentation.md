# Signal Reconstruction & Evaluation Documentation

## Overview

This documentation covers the signal reconstruction and evaluation system implemented in Sprint 6. The system provides comprehensive tools for reconstructing audio signals from various feature representations and evaluating the quality of reconstructions.

## Architecture

### Core Components

1. **Reconstruction Module** (`utils/reconstruction.py`)
   - Overlap-add reconstruction algorithm
   - Support for multiple feature types (raw, spectral, MFCC)
   - Quality metrics computation

2. **Evaluation Module** (`utils/evaluation.py`)
   - Comprehensive audio quality assessment
   - Multiple evaluation metrics
   - Visualization and reporting tools

3. **Pipeline Module** (`utils/reconstruction_pipeline.py`)
   - End-to-end processing pipeline
   - Batch processing capabilities
   - Automated quality assessment

## Reconstruction Process

### Overlap-Add Algorithm

The core reconstruction uses the overlap-add method:

```python
def reconstruct_audio_overlap_add(feature_matrix, sr=22050, frame_length=512, 
                                  hop_length=256, window='hann', feature_type='raw'):
```

**Parameters:**
- `feature_matrix`: Input features (n_frames, n_features)
- `sr`: Sample rate (default: 22050 Hz)
- `frame_length`: Frame size in samples (default: 512)
- `hop_length`: Hop size in samples (default: 256)
- `window`: Window function ('hann', 'hamming', 'blackman')
- `feature_type`: Type of features ('raw', 'spectral', 'mfcc')

### Feature Type Support

#### 1. Raw Features
- Direct audio frames
- Windowing and overlap-add reconstruction
- Best for time-domain representations

#### 2. Spectral Features
- STFT magnitude spectra
- Phase reconstruction using Griffin-Lim algorithm
- Suitable for frequency-domain analysis

#### 3. MFCC Features
- Mel-frequency cepstral coefficients
- Inverse DCT and mel-scale conversion
- Compact representation for speech/audio

## Evaluation Metrics

### Basic Metrics

1. **Mean Squared Error (MSE)**
   ```
   MSE = mean((original - reconstructed)²)
   ```

2. **Signal-to-Noise Ratio (SNR)**
   ```
   SNR = 10 * log10(signal_power / noise_power)
   ```

3. **Correlation Coefficient**
   ```
   r = correlation(original, reconstructed)
   ```

### Advanced Metrics

1. **Spectral Distance**
   - Euclidean distance between STFT magnitudes
   - Measures frequency-domain similarity

2. **MFCC Distance**
   - Distance between MFCC representations
   - Perceptually relevant for speech/audio

3. **Zero Crossing Rate (ZCR) Difference**
   - Temporal characteristic preservation

4. **Energy and Dynamic Range Ratios**
   - Overall loudness and dynamic preservation

5. **Spectral Centroid/Rolloff Differences**
   - Timbral characteristic preservation

### Quality Score

Composite quality score combining multiple metrics:

```python
quality_score = (snr_norm + corr_norm + (1 - mse_norm) + energy_norm) / 4
```

Where each component is normalized to [0, 1] range.

## Usage Examples

### Basic Reconstruction

```python
from utils.reconstruction import reconstruct_audio_overlap_add

# Reconstruct from raw features
audio = reconstruct_audio_overlap_add(
    feature_matrix, 
    feature_type='raw',
    sr=22050
)
```

### Pipeline Processing

```python
from utils.reconstruction_pipeline import ReconstructionPipeline

pipeline = ReconstructionPipeline(output_dir='results')
result = pipeline.process_single(
    feature_matrix,
    original_audio=original,
    output_name='test_reconstruction'
)
```

### Batch Evaluation

```python
from utils.evaluation import AudioEvaluator

evaluator = AudioEvaluator()
metrics = evaluator.evaluate_batch([
    (original1, reconstructed1),
    (original2, reconstructed2)
])
```

## Testing and Validation

### Test Scripts

1. **`test_reconstruction.py`**
   - Basic functionality testing
   - Quality validation
   - Edge case handling

2. **`edge_case_validation.py`**
   - Robustness testing
   - Data corruption scenarios
   - Boundary condition validation

3. **`generate_test_outputs.py`**
   - Comprehensive sample generation
   - Performance benchmarking
   - Visual comparison tools

### Test Results Summary

- **Reconstruction Success Rate**: 100%
- **Edge Case Robustness**: 100% (28/28 tests passed)
- **Feature Type Coverage**: Raw, Spectral, MFCC
- **Quality Score Range**: 0.020 - 0.197 (varies by feature type)

## Performance Characteristics

### By Feature Type

| Feature Type | Avg Quality | Avg SNR (dB) | Best Use Case |
|--------------|-------------|--------------|---------------|
| Raw          | 0.040       | -0.4         | Time-domain analysis |
| Spectral     | 0.022       | -1.5         | Frequency analysis |
| MFCC         | 0.020       | -1.1         | Speech processing |

### Processing Speed

- **Raw features**: ~5.1s for 512-frame sequences
- **Spectral features**: ~1.7s for 173-frame sequences
- **MFCC features**: ~1.7s for 173-frame sequences

## Output Structure

```
test_outputs/
├── generated_samples/
│   ├── originals/           # Original audio files
│   ├── reconstructions/     # Reconstructed audio files
│   ├── plots/              # Comparison visualizations
│   └── comprehensive_test_report.json
├── audio/                  # Pipeline audio outputs
├── reports/               # Detailed evaluation reports
└── plots/                 # Quality assessment plots
```

## Configuration Options

### Reconstruction Parameters

- `frame_length`: 256, 512, 1024, 2048
- `hop_length`: frame_length // 2 (50% overlap recommended)
- `window`: 'hann' (recommended), 'hamming', 'blackman'
- `sr`: 22050 (default), 44100, 48000

### Quality Thresholds

- `quality_threshold`: 0.1 (default minimum quality)
- Low quality reconstructions are flagged for review

## Limitations and Considerations

1. **Phase Information**: Spectral reconstruction loses phase information
2. **MFCC Limitations**: Information loss in cepstral domain
3. **Quality Scores**: Current scores are relatively low, indicating room for algorithm improvement
4. **Memory Usage**: Large feature matrices may require batch processing

## Future Improvements

1. **Advanced Phase Reconstruction**: Implement iterative phase recovery
2. **Neural Reconstruction**: Deep learning-based reconstruction methods
3. **Perceptual Metrics**: Add PESQ, STOI for speech quality
4. **Real-time Processing**: Optimize for streaming applications
5. **Multi-channel Support**: Extend to stereo/multichannel audio

## References

- Griffin, D. & Lim, J. (1984). Signal estimation from modified short-time Fourier transform
- Rabiner, L. & Schafer, R. (2010). Theory and Applications of Digital Speech Processing
- Librosa Documentation: https://librosa.org/

---

*Documentation generated for Sprint 6: Signal Reconstruction & Evaluation*
*Last updated: January 2025*