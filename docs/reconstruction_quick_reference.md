# Reconstruction System Quick Reference

## Quick Start

### 1. Basic Reconstruction

```python
from utils.reconstruction import reconstruct_audio_overlap_add
import numpy as np

# Your feature matrix (n_frames, n_features)
features = np.random.randn(100, 512)  # Example: 100 frames of 512 features

# Reconstruct audio
audio = reconstruct_audio_overlap_add(
    features, 
    feature_type='raw',  # 'raw', 'spectral', or 'mfcc'
    sr=22050
)
```

### 2. Pipeline Processing (Recommended)

```python
from utils.reconstruction_pipeline import ReconstructionPipeline

# Initialize pipeline
pipeline = ReconstructionPipeline(output_dir='my_results')

# Process single feature matrix
result = pipeline.process_single(
    feature_matrix=features,
    original_audio=original_audio,  # Optional: for quality evaluation
    output_name='my_reconstruction',
    feature_type='raw'
)

# Access results
reconstructed_audio = result['reconstructed_audio']
quality_score = result['quality_score']
audio_path = result['audio_path']
```

### 3. Batch Processing

```python
# Process multiple feature matrices
feature_list = [features1, features2, features3]
original_list = [audio1, audio2, audio3]  # Optional

results = pipeline.process_batch(
    feature_matrices=feature_list,
    original_audios=original_list,
    output_names=['recon1', 'recon2', 'recon3'],
    feature_types=['raw', 'spectral', 'mfcc']
)
```

## Feature Types

| Type | Input Shape | Description | Use Case |
|------|-------------|-------------|----------|
| `'raw'` | (n_frames, frame_length) | Direct audio frames | Time-domain analysis |
| `'spectral'` | (n_frames, n_fft//2+1) | STFT magnitude | Frequency analysis |
| `'mfcc'` | (n_frames, n_mfcc) | MFCC coefficients | Speech processing |

## Common Parameters

```python
# Standard configuration
params = {
    'sr': 22050,           # Sample rate
    'frame_length': 512,   # Frame size
    'hop_length': 256,     # Hop size (50% overlap)
    'window': 'hann',      # Window function
    'feature_type': 'raw'  # Feature type
}
```

## Quality Evaluation

```python
from utils.evaluation import AudioEvaluator

evaluator = AudioEvaluator()
metrics = evaluator.evaluate_reconstruction(
    original_audio, 
    reconstructed_audio, 
    sample_rate=22050
)

# Key metrics
print(f"SNR: {metrics['snr_db']:.1f} dB")
print(f"Correlation: {metrics['correlation']:.3f}")
print(f"Quality Score: {metrics['quality_score']:.3f}")
```

## File I/O

```python
import soundfile as sf

# Save reconstructed audio
sf.write('output.wav', reconstructed_audio, samplerate=22050)

# Load audio for comparison
original, sr = sf.read('input.wav')
```

## Error Handling

```python
try:
    result = pipeline.process_single(features, feature_type='raw')
    if result['quality_score'] < 0.1:
        print("Warning: Low quality reconstruction")
except Exception as e:
    print(f"Reconstruction failed: {e}")
```

## Testing Your Setup

```python
# Quick test with synthetic data
import numpy as np
from utils.reconstruction_pipeline import ReconstructionPipeline

# Generate test features
test_features = np.random.randn(50, 512)

# Test reconstruction
pipeline = ReconstructionPipeline()
result = pipeline.process_single(test_features, output_name='test')

print(f"✓ Reconstruction successful: {result['output_duration']:.2f}s audio")
```

## Troubleshooting

### Common Issues

1. **Empty output**: Check feature matrix shape and type
2. **Low quality**: Try different feature types or parameters
3. **Memory errors**: Use batch processing for large datasets
4. **File not found**: Ensure output directory exists

### Debug Mode

```python
# Enable verbose output
pipeline = ReconstructionPipeline(output_dir='debug', verbose=True)
```

## Performance Tips

1. **Use appropriate feature types**: Raw for time-domain, spectral for frequency
2. **Batch processing**: More efficient for multiple files
3. **Adjust frame parameters**: Larger frames = better frequency resolution
4. **Quality thresholds**: Set minimum acceptable quality scores

---

*Quick Reference for Signal Reconstruction System*