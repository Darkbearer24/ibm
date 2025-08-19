# Multilingual Speech Translation Web Interface

A user-friendly Streamlit web application for the multilingual speech translation system supporting 10 Indian languages + English.

## Features

🎯 **Multi-language Support**: Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Punjabi, Tamil, Telugu, Urdu, English

🎵 **Audio Input Options**: 
- Upload audio files (WAV, MP3, FLAC, M4A)
- Future: Microphone recording support

🧠 **AI-Powered Processing**:
- Advanced audio denoising and preprocessing
- Neural encoder-decoder architecture (5.8M parameters)
- Real-time feature extraction and model inference

📊 **Visualizations**:
- Interactive waveform comparisons
- Input/output spectrograms
- Processing statistics and metrics

💾 **Export Capabilities**:
- Download translated audio files
- Timestamped file naming

## Quick Start

### 1. Install Dependencies

```bash
# Navigate to project directory
cd ibm

# Install required packages
pip install -r requirements.txt
```

### 2. Run the Web Interface

```bash
# Start the Streamlit application
streamlit run app.py
```

The application will open in your default web browser at `http://localhost:8501`

### 3. Using the Interface

1. **Select Languages**: Choose source and target languages from the sidebar
2. **Upload Audio**: Use the file uploader to select an audio file
3. **Start Translation**: Click the "Start Translation" button
4. **View Results**: Listen to the translated audio and download if needed
5. **Analyze**: Explore visualizations and processing statistics

## System Requirements

- **Python**: 3.8 or higher
- **Memory**: 4GB RAM minimum (8GB recommended)
- **Storage**: 2GB free space for dependencies
- **Audio**: Supported formats: WAV, MP3, FLAC, M4A

## Technical Architecture

### Model Specifications
- **Architecture**: LSTM-based encoder-decoder
- **Parameters**: 5,799,965 (5.8M)
- **Input**: 441-dimensional feature vectors per 20ms frame
- **Latent Space**: 100 dimensions (4.4:1 compression)
- **Output**: Reconstructed 441-dimensional features

### Audio Processing Pipeline
1. **Preprocessing**: Denoising, normalization, silence removal
2. **Framing**: 20ms windows with 10ms overlap (50% overlap)
3. **Feature Extraction**: 441 features per frame
4. **Model Inference**: Encoder-decoder processing
5. **Reconstruction**: Audio signal reconstruction

## Configuration Options

### Sidebar Settings
- **Source/Target Languages**: Select from 11 supported languages
- **Audio Quality**: Standard or High processing quality
- **Visualizations**: Toggle waveform and spectrogram displays

### Advanced Settings (Code Level)
```python
# Model configuration
MODEL_CONFIG = {
    'input_dim': 441,
    'latent_dim': 100,
    'encoder_hidden_dim': 256,
    'decoder_hidden_dim': 256,
    'num_layers': 2,
    'dropout': 0.2
}

# Audio processing configuration
AUDIO_CONFIG = {
    'sr': 44100,           # Sample rate
    'frame_length_ms': 20, # Frame length
    'hop_length_ms': 10,   # Hop length
    'n_features': 441      # Features per frame
}
```

## Troubleshooting

### Common Issues

**1. Model Loading Error**
```
❌ Error loading model: [error message]
```
**Solution**: The system will use a randomly initialized model for demonstration if trained weights aren't found.

**2. Audio Processing Error**
```
❌ Error preprocessing audio: [error message]
```
**Solution**: 
- Check audio file format (WAV, MP3, FLAC, M4A supported)
- Ensure file is not corrupted
- Try a different audio file

**3. Memory Issues**
```
Out of memory error
```
**Solution**:
- Use shorter audio files (< 30 seconds recommended)
- Close other applications
- Restart the Streamlit application

### Performance Tips

- **File Size**: Keep audio files under 10MB for optimal performance
- **Duration**: Files under 30 seconds process fastest
- **Quality**: Use "Standard" quality for faster processing
- **Browser**: Chrome or Firefox recommended for best experience

## Development Notes

### Current Limitations
- Model is primarily trained on Bengali data
- Audio recording requires additional setup (streamlit-webrtc)
- Translation quality depends on training data availability

### Future Enhancements
- Real-time microphone recording
- Multi-language model training
- Improved translation quality metrics
- Batch processing capabilities
- API endpoint integration

## File Structure

```
ibm/
├── app.py                 # Main Streamlit application
├── models/
│   ├── encoder_decoder.py # Neural network architecture
│   └── train.py          # Training utilities
├── utils/
│   ├── denoise.py        # Audio preprocessing
│   └── framing.py        # Feature extraction
├── test_checkpoints/     # Trained model weights
├── requirements.txt      # Python dependencies
└── README_WEB_UI.md     # This file
```

## Support

For technical issues or questions:
1. Check the troubleshooting section above
2. Verify all dependencies are installed correctly
3. Ensure audio files meet the supported format requirements
4. Check the Streamlit logs in the terminal for detailed error messages

## License

This project is part of the multilingual speech translation research system. Please refer to the main project documentation for licensing information.