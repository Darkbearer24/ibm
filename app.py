"""Multilingual Speech Translation Web Interface

A Streamlit-based web application for the multilingual speech translation system.
Supports 10 Indian languages + English with audio upload/recording capabilities.
"""

import streamlit as st
import torch
import torch.nn as nn
import numpy as np
import librosa
import soundfile as sf
import io
import os
import time
from pathlib import Path
import tempfile
from datetime import datetime
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Import project modules
from models.encoder_decoder import SpeechTranslationModel
from utils.denoise import preprocess_audio_complete
from utils.framing import create_feature_matrix_advanced
from utils.pipeline_orchestrator import PipelineOrchestrator
from utils.logging_config import get_logging_manager, LogLevel, LogCategory
from utils.error_handling import (
    SpeechTranslationError, AudioProcessingError, ModelInferenceError, 
    ReconstructionError, get_error_handler
)

# Configure page
st.set_page_config(
    page_title="Multilingual Speech Translation",
    page_icon="🗣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
SUPPORTED_LANGUAGES = [
    "Bengali", "Gujarati", "Hindi", "Kannada", "Malayalam", 
    "Marathi", "Punjabi", "Tamil", "Telugu", "Urdu", "English"
]

MODEL_CONFIG = {
    'input_dim': 441,
    'latent_dim': 100,
    'output_dim': 441,
    'encoder_hidden_dim': 256,
    'decoder_hidden_dim': 256,
    'num_layers': 2,
    'dropout': 0.2,
    'bidirectional': True
}

AUDIO_CONFIG = {
    'sr': 44100,
    'frame_length_ms': 20,
    'hop_length_ms': 10,
    'n_features': 441
}

@st.cache_resource
def load_pipeline_orchestrator():
    """Load the pipeline orchestrator with caching."""
    try:
        # Initialize logging and error handling
        logging_manager = get_logging_manager()
        error_handler = get_error_handler()
        
        logging_manager.log_structured(
            LogLevel.INFO, LogCategory.SYSTEM, "app_initialization",
            "Loading pipeline orchestrator"
        )
        
        # Initialize pipeline orchestrator
        orchestrator = PipelineOrchestrator(
            model_config=MODEL_CONFIG,
            audio_config=AUDIO_CONFIG
        )
        
        # Pipeline is automatically initialized in constructor
        st.success("✅ Pipeline orchestrator initialized successfully")
        logging_manager.log_structured(
            LogLevel.INFO, LogCategory.SYSTEM, "app_initialization",
            "Pipeline orchestrator loaded successfully"
        )
        
        return orchestrator
        
    except SpeechTranslationError as e:
        error_msg = f"Speech translation system error: {str(e)}"
        st.error(f"❌ {error_msg}")
        logging_manager.log_error("app_initialization", "load_orchestrator", str(e))
        return None
    except Exception as e:
        error_msg = f"Error initializing pipeline orchestrator: {str(e)}"
        st.error(f"❌ {error_msg}")
        logging_manager.log_error("app_initialization", "load_orchestrator", str(e))
        return None

# Note: Individual processing functions are now handled by PipelineOrchestrator
# Keeping these for backward compatibility and visualization purposes

def preprocess_audio(audio_data, sr):
    """Preprocess audio data for model input (legacy function)."""
    try:
        processed_audio = preprocess_audio_complete(
            audio_data, sr,
            denoise_method='spectral_subtraction',
            normalize_method='rms',
            target_db=-20,
            remove_silence_flag=True
        )
        
        feature_result = create_feature_matrix_advanced(
            processed_audio, sr,
            frame_length_ms=AUDIO_CONFIG['frame_length_ms'],
            hop_length_ms=AUDIO_CONFIG['hop_length_ms'],
            n_features=AUDIO_CONFIG['n_features'],
            include_spectral=False,
            include_mfcc=False
        )
        
        return feature_result['feature_matrix'], processed_audio
    except Exception as e:
        st.error(f"❌ Error preprocessing audio: {e}")
        return None, None

def create_waveform_plot(original_audio, processed_audio, sr):
    """Create interactive waveform comparison plot."""
    try:
        time_original = np.linspace(0, len(original_audio) / sr, len(original_audio))
        time_processed = np.linspace(0, len(processed_audio) / sr, len(processed_audio))
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Original Audio', 'Processed Audio'),
            vertical_spacing=0.1
        )
        
        # Original audio
        fig.add_trace(
            go.Scatter(x=time_original, y=original_audio, name='Original', line=dict(color='blue')),
            row=1, col=1
        )
        
        # Processed audio
        fig.add_trace(
            go.Scatter(x=time_processed, y=processed_audio, name='Processed', line=dict(color='red')),
            row=2, col=1
        )
        
        fig.update_layout(
            height=400,
            title_text="Audio Waveform Comparison",
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Time (s)")
        fig.update_yaxes(title_text="Amplitude")
        
        return fig
    except Exception as e:
        st.error(f"❌ Error creating waveform plot: {e}")
        return None

def create_spectrogram_plot(audio, sr, title="Spectrogram"):
    """Create spectrogram plot."""
    try:
        # Compute spectrogram
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        
        fig, ax = plt.subplots(figsize=(10, 4))
        img = librosa.display.specshow(D, y_axis='hz', x_axis='time', sr=sr, ax=ax)
        ax.set_title(title)
        plt.colorbar(img, ax=ax, format='%+2.0f dB')
        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"❌ Error creating spectrogram: {e}")
        return None

def main():
    """Main application function."""
    # Header
    st.title("🗣️ Multilingual Speech Translation System")
    st.markdown("""
    **Transform speech across 10 Indian languages + English using deep learning**
    
    Upload an audio file or record directly to experience real-time speech processing and translation.
    """)
    
    # Load pipeline orchestrator
    orchestrator = load_pipeline_orchestrator()
    if orchestrator is None:
        st.error("❌ Failed to initialize pipeline orchestrator. Please check the model files.")
        return
    
    # Sidebar configuration
    with st.sidebar:
        st.header("🎛️ Configuration")
        
        # Language selection
        source_lang = st.selectbox(
            "Source Language",
            SUPPORTED_LANGUAGES,
            index=2,  # Default to Hindi
            help="Select the language of the input audio"
        )
        
        target_lang = st.selectbox(
            "Target Language",
            SUPPORTED_LANGUAGES,
            index=10,  # Default to English
            help="Select the target language for translation"
        )
        
        if source_lang == target_lang:
            st.warning("⚠️ Source and target languages are the same")
        
        # Audio settings
        st.subheader("🎵 Audio Settings")
        show_visualizations = st.checkbox("Show Visualizations", value=True)
        audio_quality = st.selectbox(
            "Processing Quality",
            ["Standard", "High"],
            help="Higher quality takes more time"
        )
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Input Audio")
        
        # Audio input options
        input_method = st.radio(
            "Choose input method:",
            ["Upload File", "Record Audio"],
            horizontal=True
        )
        
        audio_data = None
        sr = None
        
        if input_method == "Upload File":
            uploaded_file = st.file_uploader(
                "Choose an audio file",
                type=['wav', 'mp3', 'flac', 'm4a'],
                help="Supported formats: WAV, MP3, FLAC, M4A"
            )
            
            if uploaded_file is not None:
                try:
                    # Load audio file
                    audio_data, sr = librosa.load(io.BytesIO(uploaded_file.read()), sr=AUDIO_CONFIG['sr'])
                    st.success(f"✅ Loaded audio: {len(audio_data)/sr:.2f}s duration")
                    
                    # Play original audio
                    st.audio(uploaded_file.getvalue(), format='audio/wav')
                    
                except Exception as e:
                    st.error(f"❌ Error loading audio file: {e}")
        
        elif input_method == "Record Audio":
            st.info("🎤 Audio recording feature requires additional setup. Please upload a file for now.")
            # Note: Streamlit doesn't have built-in audio recording.
            # You would need to use streamlit-webrtc or similar for this feature.
    
    with col2:
        st.header("📥 Translation Output")
        
        if audio_data is not None:
            # Translation button
            if st.button("🚀 Start Translation", type="primary", use_container_width=True):
                
                # Progress tracking
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                start_time = time.time()
                
                try:
                    # Initialize logging for this session
                    session_id = f"streamlit_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    logging_manager = get_logging_manager()
                    
                    logging_manager.log_structured(
                        LogLevel.INFO, LogCategory.PROCESSING, "audio_processing",
                        f"Starting audio processing session", session_id=session_id
                    )
                    
                    # Use Pipeline Orchestrator for end-to-end processing
                    status_text.text("🔄 Processing audio through pipeline...")
                    progress_bar.progress(10)
                    
                    # Create temporary file for orchestrator
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        sf.write(temp_file.name, audio_data, sr)
                        temp_audio_path = temp_file.name
                    
                    try:
                        # Run complete pipeline
                        status_text.text("🧠 Running complete pipeline...")
                        progress_bar.progress(50)
                        
                        result = orchestrator.process_audio_file(
                            temp_audio_path,
                            source_language=source_lang.lower(),
                            target_language=target_lang.lower(),
                            quality_level=audio_quality.lower()
                        )
                        
                        if not result['success']:
                            error_msg = result.get('error', 'Unknown error')
                            st.error(f"❌ Pipeline processing failed: {error_msg}")
                            logging_manager.log_error(
                                "audio_processing", "pipeline_execution", error_msg, session_id
                            )
                            return
                        
                        # Extract results
                        output_audio = result['reconstructed_audio']
                        feature_matrix = result['feature_matrix']
                        latent_representation = result['latent_features']
                        processed_audio = result['processed_audio']
                        
                        # Step 4: Complete
                        status_text.text("✅ Translation complete!")
                        progress_bar.progress(100)
                        
                        processing_time = time.time() - start_time
                        
                    finally:
                        # Clean up temporary file
                        if os.path.exists(temp_audio_path):
                            os.unlink(temp_audio_path)
                    
                    # Display results
                    st.success(f"🎉 Translation completed in {processing_time:.2f} seconds")
                    
                    # Audio player for output
                    st.subheader("🔊 Translated Audio")
                    
                    # Convert to audio bytes for playback
                    output_bytes = io.BytesIO()
                    sf.write(output_bytes, output_audio, sr, format='WAV')
                    output_bytes.seek(0)
                    
                    st.audio(output_bytes.getvalue(), format='audio/wav')
                    
                    # Download button
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"translated_{source_lang}_to_{target_lang}_{timestamp}.wav"
                    
                    st.download_button(
                        label="📥 Download Translated Audio",
                        data=output_bytes.getvalue(),
                        file_name=filename,
                        mime="audio/wav",
                        use_container_width=True
                    )
                    
                    # Statistics
                    with st.expander("📊 Processing Statistics"):
                        col_stat1, col_stat2, col_stat3 = st.columns(3)
                        
                        with col_stat1:
                            st.metric("Processing Time", f"{processing_time:.2f}s")
                        
                        with col_stat2:
                            st.metric("Input Duration", f"{len(audio_data)/sr:.2f}s")
                        
                        with col_stat3:
                            st.metric("Output Duration", f"{len(output_audio)/sr:.2f}s")
                        
                        st.write(f"**Feature Matrix Shape:** {feature_matrix.shape}")
                        st.write(f"**Latent Representation Shape:** {latent_representation.shape}")
                        st.write(f"**Compression Ratio:** {feature_matrix.shape[1] / latent_representation.shape[1]:.1f}:1")
                        
                        # Additional pipeline statistics
                        if 'pipeline_stats' in result:
                            stats = result['pipeline_stats']
                            st.write(f"**Preprocessing Time:** {stats.get('preprocessing_time', 'N/A')}")
                            st.write(f"**Inference Time:** {stats.get('inference_time', 'N/A')}")
                            st.write(f"**Reconstruction Time:** {stats.get('reconstruction_time', 'N/A')}")
                            st.write(f"**Quality Score:** {stats.get('quality_score', 'N/A')}")
                    
                    # Visualizations
                    if show_visualizations:
                        st.subheader("📈 Audio Analysis")
                        
                        # Waveform comparison
                        waveform_fig = create_waveform_plot(audio_data, output_audio, sr)
                        if waveform_fig:
                            st.plotly_chart(waveform_fig, use_container_width=True)
                        
                        # Spectrograms
                        col_spec1, col_spec2 = st.columns(2)
                        
                        with col_spec1:
                            st.subheader("Input Spectrogram")
                            input_spec = create_spectrogram_plot(audio_data, sr, "Input Audio")
                            if input_spec:
                                st.pyplot(input_spec)
                        
                        with col_spec2:
                            st.subheader("Output Spectrogram")
                            output_spec = create_spectrogram_plot(output_audio, sr, "Translated Audio")
                            if output_spec:
                                st.pyplot(output_spec)
                
                except AudioProcessingError as e:
                    error_msg = f"Audio processing error: {str(e)}"
                    st.error(f"❌ {error_msg}")
                    logging_manager.log_error("audio_processing", "audio_error", str(e), session_id)
                    progress_bar.empty()
                    status_text.empty()
                    
                except ModelInferenceError as e:
                    error_msg = f"Model inference error: {str(e)}"
                    st.error(f"❌ {error_msg}")
                    logging_manager.log_error("audio_processing", "model_error", str(e), session_id)
                    progress_bar.empty()
                    status_text.empty()
                    
                except ReconstructionError as e:
                    error_msg = f"Reconstruction error: {str(e)}"
                    st.error(f"❌ {error_msg}")
                    logging_manager.log_error("audio_processing", "reconstruction_error", str(e), session_id)
                    progress_bar.empty()
                    status_text.empty()
                    
                except SpeechTranslationError as e:
                    error_msg = f"Speech translation system error: {str(e)}"
                    st.error(f"❌ {error_msg}")
                    logging_manager.log_error("audio_processing", "system_error", str(e), session_id)
                    progress_bar.empty()
                    status_text.empty()
                    
                except Exception as e:
                    error_msg = f"Unexpected error during processing: {str(e)}"
                    st.error(f"❌ {error_msg}")
                    logging_manager.log_error("audio_processing", "unexpected_error", str(e), session_id)
                    progress_bar.empty()
                    status_text.empty()
                    st.exception(e)
        
        else:
            st.info("👆 Please upload an audio file to start translation")
    
    # Footer information
    st.markdown("---")
    with st.expander("ℹ️ About This System"):
        st.markdown("""
        ### Multilingual Speech Translation Pipeline
        
        This system uses a deep learning encoder-decoder architecture to process and translate speech across multiple Indian languages.
        
        **Key Features:**
        - 🎯 **10 Indian Languages + English**: Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Punjabi, Tamil, Telugu, Urdu
        - 🧠 **Neural Architecture**: LSTM-based encoder-decoder with 5.8M parameters
        - 🔊 **Audio Processing**: Advanced denoising and feature extraction
        - 📊 **Real-time Visualization**: Waveforms and spectrograms
        - 💾 **Download Support**: Save translated audio files
        
        **Technical Specifications:**
        - Sample Rate: 44.1 kHz
        - Frame Length: 20ms with 10ms overlap
        - Feature Dimension: 441 per frame
        - Latent Dimension: 100 (4.4:1 compression)
        
        **Note:** This is a demonstration system. The current model is trained primarily on Bengali data.
        For production use, models would be trained on all target languages.
        """)

if __name__ == "__main__":
    main()