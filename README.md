# Multilingual Speech-Based Translation System for Indian Languages

## Project Overview

This project implements a speech translation pipeline that takes audio input in one Indian language and produces translated speech or text in another Indian language, using only open-source tools and freely available models.

## Dataset

- **Source**: Kaggle – Indian Languages Audio Dataset
- **Content**: ~5-second audio clips in 10 Indian languages
- **Format**: Mono .wav audio, 44.1 kHz sampling rate
- **Note**: No transcripts or paired translations provided

## System Architecture

**Primary Pipeline**:
```
Speech Input (Lang A) → Preprocessing → Framing → Feature Matrix Generation → 
Encoder-Decoder Model → Target Feature Matrix (Lang B) → Audio Reconstruction → 
Translated Speech Output (Lang B)
```

**Alternative Pipeline** (optional):
```
Speech Input (Lang A) → ASR → NMT → TTS → Speech Output (Lang B)
```

## Tech Stack

- **Development Environment**: Google Colab + Google Drive
- **Audio Processing**: librosa, scipy, torchaudio
- **Machine Learning**: PyTorch (primary) or TensorFlow
- **Model Architecture**: LSTM/GRU Encoder-Decoder
- **Visualization**: matplotlib, seaborn
- **Evaluation**: MSE Loss, SNR, manual listening

## Project Structure

```
project-root/
├── data/
│   ├── raw/                    # Original audio files
│   └── processed/              # Cleaned and preprocessed audio
├── notebooks/
│   ├── 01_audio_cleaning.ipynb
│   ├── 02_feature_matrix_builder.ipynb
│   ├── 03_model_architecture.ipynb
│   ├── 04_model_export_and_test.ipynb
│   ├── 05_audio_reconstruction.ipynb
│   └── 06_demo_and_evaluation.ipynb
├── models/
│   ├── encoder_decoder.py      # Model architecture
│   └── train.py               # Training script
├── outputs/
│   ├── reconstructed_audio/    # Generated audio files
│   └── graphs/                # Visualizations and plots
├── utils/
│   ├── framing.py             # Audio framing utilities
│   └── denoise.py             # Denoising functions
├── requirements.txt
├── README.md
└── report/
    └── final_report.pdf
```

## Setup Instructions

### 1. Environment Setup

```bash
# Install required packages
pip install -r requirements.txt

# For Jupyter notebook support
jupyter notebook
```

### 2. Data Preparation

1. Download the Indian Languages Audio Dataset from Kaggle
2. Place raw audio files in `data/raw/`
3. Run the preprocessing pipeline (Sprint 1)

## Sprint Plan (12 Days)

### Sprint 1: Dataset Exploration & Preprocessing (Day 1)
- Load and explore the dataset
- Visualize audio (waveforms, spectrograms)
- Apply denoising (adaptive filter)
- Normalize and save clean samples

### Sprint 2: Framing and Feature Extraction (Day 2)
- Frame audio: 20ms length, 10ms stride
- Generate 2D matrices: (frames × 441 features)
- Save framed features for model input

### Sprint 3: Model Architecture Design (Day 3)
- Build encoder-decoder model (441 → 100D → 441D)
- Define MSE loss function
- Test model on dummy data

### Sprint 4: Model Training Prep & Export (Day 4)
- Prepare training script and dataloader
- Export code for GPU training
- Save training-ready code

### Sprint 5: Campus GPU Training & Checkpoints (Days 5–8)
- Train model on full dataset
- Monitor and log loss
- Save model checkpoints

### Sprint 6: Signal Reconstruction & Evaluation (Day 9)
- Reconstruct audio using overlap-add
- Compare predicted vs. original signal
- Evaluate output quality

### Sprint 7: Demo, Visualization & Report (Days 10–12)
- Build demo for 3 language pairs
- Visualize pipeline and results
- Compile project report

## Model Architecture

- **Encoder**: 441-dimensional input → 100-dimensional latent space
- **Decoder**: 100-dimensional latent space → 441-dimensional output
- **Architecture**: LSTM or GRU based
- **Loss Function**: Mean Squared Error (MSE)

## Audio Processing Pipeline

1. **Preprocessing**: Denoising with adaptive filters
2. **Framing**: 20ms windows with 10ms stride
3. **Feature Extraction**: Convert to 441-feature vectors per frame
4. **Model Processing**: Encoder-decoder transformation
5. **Reconstruction**: Overlap-add method for audio synthesis

## Evaluation Metrics

- **Signal-to-Noise Ratio (SNR)**
- **Mean Squared Error (MSE)**
- **Manual listening tests**
- **Spectrogram comparison**

## Cost Considerations

- ✅ All tools and libraries are free and open-source
- ✅ No paid APIs required (Google, Azure, AWS)
- ✅ Google Colab free tier is sufficient
- ✅ Campus GPU resources for training

## Getting Started

1. **Clone/Download** this project
2. **Install dependencies**: `pip install -r requirements.txt`
3. **Download dataset** and place in `data/raw/`
4. **Start with Sprint 1**: Open `notebooks/01_audio_cleaning.ipynb`
5. **Follow the sprint plan** sequentially

## License

This project uses only open-source tools and libraries. Please ensure compliance with individual library licenses.

## Contributing

This is an IBM internship project. For questions or contributions, please follow your organization's guidelines.

---

**Next Steps**: 
- Create `requirements.txt` ✅
- Generate architecture diagrams
- Create dummy input/output for testing pipeline
- Begin Sprint 1: Dataset exploration