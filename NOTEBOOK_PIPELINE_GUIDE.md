# Complete Notebook Pipeline Guide

## Overview

This guide explains how to run all notebooks in the multilingual speech translation project as a complete pipeline. The project consists of 6 sequential notebooks that process audio data from raw input to final translated output.

## 🎯 Pipeline Architecture

```
Raw Audio → Preprocessing → Feature Extraction → Model Training → Validation → Reconstruction
     ↓            ↓              ↓               ↓            ↓            ↓
  Notebook 1   Notebook 2    Notebook 3     Notebook 4   Notebook 5   Notebook 6
```

## 📚 Notebook Sequence & Dependencies

### 1. **01_audio_cleaning.ipynb** - Audio Preprocessing
- **Purpose**: Clean and preprocess raw audio data
- **Input**: `data/raw/` (Original audio files from Kaggle dataset)
- **Output**: `data/processed/` (Cleaned, denoised, normalized audio)
- **Key Operations**:
  - Load audio files using librosa
  - Apply spectral subtraction denoising
  - RMS normalization to -20dB
  - Remove silence segments
  - Save processed audio files

### 2. **02_feature_matrix_builder.ipynb** - Feature Extraction
- **Purpose**: Convert audio to feature matrices for model input
- **Input**: `data/processed/` (Cleaned audio from Step 1)
- **Output**: `data/features/` (441-dimensional feature matrices)
- **Key Operations**:
  - Frame audio into 20ms windows with 10ms stride
  - Extract 441 features per frame
  - Create time-series feature matrices
  - Save feature matrices as .npy files

### 3. **03_model_architecture.ipynb** - Model Design
- **Purpose**: Design and test the encoder-decoder architecture
- **Input**: `models/encoder_decoder.py` (Model definition)
- **Output**: `test_checkpoints/test_checkpoint.pt` (Test model)
- **Key Operations**:
  - Build LSTM encoder-decoder model
  - Test forward pass with dummy data
  - Validate model architecture
  - Save test checkpoint

### 4. **04_training_preparation_and_gpu_export.ipynb** - Training Setup
- **Purpose**: Prepare training configuration and export for GPU
- **Input**: Feature matrices and model architecture
- **Output**: `outputs/sprint4_training_config.json` (Training config)
- **Key Operations**:
  - Create data loaders
  - Configure training parameters
  - Export training scripts
  - Generate configuration files

### 5. **05_cpu_training_validation.ipynb** - Model Training
- **Purpose**: Train the model and validate performance
- **Input**: Feature matrices and training configuration
- **Output**: `test_checkpoints/cpu_validation/` (Trained models)
- **Key Operations**:
  - Train encoder-decoder model
  - Monitor training loss
  - Save model checkpoints
  - Generate training curves

### 6. **06_reconstruction_and_evaluation.ipynb** - Audio Reconstruction
- **Purpose**: Reconstruct audio and evaluate translation quality
- **Input**: `test_checkpoints/cpu_validation/best_model.pt`
- **Output**: `outputs/reconstructed_audio/` (Translated audio)
- **Key Operations**:
  - Load trained model
  - Process input audio through pipeline
  - Reconstruct audio from features
  - Evaluate output quality

## 🚀 How to Run the Complete Pipeline

### Method 1: Automated Pipeline Runner (Recommended)

```bash
# Run the complete pipeline (all 6 stages)
python run_complete_pipeline.py

# Run specific stages
python run_complete_pipeline.py --stage 1    # Audio cleaning only
python run_complete_pipeline.py --start 2 --end 4  # Stages 2-4

# Show pipeline information
python run_complete_pipeline.py --info
```

### Method 2: Jupyter Lab Interactive (Current Setup)

1. **Open Jupyter Lab** (already running at http://localhost:8888)
2. **Navigate to notebooks/** folder
3. **Run notebooks in sequence**:
   - Open `01_audio_cleaning.ipynb` → Run All Cells
   - Open `02_feature_matrix_builder.ipynb` → Run All Cells
   - Open `03_model_architecture.ipynb` → Run All Cells
   - Open `04_training_preparation_and_gpu_export.ipynb` → Run All Cells
   - Open `05_cpu_training_validation.ipynb` → Run All Cells
   - Open `06_reconstruction_and_evaluation.ipynb` → Run All Cells

### Method 3: Command Line Execution

```bash
# Execute notebooks one by one using nbconvert
jupyter nbconvert --to notebook --execute notebooks/01_audio_cleaning.ipynb
jupyter nbconvert --to notebook --execute notebooks/02_feature_matrix_builder.ipynb
jupyter nbconvert --to notebook --execute notebooks/03_model_architecture.ipynb
jupyter nbconvert --to notebook --execute notebooks/04_training_preparation_and_gpu_export.ipynb
jupyter nbconvert --to notebook --execute notebooks/05_cpu_training_validation.ipynb
jupyter nbconvert --to notebook --execute notebooks/06_reconstruction_and_evaluation.ipynb
```

## 📊 Data Flow Diagram

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Raw Audio     │───▶│  Processed Audio │───▶│ Feature Matrix  │
│ data/raw/       │    │ data/processed/  │    │ data/features/  │
│ (.wav files)    │    │ (cleaned .wav)   │    │ (.npy arrays)   │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
    Notebook 1              Notebook 2              Notebook 2
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│ Model Architecture│   │ Training Config  │    │ Trained Model   │
│ models/         │    │ outputs/         │    │test_checkpoints/│
│ (encoder_decoder)│    │ (config.json)    │    │ (best_model.pt) │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
    Notebook 3              Notebook 4              Notebook 5
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐
│ Reconstructed   │
│ Audio Output    │
│ outputs/        │
│ reconstructed/  │
└─────────────────┘
         │
    Notebook 6
```

## ⚙️ Prerequisites

### Data Requirements
- **Raw Audio Data**: Place Indian Languages Audio Dataset in `data/raw/`
- **Directory Structure**: Ensure proper folder structure exists

### Software Requirements
- **Python 3.8+** with virtual environment activated
- **Jupyter Lab/Notebook** (already installed)
- **Required Libraries**: All dependencies from `requirements.txt`

### Hardware Requirements
- **Memory**: 8GB RAM recommended (4GB minimum)
- **Storage**: 5GB free space for data and outputs
- **Processing**: Multi-core CPU (GPU optional for faster training)

## 🔧 Troubleshooting

### Common Issues

1. **Missing Data Files**
   ```bash
   # Check if raw data exists
   ls data/raw/
   # If empty, download from Kaggle and extract
   ```

2. **Memory Errors**
   - Reduce batch size in training notebooks
   - Process smaller audio files first
   - Close other applications

3. **Import Errors**
   ```bash
   # Ensure virtual environment is activated
   pip install -r requirements.txt
   ```

4. **Notebook Kernel Issues**
   - Restart kernel: Kernel → Restart Kernel
   - Clear outputs: Cell → All Output → Clear

### Performance Tips

- **Sequential Execution**: Always run notebooks in order (1→2→3→4→5→6)
- **Check Outputs**: Verify each stage produces expected outputs before proceeding
- **Monitor Resources**: Watch memory usage during training (Notebook 5)
- **Save Progress**: Notebooks automatically save checkpoints

## 📈 Expected Outputs

After running the complete pipeline, you should have:

1. **Processed Audio**: Clean audio files in `data/processed/`
2. **Feature Matrices**: Numerical representations in `data/features/`
3. **Trained Model**: Best model checkpoint in `test_checkpoints/cpu_validation/`
4. **Reconstructed Audio**: Translated audio in `outputs/reconstructed_audio/`
5. **Visualizations**: Training curves and analysis plots in `outputs/feature_plots/`
6. **Reports**: Summary files and logs in `outputs/`

## 🎯 Integration with Web Interface

The trained model from this pipeline integrates with the Streamlit web interface:

- **Model Loading**: `app.py` automatically loads `test_checkpoints/cpu_validation/best_model.pt`
- **Processing Pipeline**: Web interface uses the same preprocessing functions
- **Real-time Translation**: Users can upload audio and get translations

## 📝 Next Steps

After completing the notebook pipeline:

1. **Test Web Interface**: Use the Streamlit app (http://localhost:8501) to test translations
2. **Experiment**: Try different audio files and language pairs
3. **Optimize**: Adjust model parameters for better performance
4. **Scale**: Consider GPU training for larger datasets

---

**Note**: The pipeline is designed to work with the existing project structure. Ensure all paths and dependencies are correctly set up before running.