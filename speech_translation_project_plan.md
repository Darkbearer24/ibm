## IBM Internship Language Translation Project Plan

### Project Title:

Multilingual Speech-Based Translation System for Indian Languages

---

### Project Objective:

Build a speech translation pipeline that takes audio input in one Indian language and produces translated speech or text in another Indian language, using only open-source tools, freely available models, and the provided dataset.

---

### Dataset

**Source**: Kaggle – Indian Languages Audio Dataset\
**Details**:

- \~5-second audio clips in 10 Indian languages
- No transcripts or paired translations
- Mono .wav audio
- Sampling Rate: 44.1 kHz

---

### System Architecture Overview

**Type**: Modular pipeline\
**Flow**: Speech Input (Lang A) → Preprocessing → Framing → Feature Matrix Generation → Encoder-Decoder Model → Target Feature Matrix (Lang B) → Audio Reconstruction → Translated Speech Output (Lang B)

Alternative (optional): Speech Input (Lang A) → ASR → NMT → TTS → Speech Output (Lang B)

---

### Tech Stack

| Component       | Tool/Library                       |
| --------------- | ---------------------------------- |
| Dev Environment | Google Colab + Google Drive        |
| Audio Handling  | librosa, scipy, torchaudio         |
| Modeling        | PyTorch or TensorFlow              |
| Encoder-Decoder | LSTM or GRU                        |
| Visualization   | matplotlib, seaborn                |
| Evaluation      | MSE Loss, SNR, or manual listening |
| Deployment/Demo | Colab + .wav downloads             |

---

## Project Plan & Sprints

### Sprint 1: Dataset Exploration & Preprocessing (Day 1)

**Goals:**

- Load and explore the dataset
- Visualize audio (waveforms, spectrograms)
- Apply no-pass denoising (adaptive filter)
- Normalize and save clean samples

**Deliverables:**

- Audio preprocessor function
- Comparison plots (raw vs. denoised)
- Notebook: `01_audio_cleaning.ipynb`

### Sprint 2: Framing and Feature Extraction (Day 2)

**Goals:**

- Frame audio: 20ms length, 10ms stride
- Generate 2D matrices: (frames × 441 features)
- Save framed features for model input

**Deliverables:**

- Framing module
- Audio-to-matrix conversion
- Notebook: `02_feature_matrix_builder.ipynb`

### Sprint 3: Model Architecture Design (Day 3)

**Goals:**

- Build encoder-decoder model
  - Encoder: 441 → 100D
  - Decoder: 100 → 441D
- Define MSE loss function
- Test model on dummy data

**Deliverables:**

- Model architecture script
- Forward pass with dummy input
- Notebook: `03_model_architecture.ipynb`

### Sprint 4: Model Training Prep & Export (Day 4) ✅ COMPLETED

**Goals:**

- Prepare training script and dataloader
- Export code for GPU training
- Set up training monitoring and checkpoints
- Test training pipeline on CPU

**Deliverables:**

- Training-ready code (`models/train.py`)
- Pipeline testing script (`test_training_pipeline.py`)
- Notebook: `04_training_preparation_and_gpu_export.ipynb`
- Sprint completion report

**Status**: All objectives completed. Training pipeline validated and ready for GPU export.

### Sprint 5: Local CPU Training & Validation (Days 5–6) ✅ COMPLETED

**Goals:**

- Train model on small dataset subset using CPU
- Validate complete training pipeline
- Test checkpoint saving/loading functionality
- Monitor training metrics and convergence

**Deliverables:**

- CPU training results and metrics
- Validated checkpoint system
- Training convergence analysis
- Notebook: `05_cpu_training_validation.ipynb`

**Status**: All objectives completed. Training pipeline fully validated on CPU with >80% loss reduction achieved. Checkpoint system tested and working. Advanced monitoring and configuration optimization implemented.

**Rationale**: Limited GPU access requires thorough CPU validation before expensive GPU time.

### Sprint 6: Signal Reconstruction & Evaluation Framework (Days 7–8)

**Goals:**

- Implement audio reconstruction using overlap-add
- Build evaluation framework (MSE, SNR, perceptual metrics)
- Test reconstruction on CPU-trained model
- Prepare evaluation pipeline for GPU-trained models

**Deliverables:**

- Reconstruction module (`utils/reconstruction.py`)
- Evaluation framework (`utils/evaluation.py`)
- Audio reconstruction samples
- Notebook: `06_reconstruction_and_evaluation.ipynb`

### Sprint 7: Demo Preparation & Documentation (Days 9–10)

**Goals:**

- Build interactive demo interface
- Create comprehensive documentation
- Prepare visualization tools
- Package complete system for GPU deployment

**Deliverables:**

- Demo interface: `07_interactive_demo.ipynb`
- Complete documentation and user guide
- GPU deployment package
- Project presentation materials

### Final Phase: Intensive GPU Training (When Campus GPUs Available)

**Goals:**

- Deploy validated pipeline to campus GPU environment
- Train model on full dataset with optimized hyperparameters
- Generate final model checkpoints
- Produce final translated audio samples

**Deliverables:**

- Production-ready trained model
- Final evaluation results
- Complete translated audio dataset
- Final project report and presentation

---

## Folder Structure Recommendation

```
project-root/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   ├── 01_audio_cleaning.ipynb
│   ├── ...
├── models/
│   ├── encoder_decoder.py
│   └── train.py
├── outputs/
│   ├── reconstructed_audio/
│   ├── graphs/
├── utils/
│   ├── framing.py
│   ├── denoise.py
├── requirements.txt
├── README.md
└── report/
    └── final_report.pdf
```

---

## License and Cost Considerations

- All tools/libraries are free and open-source
- Do not use paid APIs (Google, Azure, AWS)
- Colab Pro not required — free tier is sufficient

---

## Summary

You can build this entire system:

- Without paying for any tools
- Using Colab + local dev + campus GPU
- By following a sprint-based plan

Next Step Suggestions:

- Create `requirements.txt`
- Generate diagrams of architecture
- Create dummy input/output for testing pipeline

