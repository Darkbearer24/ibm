<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" class="logo" width="120"/>

# aight make a new updated md file for this

```markdown
# IBM Internship Multilingual Speech-Based Translation System  
**Project Plan & Updated Roadmap**  
_Last updated: August 2025_

---

## Overview

A modular end-to-end pipeline for translating speech between 10 Indian languages (plus English) using open-source tools and a speaker-diverse audio dataset. All components—preprocessing, feature extraction, modeling, evaluation, web UI, and experiment tracking—are engineered to be free, reproducible, and flexible for R&D or demonstration.

---

## 1. Project Objective

Build a speech translation pipeline that takes audio in Indian Language A and outputs translated speech (or optionally text) in Indian Language B, using only open-source libraries and the provided dataset, and delivering a working demo and evaluation-ready system.

---

## 2. Dataset

- Kaggle: Indian Languages Audio Dataset
- 10 languages (Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Punjabi, Tamil, Telugu, Urdu)
- ~5s mono .wav clips, 44.1kHz, unaligned, no transcript/translation

---

## 3. System Architecture

### Main Pipeline

```

Input Audio (Lang A)
↓
Preprocessing (denoise, normalize)
↓
Framing (20ms, 50% overlap) → Feature Matrix (frames x 441)
↓
Encoder-Decoder Model (LSTM, 441→100→441)
↓
Predicted Target Feature Matrix (Lang B)
↓
Overlap-Add Reconstruction
↓
Translated Speech Output (Lang B)

```

### Optionally:
```

Input Audio → ASR → MT → TTS → Output Speech (benchmark/comparison only)

```

---

## 4. Technology Stack

| Layer          | Tools                             |
|----------------|-----------------------------------|
| Dev/Notebooks  | Colab, Jupyter (for demos/dev)    |
| Audio Handling | librosa, scipy, torchaudio        |
| Modeling       | PyTorch                           |
| Features/Framing | numpy, custom utils             |
| Visualization  | matplotlib, seaborn               |
| UI             | Streamlit (web interface)         |
| Eval           | Custom MSE, SNR, listening tests  |
| Versioning     | Git, requirements.txt             |

---

## 5. Folder Structure

```

project-root/
│
├── data/
│   ├── raw/         \# Untouched audio files
│   ├── processed/   \# Cleaned, normalized
│   └── features/    \# Frame/feature numpy arrays
│
├── notebooks/
│   ├── 01_audio_cleaning.ipynb
│   ├── 02_feature_matrix_builder.ipynb
│   ├── 03_model_architecture.ipynb
│   ├── 04_training_preparation_and_gpu_export.ipynb
│   ├── 05_cpu_training_validation.ipynb
│   ├── 06_reconstruction_and_evaluation.ipynb
│   └── 07_interactive_demo.ipynb
│
├── models/
│   ├── encoder_decoder.py
│   ├── train.py
│   ├── enhanced_trainer.py
│
├── utils/
│   ├── denoise.py
│   ├── framing.py
│   ├── reconstruction.py
│   └── evaluation.py
│
├── outputs/
│   ├── reconstructed_audio/
│   ├── metrics/
│   ├── graphs/
│
├── app.py              \# Streamlit UI
├── requirements.txt    \# All dependencies
├── report/             \# Presentation/final report
│   └── final_report.pdf
└── README.md           \# Project docs (this file)

```

---

## 6. Sprints & Status

### **Sprint 1: Dataset Exploration & Preprocessing**  
- ✅ Dataset loading; waveform/spec visualization  
- ✅ Adaptive denoising ("no pass filter," spectral subtraction)  
- ✅ Normalization; save clean samples  

### **Sprint 2: Framing & Feature Extraction**  
- ✅ Frame to 20ms (10ms stride), 441-feature vectors  
- ✅ Save feature matrices for modeling  

### **Sprint 3: Model Architecture**  
- ✅ Encoder: 441→100 | Decoder: 100→441  
- ✅ LSTM layers; forward pass tested  
- ✅ MSE loss function  

### **Sprint 4: Training Pipeline Prep**  
- ✅ DataLoader, batch/feed, checkpointing  
- ✅ CPU/Colab training validation, export for GPU  

### **Sprint 5: Local CPU Training & Validation**  
- ✅ Model convergence on test data  
- ✅ Checkpoint logic, config templates  
- ✅ Loss reduction & metrics report  

### **Sprint 6: Signal Reconstruction & Evaluation** [CURRENT]  
**Goals:**  
- Overlap-add reconstruction script  
- MSE/SNR/perceptual evaluation utilities  
- Reconstruct and save test outputs  
- Edge case/robustness validation  

### **Sprint 7: System Integration & End-to-End Test**  
**Goals:**  
- Integrate preprocessor, model, recon, and UI  
- Confirm "audio in → translation → audio out" in local/web GUI  
- Document error flows, UI polish, user testing  

### **Sprint 8: Intensive GPU Training & Model Selection**  
**Goals:**  
- Full data GPU training at campus (multi-day)  
- Monitor/track best model & save artifacts  
- Reconstruct sample outputs for demo  

### **Sprint 9: Post-Training QA, Demo, & Final Reporting**  
**Goals:**  
- Evaluate model with untrained test data  
- Web UI demo with live translations  
- Compile results, metrics, and documentation  
- Prep project for delivery, showcase, or future deployment  

---

## 7. UI Layer (Streamlit)

**Features:**  
- Upload/play audio, language select, “Translate” button, playback/download output  
- Shows input/output waveforms, optional spectrograms  
- Warnings on error, file limits, success messages  
- Modular: connects to backend utils/models, extensible for new features

---

## 8. Evaluation & Metrics

- **Automated**: MSE, SNR between reconstructed and reference outputs  
- **Subjective**: Listening for naturalness/accuracy, user ratings  
- **Reporting**: Save scores, sample pairs, anecdotal feedback in outputs/metrics  
- **Ready for formal reporting and qualitative review** after full training

---

## 9. Usage

**Quick Start**
```

pip install -r requirements.txt
streamlit run app.py

```
- Choose input/output language, upload file, click "Translate"
- Download/play result, view waveforms, review stats

---

## 10. Notes & Future Directions

- UI and all backend modules are ready for main integration  
- Intensive model training is scheduled for campus GPUs; current models are on Bengali subset (can scale to all languages)  
- After sprint 9, project will be polished for presentation, possible open-source release, and real-world piloting  
- Future sprints can add: multi-speaker support, API batch modes, attention/transformer models, TTS for text-only translation

---

**For bugs/support:**  
- Check troubleshooting in UI  
- Confirm file types/length  
- For backend/model issues, raise as GitHub/README issue

---

**Contributors:**  
- [Names/roles as appropriate]  
- IBM Internship, Indian Language Technologies Project

---
```

