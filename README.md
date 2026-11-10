# Lip-to-Speech Synthesis for Dysarthric Individuals (UA-Speech)

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-0.10%2B-00C7B7.svg)](https://mediapipe.dev/)
[![License: Academic](https://img.shields.io/badge/License-Academic--NonCommercial-lightgrey.svg)](#license)

This repository contains the official codebase for the Master's Thesis (**Trabajo de Fin de Máster - TFM**) at **Universidad Autónoma de Madrid (UAM)**, within the *Master in Deep Learning for Audio and Visual Artificial Intelligence*.

The project addresses **silent visual-to-speech reconstruction (Lip-to-Speech)** tailored for dysarthric individuals. Using high-resolution visual sequences of lip dynamics, the pipeline reconstructs intelligible acoustic waveforms via end-to-end deep neural architectures.

---

## Project Motivation & Research Goals

Dysarthria is a motor speech disorder characterized by impaired muscle control, resulting in distorted speech production, breathiness, and reduced acoustic intelligibility. While standard Automatic Speech Recognition (ASR) struggles with pathological acoustic inputs, **visual articulatory features (lip and jaw dynamics) retain crucial spatio-temporal structure**.

### Key Objectives:
1. **Spatio-Temporal Visual Feature Extraction**: Extract normalized lip Region of Interest (ROI) bounding sequences using MediaPipe 3D Face Mesh topology.
2. **Acoustic-Visual Cross-Modal Alignment**: Map silent visual sequences onto high-resolution 80-band Mel-spectrogram representations.
3. **Neural Speech Resynthesis**: Synthesize high-fidelity time-domain speech waveforms conditioned on predicted Mel-spectrograms using a fine-tuned HiFi-GAN vocoder.
4. **Intelligibility Evaluation**: Benchmark performance across severe, moderate, and mild dysarthric speakers using objective phonetic and acoustic metrics.

---

## System Architecture

```text
  +-----------------------+
  | Silent Video Sequence | (MP4 / 29.97 fps)
  +-----------+-----------+
              |
              v
  +-----------------------+      +-------------------------------+
  |  MediaPipe Face Mesh  | ---> | Lip ROI Crop & Normalization  |
  +-----------------------+      +---------------+---------------+
                                                 |
                                                 v [Frames x C x H x W]
                                   +---------------------------+
                                   | 3D-CNN / ResNet Frontend  |
                                   +-------------+-------------+
                                                 |
                                                 v (Visual Embeddings)
                                   +---------------------------+
                                   | Temporal Conformer/Transf |
                                   +-------------+-------------+
                                                 |
                                                 v
  +-----------------------+        +---------------------------+
  | Target Mel-Spec (GT)  | <----> |  Predicted Mel-Spec Loss  |
  +-----------------------+        +-------------+-------------+
  (Ground Truth Audio)                           |
                                                 v
                                   +---------------------------+
                                   |   HiFi-GAN Neural Vocoder |
                                   +-------------+-------------+
                                                 |
                                                 v
                                   +---------------------------+
                                   |  Reconstructed Speech WAV |
                                   +---------------------------+
```

---

## Dataset Structure & Setup (UA-Speech)

The UA-Speech (University of Illinois) corpus is an isolated-word dysarthric speech database containing video recordings, synchronized multi-channel audio, and standardized metadata across control and cerebral palsy speakers.

### Data Split Overview
- **Total Subjects**: 15 speakers distributed across severity bands (Control, Mild, Moderate, Severe).
- **Acoustic Formats**: Original raw audio, peak-normalized audio, and spectral noise-reduced audio.
- **Visual Formats**: High-definition video streams centered on facial movement.

## Evaluation Metrics Matrix

The system performance is quantitatively benchmarked using three evaluation tiers:

| Metric Category | Metric Name | Target Scope | Ideal Range / Goal |
| :--- | :--- | :--- | :--- |
| **Spectral Fidelity** | Mel-L1 Loss | Distance between predicted and GT spectrograms | $\to 0.0$ |
| **Intelligibility** | STOI | Short-Time Objective Intelligibility | $0.0 - 1.0$ (Higher is better) |
| **Acoustic Quality** | PESQ | Perceptual Evaluation of Speech Quality | $1.0 - 4.5$ (Higher is better) |
| **ASR Accuracy** | WER | Word Error Rate evaluated using Whisper ASR | $\to 0.0\%$ |

---

##  Repository Structure

```text
ua-speech-lip-to-speech/
├── configs/              # YAML hyperparameter configurations
│   └── baseline.yaml     # Baseline experiment parameters
├── data/                 # Ignored by Git (Data root)
│   ├── raw/              # Raw unpacked UA-Speech TGZ folders
│   └── processed/        # Cached tensors (.pt) and NumPy arrays (.npy)
├── notebooks/            # Exploratory Data Analysis & visual inspections
│   └── 01_eda_landmarks.ipynb
├── src/                  # Core Python modules
│   ├── __init__.py
│   ├── dataset.py        # PyTorch Dataset and DataLoaders
│   ├── preprocessing.py  # MediaPipe extraction & alignment pipeline
│   └── models/           # Neural network implementations
│       ├── __init__.py
│       ├── lip_encoder.py # 3D-CNN/ResNet visual feature extractor
│       └── vocoder.py     # HiFi-GAN synthesis wrapper
├── utils/                # Helper functions and metric calculators
│   ├── __init__.py
│   ├── metrics.py        # STOI, PESQ, and WER scripts
│   └── visualization.py  # Spectrogram & landmark plotting
├── .gitignore            # Strict rule-set to prevent data leakage to Git
├── evaluate.py           # Evaluation script on test dataset
├── README.md             # Project documentation
├── requirements.txt      # Python dependencies
└── train.py              # Main model training loop
```

---

##  Author & Academic Attribution

- **Author**: Rafael Carrillo Arroyo
- **Degree**: Master's Degree in Deep Learning for Audio and Visual AI
- **Institution**: Universidad Autónoma de Madrid (UAM), Spain
