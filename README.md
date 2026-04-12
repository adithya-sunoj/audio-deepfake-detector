# Audio Deepfake Detection: Wav2Vec2 + Vocal Physics

Most AI audio deepfake detectors are great at catching pixel-perfect digital artifacts in clean audio. But what happens when that audio is compressed into an MP3 or sent over a bad phone connection? The artifacts get destroyed, and the AI fails.

This project tackles that problem by fusing massive **Self-Supervised Learning (SSL)** embeddings with **Biological Vocal Features** (the actual physics of the human voice) to create a detector that survives real-world audio degradation. 

---

## How It Works

The system processes **over 600,000 audio files** from the ASVspoof datasets, extracting exactly **1,550 features** from every single file:
* **1,536 SSL Features:** Extracted using a pre-trained Wav2Vec2 foundation model. 
* **14 Biological Features:** Classical speech physics that are immune to MP3 compression.
  * **Pitch:** `meanF0`, `stdevF0`, `f0_cov`
  * **Noise:** `hnr` (Harmonics-to-Noise Ratio)
  * **Jitter & Shimmer:** 10 micro-fluctuations in the vocal cords.

We push all 1,550 features into a custom PyTorch Multi-Layer Perceptron (MLP). To stop the massive Wav2Vec2 vector from completely overshadowing the 14 bio features during training, we use strict `StandardScaler` normalization and an aggressive `Dropout(0.4)` layer. 

---

## Tech Stack & Engineering Highlights
* **PyTorch:** Custom neural network architecture and streaming inference.
* **Pandas & Scikit-learn:** Feature engineering, scaling, and preprocessing.
* **RAM-Safe Dataloading:** Loading a 600,000-row x 1,550-column `float64` dataframe will instantly nuke 16GB of system RAM. We built a custom PyTorch `DataLoader` (with `num_workers=0` for Windows multiprocessing safety) to stream the CSV row-by-row directly from the SSD, keeping the memory footprint completely flat during massive evaluation loops.

## How to Run

**1. Train the Fusion Model**
This will train the MLP on the extracted features and save the best weights (`fusion_academic_mlp_weights.pt`) and the scaler (`fusion_academic_scaler.pkl`).
```bash
python train_fusion.py
```

**2. Evaluate the Model**
This script merges the SSL and Bio CSV datasets, applies the scaler, and runs the streaming inference to calculate the Equal Error Rate (EER) across the evaluation sets.
```bash
python evaluate_fusion_df.py
```
