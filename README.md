# CoughNet — COVID-19 Detection from Cough Audio

A deep learning pipeline that screens for COVID-19 from cough recordings using Mel spectrogram features and a custom 4-block CNN ("CoughNet"). Built on the [COUGHVID dataset](https://zenodo.org/record/4498364).

> ⚠️ **Disclaimer:** This project is for academic and research purposes only. It is **not** a validated medical diagnostic tool and should not be used for clinical decision-making. See [Known Limitations](#known-limitations).

---

## Results

Evaluated on a held-out test set of 1,200 samples (600 COVID/symptomatic, 600 healthy), using a 0.30 decision threshold:

| Metric                       | Score  |
| ----------------------------- | ------ |
| Test Accuracy                 | 73.75% |
| AUC-ROC                       | 0.815  |
| COVID Precision                | 74.87% |
| COVID Sensitivity (Recall)     | 71.50% |
| Specificity (Healthy recall)   | 76.00% |
| COVID F1-score                 | 0.732  |

Full classification report:

```
              precision    recall  f1-score   support

 Healthy (0)     0.7273    0.7600    0.7433       600
   COVID (1)     0.7487    0.7150    0.7315       600

    accuracy                         0.7375      1200
   macro avg     0.7380    0.7375    0.7374      1200
weighted avg     0.7380    0.7375    0.7374      1200
```

**Threshold sweep** (decision threshold vs. precision/recall/F1 trade-off):

| Threshold | Precision | Recall | F1     |
| --------- | --------- | ------ | ------ |
| 0.20      | 0.6028    | 0.8650 | 0.7105 |
| 0.25      | 0.6769    | 0.7717 | 0.7212 |
| **0.30**  | **0.7487**| **0.7150** | **0.7315** |
| 0.35      | 0.7972    | 0.6683 | 0.7271 |
| 0.40      | 0.8472    | 0.6467 | 0.7335 |
| 0.45      | 0.8960    | 0.6317 | 0.7410 |
| 0.50      | 0.9248    | 0.6150 | 0.7387 |

The threshold that maximizes Youden's J statistic (sensitivity + specificity) is **0.72**, giving 60.0% sensitivity and 98.2% specificity — useful if the goal is to minimize false positives rather than screen broadly.

> **Note:** Lower thresholds (0.20–0.30) favor sensitivity and are recommended for screening use, where missing a positive case is costlier than a false alarm. Higher thresholds trade recall for precision.

---

## Project Structure

```
COVID-19-detection-using-cough-audio/
├── notebook/            # Data preprocessing, training, and evaluation notebooks (CoughNet pipeline)
├── utils/                # Helper modules (audio loading, segmentation, feature extraction)
├── audio_test/           # Sample cough audio files for local testing/inference
├── app.py                # Application entry point for running inference on new audio
├── test.py               # Script for testing/evaluating the trained model
├── requirements.txt       # Python dependencies
└── README.md
```

---

## Pipeline Overview

### 1. Data — COUGHVID

- **Source:** [COUGHVID](https://zenodo.org/record/4498364) public dataset — 16,224 raw recordings with metadata (`status`, `cough_detected`, `age`, `gender`, `respiratory_condition`, `fever_muscle_pain`).
- **Raw label distribution:** healthy = 12,479, symptomatic = 2,590, COVID-19 = 1,155.
- **Label merge:** `COVID-19` + `symptomatic` → **positive** class; `healthy` → **negative** class (COUGHVID has no RT-PCR confirmation, so COVID-19 and self-reported symptomatic cases are pooled).
- **Quality filtering:** kept only recordings with `cough_detected ≥ 0.8` and no missing values in `age`, `gender`, `respiratory_condition`, `fever_muscle_pain`.
- **After filtering:** 10,553 recordings — healthy = 8,236, symptomatic = 2,317.

### 2. Class Balancing

- Healthy class **downsampled** to 6,000 samples.
- Symptomatic class **augmented** from 2,317 → 6,000 samples (3,683 synthetic samples added), split roughly evenly between two augmentation techniques:
  - **Pitch shift** (waveform-level, applied before Mel extraction)
  - **SpecAugment** (spectrogram-level masking, applied after Mel extraction)
- **Final balanced dataset:** 12,000 samples (6,000 healthy / 6,000 symptomatic).

### 3. Audio Preprocessing

- Resampled to **22,050 Hz**, mono.
- Clipped/padded to **3 seconds** (66,150 samples).
- Custom energy-based cough segmentation (`segment_cough`) using an RMS-derived threshold with padding to isolate the cough event from silence/noise.

### 4. Feature Extraction — Mel Spectrogram

```
n_fft      = 512
hop_length = 512
n_mels     = 128
```

Each 3-second clip is converted to a Mel spectrogram, normalized to `[0, 1]`, and resized to a **128×128** image saved as a `.npy` array (and optionally a `.png` for visualization).

### 5. Model Architecture — CoughNet CNN

```
Input: (128, 128, 1)
│
├── Block 1: Conv2D(32)  × 2 → BatchNorm → MaxPool(2×2) → Dropout
├── Block 2: Conv2D(64)  × 2 → BatchNorm → MaxPool(2×2) → Dropout
├── Block 3: Conv2D(128) × 2 → BatchNorm → MaxPool(2×2) → Dropout
├── Block 4: Conv2D(256) × 2 → BatchNorm → MaxPool(2×2) → Dropout
│
├── GlobalAveragePooling2D
└── Dense(1, sigmoid)

Total params:     1,173,857 (~4.48 MB)
Trainable params: 1,172,897
```

### 6. Training

| Setting                   | Value                                       |
| -------------------------- | -------------------------------------------- |
| Optimizer                  | Adam (lr = 1e-4)                             |
| Loss                       | Binary cross-entropy                         |
| Batch size                 | 32                                            |
| Max epochs                 | 20                                            |
| Early stopping              | patience=5, monitor=`val_auc_roc`             |
| LR schedule                | ReduceLROnPlateau (factor=0.5, patience=3)    |
| Train / Val / Test split    | 9,600 / 1,200 / 1,200 (80% / 10% / 10%, stratified) |

Class split: Train = 4,800 COVID / 4,800 healthy · Val = 600 / 600 · Test = 600 / 600.

### 7. Evaluation

See [Results](#results) above. Model checkpointing tracks `val_auc_roc`; the best checkpoint (by validation AUC-ROC) is reloaded for final test-set evaluation and threshold analysis.

---

## Known Limitations

- **No RT-PCR confirmation:** COUGHVID has no lab-confirmed COVID diagnoses. Positive labels combine confirmed COVID-19 and self-reported "symptomatic" cases, introducing label noise.
- **Per-sample normalization:** Each Mel spectrogram is normalized independently, discarding absolute signal amplitude information.
- **Threshold sensitivity:** Performance varies meaningfully with the chosen decision threshold (see sweep table) — the "right" threshold depends on whether the use case prioritizes sensitivity (screening) or precision.
- **Dataset demographic bias:** COUGHVID skews toward younger, non-clinical, crowdsourced populations, which may limit generalization to broader or clinical populations.
- **Synthetic augmentation:** ~61% of the positive training samples are augmented (pitch-shifted or SpecAugment-masked) rather than original recordings, which may affect how well the model generalizes to unseen real coughs.

---

## Setup

```bash
pip install -r requirements.txt
```

Core dependencies:
- Python 3.9+
- TensorFlow 2.x
- librosa, soundfile
- scikit-learn, pandas, numpy
- matplotlib, tqdm
- GPU recommended for training (developed/tested on NVIDIA Tesla T4)

---

## References

- Orlandic et al. (2021). *The COUGHVID crowdsourcing dataset, a corpus for the study of large-scale cough analysis algorithms.* Scientific Data.
- Park et al. (2019). *SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition.* Interspeech.
- COUGHVID Dataset: <https://zenodo.org/record/4498364>

---

## Disclaimer

This project is for academic and research purposes only. It is not a medical diagnostic tool and should not be used for clinical decision-making.
