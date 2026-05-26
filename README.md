# CoughNet — COVID-19 Detection from Cough Audio

A deep learning pipeline that classifies COVID-19 from cough recordings using Mel spectrogram features and a custom 4-block CNN. Built on the [COUGHVID dataset](https://zenodo.org/record/4498364).

---

## Results

| Metric | Score |
|---|---|
| Test Accuracy | 73.75% |
| AUC-ROC | 0.765 |
| COVID Precision | 85.7% |
| Specificity (Healthy recall) | 90.5% |
| COVID Sensitivity (Recall) | 57.0% |

> **Note:** Results use a 0.5 decision threshold. Lowering to 0.3–0.35 increases COVID sensitivity above ~75% at the cost of some precision — recommended for screening use.

---

## Project Structure

```
coughnet/
├── Preprcessing_Feature_extraction_COVID.ipynb   # Audio loading, silence removal, mel extraction, augmentation
├── Copy_of_Not_augmented.ipynb                   # Baseline CNN (no augmentation)
├── Augmented_mlflow_autolog.ipynb                # Augmented CNN with MLflow tracking
└── README.md
```

---

## Pipeline Overview

### 1. Data — COUGHVID

- **Source:** COUGHVID public dataset (16,224 recordings)
- **Labels:** `COVID-19` + `symptomatic` → positive class; `healthy` → negative class
- **Filtering:** retained only recordings with `cough_detected ≥ 0.5` and no missing medical fields
- **Final distribution after filtering:** ~3,700 positive (COVID/symptomatic), ~12,000 healthy

### 2. Preprocessing

- Resampled all audio to **22,050 Hz** (mono)
- Silence removed using `unsilence` library
- Clipped or zero-padded to **3 seconds** (66,150 samples)
- Near-silent recordings (`max amplitude < 0.01`) discarded

### 3. Feature Extraction — Mel Spectrogram

```
n_fft      = 512
hop_length = 512
n_mels     = 128
```

Pipeline per sample:
1. `librosa.feature.melspectrogram` → shape `(128, T)`
2. Convert to dB scale: `librosa.power_to_db(mel, ref=np.max)`
3. Resize to `128×128` via bilinear interpolation (PIL)
4. Normalize to `[0, 1]` per sample

### 4. Data Augmentation (positive class only)

Two augmentation strategies applied exclusively to COVID/symptomatic samples to reach a target of **6,000 samples per class**:

| Technique | Applied at | Parameters |
|---|---|---|
| Pitch shift | Waveform level | n_steps = −4 semitones |
| SpecAugment | Spectrogram level | F=30 mel bins, T=30 time frames |

Healthy class downsampled to 6,000; no augmentation applied.

### 5. Model Architecture — CoughNet CNN

```
Input: (128, 128, 1)
│
├── Block 1: Conv2D(32) × 2 → BatchNorm → MaxPool(2×2) → Dropout(0.25)
├── Block 2: Conv2D(64) × 2 → BatchNorm → MaxPool(2×2) → Dropout(0.25)
├── Block 3: Conv2D(128) × 2 → BatchNorm → MaxPool(2×2) → Dropout(0.30)
├── Block 4: Conv2D(256) × 2 → BatchNorm → MaxPool(2×2) → Dropout(0.30)
│
├── GlobalAveragePooling2D
└── Dense(1, sigmoid)

Total parameters: 1,173,857 (~4.5 MB)
```

### 6. Training

| Setting | Value |
|---|---|
| Optimizer | Adam (lr = 1e-4) |
| Loss | Binary cross-entropy |
| Batch size | 32 |
| Max epochs | 20 |
| Early stopping | patience=5, monitor=val_auc_roc |
| LR schedule | ReduceLROnPlateau (factor=0.5, patience=3) |
| Train / Val / Test split | 80% / 10% / 10% (stratified) |

### 7. Evaluation

Evaluated on a held-out test set of 1,200 samples (600 COVID, 600 Healthy):

```
              precision    recall  f1-score   support

 Healthy (0)     0.6779    0.9050    0.7752       600
   COVID (1)     0.8571    0.5700    0.6847       600

    accuracy                         0.7375      1200
   macro avg     0.7675    0.7375    0.7299      1200
```

---

## Known Limitations

- **No RT-PCR confirmation:** The COUGHVID dataset does not include lab-confirmed COVID diagnoses. Positive labels include self-reported symptomatic cases, which introduces label noise.
- **Per-sample normalization:** Each spectrogram is normalized independently, meaning signal amplitude information is discarded. Global normalization fitted on the training set would be more rigorous.
- **Sensitivity at 0.5 threshold:** 57% sensitivity means ~43% of COVID cases are missed. For a real screening tool, a lower decision threshold (0.3–0.35) should be used.
- **Dataset demographic bias:** COUGHVID skews toward younger, non-clinical populations, which may limit generalization.

---

## Setup

```bash
pip install librosa soundfile tensorflow scikit-learn matplotlib tqdm unsilence
```

**Requirements:**
- Python 3.9+
- TensorFlow 2.x
- librosa 0.10+
- GPU recommended for training (tested on NVIDIA T4/P100)

---

## References

- Orlandic et al. (2021). *The COUGHVID crowdsourcing dataset, a corpus for the study of large-scale cough analysis algorithms.* Scientific Data.
- Park et al. (2019). *SpecAugment: A Simple Data Augmentation Method for Automatic Speech Recognition.* Interspeech.
- COUGHVID Dataset: https://zenodo.org/record/4498364

---

## Disclaimer

This project is for academic and research purposes only. It is not a medical diagnostic tool and should not be used for clinical decision-making.
