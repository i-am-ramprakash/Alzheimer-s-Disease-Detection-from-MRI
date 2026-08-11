# Comprehensive Project Architecture, Structure, & Operational Guide

## 1. Overall Project Overview

This repository provides an educational, production-grade machine learning system for classifying standardized 2D brain MRI slices into four Alzheimer's disease cognitive progression stages:

1. **Non Demented** (Healthy control scans)
2. **Very Mild Demented** (Early subtle structural changes)
3. **Mild Demented** (Noticeable ventricular enlargement and temporal cortical atrophy)
4. **Moderate Demented** (Severe cortical atrophy and marked tissue volume loss)

### Core Objective & Benchmark Model
The system features a benchmark **Radial Basis Function Support Vector Machine (RBF-SVM)** trained on aligned $32 \times 32$ grayscale pixel features, achieving **99.2188% test accuracy** (1,270 / 1,280 correct test predictions).

For comparative learning, the repository also includes custom 2D Convolutional Neural Network (CNN) and MobileNetV2 transfer learning pipelines.

### Web Application & Interactive Workstation
The project includes a full-featured, zero-dependency interactive **Clinical Diagnostic Web Workspace** served via a Python REST API backend (`app.py`) and a Single Page Application (`static/`). It provides real-time model inference, interactive canvas MRI zooming/panning, spatial occlusion heatmap explainability, dataset exploration, decision score distributions, and PDF report exports.

> **Clinical Disclaimer:** Educational and research use only. This project does not provide medical diagnoses and must not be used for clinical health decisions.

---

## 2. Complete Repository Directory & File Structure

```text
Alzheimer-s-Disease-Detection-from-MRI/
├── app.py                      Python REST API server and static web host (http.server)
├── static/                     Interactive Web Workstation Frontend
│   ├── index.html              Single Page Application HTML5 markup (Mockup-matched layout)
│   ├── styles.css              Clean white & green clinical color palette CSS styling
│   └── app.js                  Client-side ES6 JS engine (Canvas, Heatmaps, API, State)
│
├── alzheimer_detection/        Core Python Package (ML Pipelines & Utilities)
│   ├── __init__.py             Package initialization & version metadata
│   ├── constants.py            Canonical class order keys, display labels, supported formats
│   ├── config.py               Type-safe dataclass configurations for neural training
│   ├── dataset.py              Directory discovery, alias mapping, auditing & TF loading
│   ├── prepare_dataset.py      Deterministic stratified train/test dataset splitter
│   ├── audit.py                CLI tool for dataset structural integrity checks
│   ├── classical.py            RBF-SVM feature extraction, training, & prediction pipeline
│   ├── model.py                Custom Keras CNN & MobileNetV2 architectures
│   ├── training.py             Custom CNN training workflow & callbacks
│   ├── transfer_training.py    Two-stage MobileNetV2 transfer learning pipeline
│   ├── prediction.py           Prediction module for Keras .h5 models
│   └── evaluation.py           Classification reports & confusion matrix PNG renderer
│
├── models/                     Pre-trained Model Artifacts
│   ├── alzheimer_svm.joblib    Bundled 99.22% RBF-SVM scikit-learn model
│   └── model_metadata.json     Model configuration, feature size, and label order
│
├── results/                    Published Held-Out Evaluation Artifacts
│   ├── evaluation.json         Held-out metrics, accuracy, and confusion matrix JSON
│   └── confusion_matrix.png    Visual confusion matrix plot for publication
│
├── data/                       Dataset Directory (Excluded from git tracking)
│   └── alzheimer_mri_clean/    Prepared train and test split directories
│       ├── train/              80% Stratified Training Split (5,120 images)
│       └── test/               20% Held-Out Testing Split (1,280 images)
│
├── docs/                       Comprehensive Learning & Architecture Documentation
│   ├── README.md               Learning Center index & suggested study plan
│   ├── PROJECT_STRUCTURE_AND_ARCHITECTURE.md (This document)
│   ├── PROJECT_WALKTHROUGH.md   In-depth code walkthrough & function explanations
│   ├── HANDS_ON_GUIDE.md        Step-by-step terminal execution guide
│   ├── METRICS_AND_LIMITATIONS.md Data leakage, image vs patient accuracy analysis
│   ├── GLOSSARY.md              Machine learning & medical terminology dictionary
│   └── EXPERIMENT_LOG_TEMPLATE.md Template for controlled experiment logs
│
├── tests/                      Automated Unit & Behavioral Tests
│   ├── test_classical.py       Tests for SVM feature extraction & prediction logic
│   ├── test_dataset.py         Tests for dataset discovery, auditing & class mapping
│   └── test_model.py           Tests for neural network tensor shapes & compiles
│
├── pyproject.toml              Package build configuration & metadata
├── requirements.txt            Pinned Python dependencies
├── MODEL_CARD.md               Formal Model Card detailing dataset, usage, & limitations
└── LICENSE                     MIT License
```

---

## 3. System Architecture & Data Flow

```text
[ Raw Input MRI Slice (.jpg / .png) ]
               │
               ▼
   [ Image Preprocessing ] 
   (Convert to Grayscale -> Resize to 32x32 -> Normalize to [0.0, 1.0])
               │
               ▼
     [ 1,024 Feature Vector ]
               │
               ▼
    [ StandardScaler Pipeline ] ──► (Standardizes features using training mean & std)
               │
               ▼
     [ RBF-SVM Classifier ] ────► (Computes decision boundary distances for 4 classes)
               │
               ├──────────────────────────────────────────┐
               ▼                                          ▼
   [ Softmax Decision Scores ]               [ Occlusion Sensitivity Heatmap ]
               │                             (Slides 4x4 mask, measures score drops)
               │                                          │
               └────────────────────┬─────────────────────┘
                                    ▼
                 [ Interactive Web UI / REST API Output ]
                 (Stage Diagnosis, Confidence %, Heatmap, PDF)
```

---

## 4. Technologies Used & Component Responsibilities

| Technology | Role & Function in Project |
| :--- | :--- |
| **Python 3.8+** | Core programming language for data loading, model inference, web server, and CLI tools. |
| **Scikit-Learn (`sklearn`)** | Provides the `SVC` (RBF-SVM classifier), `StandardScaler`, `make_pipeline`, classification reports, and confusion matrix computation. |
| **TensorFlow / Keras** | Used in benchmark neural network experiments (`model.py`, `training.py`, `transfer_training.py`) to build custom CNNs and fine-tune MobileNetV2. |
| **Pillow (`PIL`) & OpenCV** | Performs image opening, bilinear downsampling ($32 \times 32$), channel conversions, and thermal heatmap colormap blending. |
| **Joblib** | Serializes the trained scikit-learn pipeline into compressed `.joblib` files (`models/alzheimer_svm.joblib`). |
| **Python `http.server` & `json`** | Zero-dependency REST API web server (`app.py`) providing `/api/predict`, `/api/samples`, and `/api/metrics` endpoints. |
| **HTML5, CSS3, ES6 JavaScript** | Single Page Application frontend (`static/`) implementing canvas pan/zoom, heatmap toggles, progress bars, and modal windows. |
| **Pytest** | Automated unit test runner (`pytest`) verifying dataset auditing, feature extraction, and model inference contracts. |

---

## 5. How-To & Operational Guidance

### 5.1 Environment Setup
Create a virtual environment and install dependencies:

```powershell
# Create virtual environment
python -m venv .venv

# Activate environment (Windows PowerShell)
.venv\Scripts\Activate.ps1

# Upgrade pip and install pinned requirements
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

### 5.2 Running the Interactive Web Application
Launch the REST API server and web workstation:

```powershell
python app.py
```

Open your browser and navigate to: **[http://localhost:5000](http://localhost:5000)**

**Web App Features Available:**
- **Interactive MRI Canvas:** Zoom ($50\% - 300\%$), Pan (mouse drag), Rotate ($90^\circ$), Reset View.
- **Explainability View Modes:** `Original MRI`, `Heatmap`, `Overlay` (sensitivity heatmap blended on scan).
- **1-Click Preset Testing:** Select sample scans from all 4 categories.
- **Drag & Drop Upload:** Test custom `.jpg` or `.png` MRI files with instant validation.
- **Confidence Quality Rating:** `High Confidence`, `Moderate Confidence`, `Low Confidence`.
- **Interactive Confusion Matrix:** Click matrix cells to view exact counts and percentages.
- **PDF Report Export:** Click *Download PDF* to export a printable analysis report.

---

### 5.3 Single Image Prediction (CLI)
Run inference on a single image via command line using the bundled SVM model:

```powershell
python -m alzheimer_detection.classical predict `
  --model models\alzheimer_svm.joblib `
  --image "data\alzheimer_mri_clean\test\NonDemented\non_1001.jpg"
```

---

### 5.4 Data Preparation & Auditing (CLI)
If you have downloaded the 4-class Kaggle Alzheimer dataset, organize it as:

```text
source/
├── MildDemented/
├── ModerateDemented/
├── NonDemented/
└── VeryMildDemented/
```

Prepare a deterministic 80/20 stratified split:

```powershell
python -m alzheimer_detection.prepare_dataset `
  --source-dir "C:\path\to\source" `
  --output-dir data\alzheimer_mri_clean `
  --test-fraction 0.20 `
  --seed 42
```

Audit dataset integrity:

```powershell
python -m alzheimer_detection.audit `
  --data-dir data\alzheimer_mri_clean `
  --verify-images
```

---

### 5.5 Retraining the Recommended RBF-SVM Model (CLI)
Train an SVM model on your prepared data:

```powershell
python -m alzheimer_detection.classical train `
  --data-dir data\alzheimer_mri_clean `
  --output-dir artifacts\svm-01 `
  --c 10.0
```

The command trains on `train/`, evaluates once on `test/`, and writes:
- `best_model.joblib`: Trained pipeline.
- `model_metadata.json`: Feature size and class mapping.
- `evaluation.json`: Held-out metrics.
- `confusion_matrix.png`: Error distribution heatmap plot.

---

### 5.6 Running Automated Test Suite
Run unit tests to verify code logic:

```powershell
python -m pytest
```
