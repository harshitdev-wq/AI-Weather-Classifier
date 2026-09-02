# AI Weather Classifier

<p align="center">
  <strong>AeroVision CV</strong><br>
  Computer vision for classifying weather scenes as <strong>Fog</strong>, <strong>Rain</strong>, or <strong>Snow</strong>.
</p>

<p align="center">
  <a href="https://github.com/harshitdev-wq/AI-Weather-Classifier"><img src="https://img.shields.io/github/stars/harshitdev-wq/AI-Weather-Classifier?style=flat-square" alt="GitHub stars"></a>
  <img src="https://img.shields.io/badge/model-ResNet18-111827?style=flat-square" alt="ResNet18">
  <img src="https://img.shields.io/badge/framework-PyTorch-ee4c2c?style=flat-square" alt="PyTorch">
  <img src="https://img.shields.io/badge/API-FastAPI-009688?style=flat-square" alt="FastAPI">
  <img src="https://img.shields.io/badge/frontend-HTML%20%2F%20CSS%20%2F%20JS-2563eb?style=flat-square" alt="Frontend">
  <img src="https://img.shields.io/badge/license-MIT-16a34a?style=flat-square" alt="MIT License">
</p>

## Overview

**AI Weather Classifier** is an end-to-end computer-vision project that takes a weather photograph and predicts whether the scene is dominated by **fog, rain, or snow**.

The project combines a fine-tuned **ResNet18** image classifier with a lightweight **FastAPI** inference service and a polished browser interface. The frontend is intentionally framework-free for easy deployment, while the model and API remain Python/PyTorch based.

The final model reached **93.29% validation accuracy** on a reproducible held-out split of the labelled training set.

> **Metric note:** 93.29% is validation accuracy. The separate external test folder contains unlabeled images, so the project does not claim an accuracy score for that test set.

## Highlights

- ResNet18 transfer learning with ImageNet initialization
- Three-class weather recognition: **Fog / Rain / Snow**
- Reproducible 80/20 validation split with seed `42`
- Image augmentation during training
- GPU-aware PyTorch inference with CUDA when available
- FastAPI endpoint for real image inference
- Browser upload, drag-and-drop, preview, sample inputs, loading states, and errors
- No fake client-side predictions when the API is unavailable
- Confidence shown separately from measured model accuracy
- Static frontend is deployable independently from the heavyweight PyTorch backend

## Model Performance

| Metric | Result |
|---|---:|
| Validation accuracy | **93.29%** |
| Validation samples | 417 |
| Correct predictions | 389 |
| Incorrect predictions | 28 |
| Input | 224 × 224 RGB |
| Classes | Fog / Rain / Snow |
| Backbone | ResNet18 |
| Validation split | 20% |
| Random seed | 42 |

### Class-wise validation metrics

| Class | Precision | Recall | F1 |
|---|---:|---:|---:|
| Fog | 0.9333 | 0.9256 | 0.9295 |
| Rain | 0.9338 | 0.9137 | 0.9236 |
| Snow | 0.9317 | 0.9554 | 0.9434 |
| **Macro average** | **0.9329** | **0.9316** | **0.9322** |

### Confusion matrix

```text
Actual \ Predicted    Fog   Rain  Snow
Fog                  112      3     6
Rain                   7    127     5
Snow                   1      6   150
```

The full generated report is available at `reports/evaluation_report.txt` after evaluation.

## Architecture

```text
                    ┌─────────────────────────────┐
                    │        Browser UI           │
                    │  Upload / Preview / Result  │
                    └──────────────┬──────────────┘
                                   │ multipart/form-data
                                   ▼
                    ┌─────────────────────────────┐
                    │        FastAPI API           │
                    │       POST /predict         │
                    └──────────────┬──────────────┘
                                   │
                         Resize + Normalize
                                   │
                                   ▼
                    ┌─────────────────────────────┐
                    │         ResNet18            │
                    │      Transfer Learned       │
                    └──────────────┬──────────────┘
                                   │
                                Softmax
                                   │
                                   ▼
                     Fog  /  Rain  /  Snow
```

## Repository Structure

```text
AI-Weather-Classifier/
├── backend/
│   └── main.py                 # FastAPI inference service
├── frontend/
│   ├── index.html              # AeroVision CV interface
│   ├── script.js               # Upload + API interaction
│   └── style.css               # UI styling
├── models/
│   └── README.md               # Checkpoint instructions
├── reports/
│   └── evaluation_report.txt   # Validation metrics
├── data/                       # Local dataset (ignored by Git)
├── train.py                    # Model training
├── evaluate.py                 # Validation evaluation
├── predict.py                  # Single-image / batch prediction
├── requirements.txt            # Python dependencies
├── .gitignore
├── LICENSE
└── README.md
```

## Dataset Layout

The training code expects the labelled dataset at:

```text
data/
└── 3_3_train/
    └── train/
        ├── fog/
        ├── rain/
        └── snow/
```

The external unlabeled prediction set is expected at:

```text
data/
└── 3_3_test_fin/
    └── test/
        ├── 0.png
        ├── 1.png
        └── ...
```

The raw dataset is intentionally excluded from Git because it is large and is not required to review the source code.

## Model Checkpoint

The trained checkpoint is intentionally not committed to the repository.

After training, the application expects:

```text
models/weather_resnet18.pth
```

Generate it with:

```bash
python train.py
```

For a hosted API, provide this checkpoint to the backend deployment using the host's persistent storage or artifact/model-storage mechanism.

## Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/harshitdev-wq/AI-Weather-Classifier.git
cd AI-Weather-Classifier
```

### 2. Install Python dependencies

```bash
python -m pip install -r requirements.txt
```

For NVIDIA GPU acceleration, install the PyTorch build appropriate for your CUDA environment.

### 3. Add the dataset

Place the labelled images under:

```text
data/3_3_train/train/fog/
data/3_3_train/train/rain/
data/3_3_train/train/snow/
```

### 4. Train

```bash
python train.py
```

The best validation checkpoint is written to:

```text
models/weather_resnet18.pth
```

### 5. Evaluate

```bash
python evaluate.py
```

This recreates the same validation split using seed `42` and writes:

```text
reports/evaluation_report.txt
```

### 6. Predict one image

```bash
python predict.py "data/3_3_test_fin/test/0.png"
```

### 7. Predict the complete unlabeled test folder

```bash
python predict.py
```

The generated CSV is written to:

```text
reports/weather_predictions.csv
```

### 8. Start the FastAPI backend

From the repository root:

```bash
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

Health endpoint:

```text
http://127.0.0.1:8000/health
```

### 9. Serve the frontend locally

From the repository root, in a second terminal:

```bash
python -m http.server 5500
```

Open:

```text
http://127.0.0.1:5500/frontend/index.html
```

## API Contract

### `GET /`

Returns service metadata, model status, supported classes, and validation accuracy.

### `GET /health`

Reports whether the model checkpoint is loaded successfully.

### `POST /predict`

Accepts a multipart form upload using the field name `file`.

Supported image types:

```text
image/jpeg
image/png
image/webp
```

Maximum request image size: **15 MB**.

Example response:

```json
{
  "prediction": "rain",
  "confidence": 99.96,
  "probabilities": {
    "fog": 0.01,
    "rain": 99.96,
    "snow": 0.03
  },
  "validation_accuracy": 93.29,
  "device": "cuda"
}
```

## Confidence vs Accuracy

The API returns a per-image **softmax confidence**. That value describes the distribution produced by the network for the submitted image; it is not the same thing as validation accuracy and does not guarantee a correct prediction.

The UI therefore presents these as separate concepts.

## Deployment

The project is intentionally split into two deployable pieces because PyTorch and TorchVision are too large for a standard Vercel Python Function bundle.

### Frontend: Vercel

Deploy the `frontend/` directory as a static site.

Recommended Vercel configuration:

```text
Root Directory: frontend
Framework Preset: Other
Build Command: none
Output Directory: .
Install Command: none
```

This prevents Vercel from trying to install the backend's heavyweight PyTorch dependencies.

### Backend: Python-capable host

Deploy `backend/main.py` on a service that supports Python and the required PyTorch runtime. The backend deployment must have access to:

```text
models/weather_resnet18.pth
```

After the backend is deployed, set the frontend's **API Config** endpoint to the public `/predict` URL of that service.

## Engineering Notes

### Why ResNet18?

ResNet18 is a compact convolutional architecture that provides a strong transfer-learning baseline for small-to-medium image classification tasks while remaining practical for local GPU inference.

### Why the repository ignores data and checkpoints

The source repository is intended to remain lightweight, reviewable, and reproducible. Large image collections and binary model artifacts are therefore excluded from normal Git history.

### Why there is no simulated fallback

A disconnected client should report an inference failure rather than fabricate a weather label. This keeps the demo trustworthy and makes production issues visible.

## Troubleshooting

**Vercel reports a huge Python bundle:** confirm that the Vercel project root is set to `frontend`. The static frontend does not require the Python dependencies.

**The API reports `503 Model checkpoint not found`:** place `weather_resnet18.pth` at `models/weather_resnet18.pth` in the backend environment.

**The frontend says `Local API Target`:** when running locally, start the FastAPI server on `127.0.0.1:8000` or use API Config to select another endpoint.

**Predictions work locally but not online:** the frontend and backend must be deployed separately unless the hosting platform supports the full PyTorch runtime and artifact size.

## License

Released under the [MIT License](LICENSE).

## Author

Built by **Harshit** as a practical deep-learning and deployment project.
