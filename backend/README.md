# FastAPI Backend

This directory contains the Python inference service for **AeroVision CV**.

## Run locally

From the repository root:

```bash
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

API docs:

```text
http://127.0.0.1:8000/docs
```

Health:

```text
http://127.0.0.1:8000/health
```

Prediction endpoint:

```text
POST /predict
```

The multipart form field must be named `file`.

## Model artifact

The service expects the trained checkpoint at:

```text
models/weather_resnet18.pth
```

The binary checkpoint is intentionally ignored by Git. Train it locally with:

```bash
python train.py
```

For a hosted deployment, provide the checkpoint through the platform's persistent storage or model-artifact mechanism.

## Runtime behavior

- Uses CUDA automatically when a compatible NVIDIA runtime is available.
- Falls back to CPU when CUDA is unavailable.
- Loads the checkpoint lazily so a missing artifact does not prevent the service from booting.
- Returns HTTP `503` for prediction requests while the model artifact is unavailable.
- Accepts JPG, PNG, and WEBP images up to 15 MB.
- Returns the predicted class, confidence, class probabilities, validation accuracy, and inference device.

## Expected response

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
