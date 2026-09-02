"""FastAPI inference service for the AI Weather Classifier."""
from __future__ import annotations

import os
from io import BytesIO

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, UnidentifiedImageError
from torchvision import models, transforms

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "weather_resnet18.pth")
CLASSES = ["fog", "rain", "snow"]
VALIDATION_ACCURACY = 93.29
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="AI Weather Classifier API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

_model = None
_model_error = None


def build_model():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Sequential(
        torch.nn.Dropout(0.30),
        torch.nn.Linear(model.fc.in_features, len(CLASSES)),
    )
    return model


def get_model():
    """Load lazily so deployment can build even before the checkpoint is supplied."""
    global _model, _model_error
    if _model is not None:
        return _model
    if _model_error is not None:
        raise RuntimeError(_model_error)
    if not os.path.isfile(MODEL_PATH):
        _model_error = (
            f"Model checkpoint not found at {MODEL_PATH}. "
            "Add models/weather_resnet18.pth before enabling inference."
        )
        raise RuntimeError(_model_error)
    try:
        model = build_model()
        state_dict = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True)
        model.load_state_dict(state_dict)
        model.to(DEVICE).eval()
        _model = model
        return model
    except Exception as exc:
        _model_error = f"Unable to load model checkpoint: {exc}"
        raise RuntimeError(_model_error) from exc


def model_status() -> tuple[bool, str | None]:
    try:
        get_model()
        return True, None
    except RuntimeError as exc:
        return False, str(exc)


@app.get("/")
@app.get("/api/")
def root():
    loaded, error = model_status()
    return {
        "name": "AI Weather Classifier API",
        "model": "ResNet18",
        "classes": CLASSES,
        "validation_accuracy": VALIDATION_ACCURACY,
        "device": str(DEVICE),
        "model_loaded": loaded,
        "model_error": error,
        "docs": "/docs",
    }


@app.get("/health")
@app.get("/api/health")
def health():
    loaded, error = model_status()
    return {
        "status": "ok" if loaded else "degraded",
        "device": str(DEVICE),
        "model_loaded": loaded,
        "model_error": error,
    }


async def _predict(file: UploadFile):
    allowed = {"image/jpeg", "image/png", "image/webp"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=415, detail="Unsupported image type. Use JPG, PNG, or WEBP.")

    raw = await file.read()
    if len(raw) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds the 15 MB upload limit.")

    try:
        image = Image.open(BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable image.") from exc

    try:
        model = get_model()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    tensor = _transform(image).unsqueeze(0).to(DEVICE)
    with torch.inference_mode():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
        index = int(probabilities.argmax())

    return {
        "prediction": CLASSES[index],
        "confidence": round(float(probabilities[index] * 100), 2),
        "probabilities": {
            name: round(float(probabilities[i] * 100), 2)
            for i, name in enumerate(CLASSES)
        },
        "validation_accuracy": VALIDATION_ACCURACY,
        "device": str(DEVICE),
    }


@app.post("/predict")
@app.post("/api/predict")
async def predict(file: UploadFile = File(...)):
    return await _predict(file)
