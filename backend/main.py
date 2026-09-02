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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="AI Weather Classifier API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def build_model():
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Sequential(torch.nn.Dropout(0.30), torch.nn.Linear(model.fc.in_features, len(CLASSES)))
    return model


if not os.path.isfile(MODEL_PATH):
    raise RuntimeError(
        f"Model checkpoint not found at {MODEL_PATH}. "
        "Train the model first with `python train.py`."
    )

model = build_model()
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.to(DEVICE).eval()


@app.get("/")
def root():
    return {
        "name": "AI Weather Classifier API",
        "model": "ResNet18",
        "classes": CLASSES,
        "validation_accuracy": 93.29,
        "device": str(DEVICE),
        "docs": "/docs",
    }


@app.get("/health")
def health():
    return {"status": "ok", "device": str(DEVICE), "model_loaded": True}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    allowed = {"image/jpeg", "image/png", "image/webp"}
    if file.content_type not in allowed:
        raise HTTPException(status_code=415, detail="Unsupported image type. Use JPG, PNG, or WEBP.")

    raw = await file.read()
    if len(raw) > 15 * 1024 * 1024:
        raise HTTPException(status_code=413, detail="Image exceeds the 15 MB upload limit.")

    try:
        image = Image.open(BytesIO(raw)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise HTTPException(status_code=400, detail="The uploaded file is not a readable image.") from exc

    tensor = _transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
        index = int(probabilities.argmax())

    return {
        "prediction": CLASSES[index],
        "confidence": round(float(probabilities[index] * 100), 2),
        "probabilities": {name: round(float(probabilities[i] * 100), 2) for i, name in enumerate(CLASSES)},
        "validation_accuracy": 93.29,
        "device": str(DEVICE),
    }
