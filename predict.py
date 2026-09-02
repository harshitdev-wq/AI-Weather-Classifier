"""Predict the weather class for one image or an entire unlabeled test folder."""
from __future__ import annotations

import csv
import os
import sys

import torch
from PIL import Image
from torchvision import models, transforms

MODEL_PATH = "models/weather_resnet18.pth"
TEST_DIR = "data/3_3_test_fin/test"
OUTPUT_CSV = "reports/weather_predictions.csv"
CLASSES = ["fog", "rain", "snow"]

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])


def load_model(device):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Sequential(torch.nn.Dropout(0.30), torch.nn.Linear(model.fc.in_features, len(CLASSES)))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    return model.to(device).eval()


def predict_image(model, device, path):
    with Image.open(path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
        index = int(probabilities.argmax())
    return CLASSES[index], float(probabilities[index] * 100)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    model = load_model(device)

    if len(sys.argv) > 1:
        path = sys.argv[1]
        label, confidence = predict_image(model, device, path)
        print(f"Prediction: {label.upper()}")
        print(f"Confidence: {confidence:.2f}%")
        return

    if not os.path.isdir(TEST_DIR):
        raise FileNotFoundError(f"Test directory not found: {TEST_DIR}")
    files = sorted(f for f in os.listdir(TEST_DIR) if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")))
    rows = []
    for i, name in enumerate(files, 1):
        label, confidence = predict_image(model, device, os.path.join(TEST_DIR, name))
        rows.append((name, label, f"{confidence:.2f}"))
        print(f"[{i}/{len(files)}] {name} -> {label} ({confidence:.2f}%)")

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "prediction", "confidence_percent"])
        writer.writerows(rows)
    print(f"Saved predictions -> {OUTPUT_CSV}")


if __name__ == "__main__":
    main()
