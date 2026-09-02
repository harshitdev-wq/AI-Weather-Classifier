"""Evaluate the saved ResNet18 checkpoint on the reproducible validation split."""
from __future__ import annotations

import os
import random

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

TRAIN_DIR = "data/3_3_train/train"
MODEL_PATH = "models/weather_resnet18.pth"
REPORT_PATH = "reports/evaluation_report.txt"
IMAGE_SIZE = 224
BATCH_SIZE = 32
VALIDATION_SPLIT = 0.20
SEED = 42


def build_model(num_classes: int):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Sequential(torch.nn.Dropout(0.30), torch.nn.Linear(model.fc.in_features, num_classes))
    return model


def main() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    full = datasets.ImageFolder(TRAIN_DIR, transform=transform)
    indices = list(range(len(full)))
    random.Random(SEED).shuffle(indices)
    split = int(len(full) * VALIDATION_SPLIT)
    val = Subset(full, indices[:split])
    loader = DataLoader(val, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = build_model(len(full.classes))
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()

    matrix = [[0 for _ in full.classes] for _ in full.classes]
    with torch.no_grad():
        for images, labels in loader:
            preds = model(images.to(device)).argmax(1).cpu().tolist()
            for actual, pred in zip(labels.tolist(), preds):
                matrix[actual][pred] += 1

    total = sum(map(sum, matrix))
    correct = sum(matrix[i][i] for i in range(len(matrix)))
    accuracy = correct / total if total else 0.0

    lines = [
        "AI Weather Classifier — Evaluation Report",
        "=" * 48,
        f"Validation samples : {total}",
        f"Correct            : {correct}",
        f"Validation accuracy: {accuracy * 100:.2f}%",
        "",
        "Class-wise metrics:",
    ]
    macro_p = macro_r = macro_f1 = 0.0
    for i, name in enumerate(full.classes):
        tp = matrix[i][i]
        fp = sum(matrix[r][i] for r in range(len(matrix))) - tp
        fn = sum(matrix[i][c] for c in range(len(matrix))) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        macro_p += precision
        macro_r += recall
        macro_f1 += f1
        lines.append(f"  {name:<6} precision={precision:.4f} recall={recall:.4f} f1={f1:.4f}")

    k = len(full.classes)
    lines.extend([
        "",
        f"Macro avg     precision={macro_p/k:.4f} recall={macro_r/k:.4f} f1={macro_f1/k:.4f}",
        "",
        "Confusion matrix (rows=actual, columns=predicted):",
        "             " + "  ".join(f"{c:>5}" for c in full.classes),
    ])
    for name, row in zip(full.classes, matrix):
        lines.append(f"  {name:<8} " + "  ".join(f"{v:>5}" for v in row))

    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    report = "\n".join(lines) + "\n"
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        f.write(report)
    print(report)
    print(f"Saved report -> {REPORT_PATH}")


if __name__ == "__main__":
    main()
