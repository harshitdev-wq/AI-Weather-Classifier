"""Train the ResNet18 weather classifier.

Expected dataset layout:
    data/3_3_train/train/{fog,rain,snow}/...
"""
from __future__ import annotations

import copy
import os
import random

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

TRAIN_DIR = "data/3_3_train/train"
MODEL_PATH = "models/weather_resnet18.pth"
IMAGE_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 12
LEARNING_RATE = 1e-4
VALIDATION_SPLIT = 0.20
SEED = 42


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model(num_classes: int) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT
    model = models.resnet18(weights=weights)
    model.fc = nn.Sequential(nn.Dropout(0.30), nn.Linear(model.fc.in_features, num_classes))
    return model


def main() -> None:
    seed_everything(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    if not os.path.isdir(TRAIN_DIR):
        raise FileNotFoundError(f"Training directory not found: {TRAIN_DIR}")

    train_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.20, contrast=0.20, saturation=0.15),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    base = datasets.ImageFolder(TRAIN_DIR)
    train_ds = datasets.ImageFolder(TRAIN_DIR, transform=train_tf)
    val_ds = datasets.ImageFolder(TRAIN_DIR, transform=val_tf)
    indices = list(range(len(base)))
    rng = random.Random(SEED)
    rng.shuffle(indices)
    split = int(len(base) * VALIDATION_SPLIT)
    val_indices, train_indices = indices[:split], indices[split:]

    device_pin = device.type == "cuda"
    train_loader = DataLoader(Subset(train_ds, train_indices), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=device_pin)
    val_loader = DataLoader(Subset(val_ds, val_indices), batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=device_pin)

    print(f"Classes: {base.classes}")
    print(f"Images: {len(base)} | Train: {len(train_indices)} | Validation: {len(val_indices)}")

    model = build_model(len(base.classes)).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_acc = -1.0
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        train_correct = train_total = 0
        train_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * labels.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += labels.size(0)

        model.eval()
        val_correct = val_total = 0
        val_loss = 0.0
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                val_loss += criterion(logits, labels).item() * labels.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += labels.size(0)

        train_acc = train_correct / train_total
        val_acc = val_correct / val_total
        print(f"Epoch {epoch + 1:02d}/{EPOCHS} | Train loss {train_loss/train_total:.4f} | Train acc {train_acc*100:.2f}% | Val loss {val_loss/val_total:.4f} | Val acc {val_acc*100:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"  Saved best model -> {MODEL_PATH}")

    print(f"Best validation accuracy: {best_acc * 100:.2f}%")


if __name__ == "__main__":
    main()
