# Model checkpoint

The trained ResNet18 checkpoint is intentionally not committed to this repository.

After training, the expected path is:

```text
models/weather_resnet18.pth
```

Generate it with:

```bash
python train.py
```

Keeping the binary checkpoint and raw dataset out of Git makes the source repository lightweight and reviewable.