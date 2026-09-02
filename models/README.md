# Model checkpoint

The trained ResNet18 checkpoint is intentionally not committed to this repository.

The API expects:

```text
models/weather_resnet18.pth
```

Generate it with:

```bash
python train.py
```

## Local inference

After training, start the API with:

```bash
python -m uvicorn backend.main:app --reload
```

## Vercel deployment

The Vercel function is designed to **build without the checkpoint**. The `/health` endpoint reports `degraded` until the checkpoint is available, and `/predict` returns HTTP 503 rather than crashing the deployment.

For a fully working hosted inference service, the checkpoint must be supplied to the deployed runtime. Do not place the dataset in the repository. For production hosting, a smaller ONNX checkpoint plus `onnxruntime` is recommended if the PyTorch dependency bundle exceeds the platform's function-size limit.

Keeping raw image data and large binary artifacts out of Git keeps the source repository lightweight and reviewable.