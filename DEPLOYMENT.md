# Deployment Runbook

## Recommended architecture

Deploy the two parts independently:

```text
Browser
   │
   ▼
Vercel ── static frontend (`frontend/`)
   │
   │ HTTPS POST /predict
   ▼
Python host ── FastAPI + PyTorch + ResNet18 checkpoint
```

This keeps the Vercel deployment lightweight and avoids packaging the large PyTorch/TorchVision runtime into a Vercel Python Function.

## Vercel frontend

Create/import the repository as a Vercel project and set:

```text
Root Directory: frontend
Framework Preset: Other
Build Command: leave empty
Output Directory: .
Install Command: leave empty
```

Do not point the Vercel project at the repository root for this frontend deployment. The Python training and backend files are intentionally outside the frontend deployment boundary.

## FastAPI backend

Use a Python-capable host for `backend/main.py`.

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Start the service:

```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

Provide the model checkpoint at:

```text
models/weather_resnet18.pth
```

Verify:

```text
GET /health
GET /docs
```

## Connect the frontend

Open **API Config** in AeroVision CV and enter:

```text
https://YOUR-BACKEND-DOMAIN/predict
```

The backend must allow CORS requests from the deployed frontend and expose:

```text
GET  /health
POST /predict
```

## Troubleshooting

### Vercel tries to install PyTorch

The Vercel project's **Root Directory** is almost certainly still the repository root. Change it to:

```text
frontend
```

Then redeploy without using the previous build cache.

### API returns 503

Check the backend logs. The most common cause is a missing model checkpoint at:

```text
models/weather_resnet18.pth
```

### Browser reports a CORS error

Check the FastAPI CORS configuration and make sure the frontend's deployed origin is allowed.

### Predictions are slow

CPU inference is expected to be slower than CUDA inference. The local development environment can use an NVIDIA GPU when the PyTorch installation has CUDA support.
