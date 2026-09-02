# AI Weather Classifier

A computer-vision application that classifies weather scenes into **fog**, **rain**, and **snow** using a pretrained **ResNet18** model with transfer learning.

## Deployment architecture

```text
Browser → Vercel (static frontend) → FastAPI backend → ResNet18
```

Vercel's standard Python Functions have a 500 MB function bundle limit. The PyTorch/Torchvision dependency tree for this project is much larger, so packaging the inference service directly as a Vercel Python Function produces a multi-gigabyte bundle and fails before deployment. citeturn790776search2

### Results

| Metric | Result |
|---|---:|
| Validation accuracy | **93.29%** |
| Validation samples | 417 |
| Correct predictions | 389 |
| Classes | Fog / Rain / Snow |
| Input size | 224 × 224 RGB |

> **Important:** 93.29% is measured on the held-out validation split. The external test images are unlabeled, so no test-set accuracy is claimed.

## Repository structure

```text
AI-Weather-Classifier/
├── backend/
│   └── main.py             # FastAPI inference service
├── frontend/
│   ├── index.html
│   ├── script.js
│   └── style.css
├── models/
│   └── README.md
├── reports/
├── data/                   # local dataset; ignored by Git
├── train.py
├── evaluate.py
├── predict.py
├── requirements.txt
└── README.md
```

## Local setup

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Place the labelled dataset at:

```text
data/3_3_train/train/fog/
data/3_3_train/train/rain/
data/3_3_train/train/snow/
```

Train:

```bash
python train.py
```

Evaluate:

```bash
python evaluate.py
```

Run the API:

```bash
python -m uvicorn backend.main:app --reload
```

Serve the frontend:

```bash
python -m http.server 5500
```

Open `http://127.0.0.1:5500/frontend/index.html`.

## Production deployment

### Frontend — Vercel

Deploy this repository as a **static site**. The Vercel deployment should not package the Python backend or PyTorch dependencies. The root rewrite points `/` to `frontend/index.html`.

### Backend — Python service

Deploy `backend/main.py` to a Python service that supports the PyTorch dependency tree. Use:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port $PORT
```

The backend must have `models/weather_resnet18.pth` available. After the backend is public, enter its `/predict` URL in the frontend's **API settings** dialog.

## API

`GET /health` returns backend and model status.

`POST /predict` accepts a multipart image field named `file` and returns the prediction, confidence, class probabilities, validation accuracy, and inference device.

## Model

The trained checkpoint is intentionally excluded from Git. After training, place it at:

```text
models/weather_resnet18.pth
```

## Confidence

The confidence shown by the UI is the model's softmax output for the selected image. It is not a guarantee of correctness and should not be confused with the 93.29% validation accuracy.

## License

MIT.