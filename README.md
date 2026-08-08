# Plant Disease Classification using CNN

AI-assisted plant disease diagnosis using an EfficientNet-B3 model, MSP confidence checks, and GOAD anomaly detection. The project is organized as a React frontend with a FastAPI backend.

## Project Structure

```text
frontend/          React + Vite application
backend/           FastAPI inference API
models/            PyTorch model and anomaly parameters
reports/figures/   Evaluation charts
research/          Notebook and experiment workspace
scripts/           Training/evaluation/export placeholders
```

## Run Locally

Create and activate a Python environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Start the backend:

```bash
uvicorn backend.app.main:app --reload
```

Start the frontend:

```bash
cd frontend
npm install
npm run dev -- --host 127.0.0.1
```

Open `http://127.0.0.1:5173`.

## API

- `GET /api/health` returns model and anomaly artifact availability.
- `POST /api/predict` accepts an uploaded JPG/PNG file and returns the prediction result.

Example:

```bash
curl -X POST http://127.0.0.1:8000/api/predict \
  -F "file=@leaf.jpg"
```

## Docker

```bash
docker compose up --build
```

The frontend is exposed on `http://127.0.0.1:5173` and the backend on `http://127.0.0.1:8000`.

## Tests

```bash
pytest backend/tests
cd frontend && npm run build
```

## Model Artifacts

The inference backend expects:

- `models/plant-disease-model.pth`
- `models/anomaly_params.pkl`

The full checkpoint is kept at `models/plant-disease-model-complete.pth` for reference.
