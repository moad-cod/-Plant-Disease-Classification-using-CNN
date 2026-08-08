# PROJECT

**Plant-Disease-Classification-using-CNN**

A plant disease diagnosis application split into a React/Vite frontend and a FastAPI inference backend. The backend serves an EfficientNet-B3 model trained for 38 PlantVillage classes and includes MSP + GOAD anomaly detection.

---

# STACK

| Layer | Technology |
| --- | --- |
| Frontend | React, Vite, CSS, lucide-react |
| Backend | FastAPI, Uvicorn |
| ML | PyTorch, torchvision, EfficientNet-B3 |
| Image | Pillow |
| Numerical | NumPy |
| Tests | pytest, FastAPI TestClient |
| Packaging | Docker, docker-compose |

---

# FOLDER_STRUCTURE

```text
Plant-Disease-Classification-using-CNN/
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   ├── pages/
│   │   ├── services/
│   │   ├── assets/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.jsx
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── postcss.config.js
├── backend/
│   ├── app/
│   │   ├── main.py
│   │   ├── api/
│   │   ├── core/
│   │   ├── schemas/
│   │   └── services/
│   └── tests/
├── models/
├── reports/figures/
├── research/
├── scripts/
├── tests/
├── Dockerfile.backend
├── Dockerfile.frontend
├── docker-compose.yml
├── requirements.txt
└── README.md
```

---

# ENTRY_POINTS

| Target | Command | URL |
| --- | --- | --- |
| Backend API | `uvicorn backend.app.main:app --reload` | `http://127.0.0.1:8000` |
| Frontend | `cd frontend && npm run dev -- --host 127.0.0.1` | `http://127.0.0.1:5173` |
| Backend tests | `pytest backend/tests` | - |
| Frontend build | `cd frontend && npm run build` | - |

---

# BACKEND_FLOW

1. `backend.app.main` creates the FastAPI app and mounts `/api/health` and `/api/predict`.
2. `/api/health` checks whether the model and anomaly parameter files exist.
3. `/api/predict` accepts a JPG/PNG upload, decodes it with Pillow, and lazily loads model resources.
4. Inference resizes to 300 x 300, normalizes with ImageNet statistics, runs EfficientNet-B3, then applies MSP and GOAD anomaly checks.
5. The response returns anomaly status, confidence, top-5 predictions, disease metadata, and treatment guidance.

---

# ARTIFACTS

| Path | Purpose |
| --- | --- |
| `models/plant-disease-model.pth` | Inference model weights |
| `models/plant-disease-model-complete.pth` | Full checkpoint artifact |
| `models/anomaly_params.pkl` | MSP and GOAD thresholds/parameters |
| `reports/figures/*.png` | Evaluation and distribution figures |

---

# KNOWN_NOTES

- The training pipeline is not included; `scripts/train.py`, `scripts/evaluate.py`, and `scripts/export_model.py` are placeholders for future integration.
- The backend loads the model lazily on the first prediction request, not during health checks.
- The frontend API base URL defaults to `http://127.0.0.1:8000/api` and can be overridden with `VITE_API_BASE_URL`.
