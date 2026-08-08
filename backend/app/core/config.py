from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports" / "figures"

MODEL_PATH = MODELS_DIR / "plant-disease-model.pth"
COMPLETE_MODEL_PATH = MODELS_DIR / "plant-disease-model-complete.pth"
ANOMALY_PARAMS_PATH = MODELS_DIR / "anomaly_params.pkl"

APP_NAME = "Plant Disease Classification API"
API_PREFIX = "/api"
IMAGE_SIZE = 300
NUM_CLASSES = 38
GOAD_EPSILON = 0.1
