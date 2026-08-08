from dataclasses import dataclass
from functools import lru_cache
import logging
import pickle

import torch
import torch.nn as nn
import torchvision.models as models

from backend.app.core.config import ANOMALY_PARAMS_PATH, MODEL_PATH, NUM_CLASSES

log = logging.getLogger(__name__)


CLASSES = sorted(
    [
        "Apple___Apple_scab",
        "Apple___Black_rot",
        "Apple___Cedar_apple_rust",
        "Apple___healthy",
        "Blueberry___healthy",
        "Cherry_(including_sour)___Powdery_mildew",
        "Cherry_(including_sour)___healthy",
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "Corn_(maize)___Common_rust_",
        "Corn_(maize)___Northern_Leaf_Blight",
        "Corn_(maize)___healthy",
        "Grape___Black_rot",
        "Grape___Esca_(Black_Measles)",
        "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)",
        "Grape___healthy",
        "Orange___Haunglongbing_(Citrus_greening)",
        "Peach___Bacterial_spot",
        "Peach___healthy",
        "Pepper,_bell___Bacterial_spot",
        "Pepper,_bell___healthy",
        "Potato___Early_blight",
        "Potato___Late_blight",
        "Potato___healthy",
        "Raspberry___healthy",
        "Soybean___healthy",
        "Squash___Powdery_mildew",
        "Strawberry___Leaf_scorch",
        "Strawberry___healthy",
        "Tomato___Bacterial_spot",
        "Tomato___Early_blight",
        "Tomato___Late_blight",
        "Tomato___Leaf_Mold",
        "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites Two-spotted_spider_mite",
        "Tomato___Target_Spot",
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
        "Tomato___Tomato_mosaic_virus",
        "Tomato___healthy",
    ]
)

DISEASE_INFO = {
    "Apple_scab": ("medium", "Fungal lesions causing dark scabby spots on leaves and fruit.", "Apply fungicide. Remove infected leaves. Prune for better airflow."),
    "Black_rot": ("high", "Fungal disease with circular brown spots and dark margins.", "Prune infected branches. Apply copper fungicide. Remove mummified fruit."),
    "Cedar_apple_rust": ("medium", "Orange rust spots on upper leaf surface, tubes below.", "Apply myclobutanil in spring. Remove nearby cedar trees if possible."),
    "Powdery_mildew": ("medium", "White powdery fungal coating on the leaf surface.", "Apply sulfur or neem-based fungicide. Improve air circulation."),
    "Cercospora_leaf_spot": ("medium", "Circular gray spots with dark purple borders.", "Apply fungicide. Remove crop debris. Avoid overhead watering."),
    "Common_rust": ("medium", "Orange-brown rust pustules on both leaf surfaces.", "Apply fungicide early in season. Use resistant corn varieties."),
    "Northern_Leaf_Blight": ("high", "Long tan cigar-shaped lesions with dark wavy borders.", "Apply fungicide. Rotate crops annually. Use resistant varieties."),
    "Esca": ("high", "Tiger-stripe yellowing pattern with dark wood streaks.", "No cure available. Remove infected vines to prevent spread."),
    "Leaf_blight": ("high", "Brown water-soaked lesions spreading rapidly across leaves.", "Apply copper fungicide. Avoid wet foliage. Improve drainage."),
    "Haunglongbing": ("high", "Yellow shoots and blotchy mottled asymmetric leaves.", "No cure. Remove infected trees immediately to stop spread."),
    "Bacterial_spot": ("medium", "Small water-soaked spots turning angular and brown.", "Apply copper spray. Avoid working with plants when foliage is wet."),
    "Early_blight": ("medium", "Dark concentric-ring spots like a target on leaves.", "Apply fungicide. Remove lower infected leaves. Mulch around base."),
    "Late_blight": ("high", "Water-soaked lesions turning dark brown, white mold below.", "Apply fungicide immediately. Remove and destroy infected plants."),
    "Leaf_Mold": ("medium", "Pale yellow spots above, olive-green mold on underside.", "Improve ventilation. Reduce humidity. Apply appropriate fungicide."),
    "Septoria_leaf_spot": ("medium", "Small circular spots with dark border and light gray center.", "Apply fungicide. Remove infected lower leaves promptly."),
    "Spider_mites": ("medium", "Yellow stippling with fine silky webbing on leaves.", "Apply miticide or neem oil. Increase ambient humidity around plants."),
    "Target_Spot": ("medium", "Circular brown spots with distinct concentric rings.", "Apply fungicide. Improve air circulation around the plant canopy."),
    "Yellow_Leaf_Curl_Virus": ("high", "Upward leaf curl, yellowing margins, stunted growth.", "No cure. Remove infected plants. Control whitefly vectors."),
    "mosaic_virus": ("high", "Mosaic light-dark green mottled pattern, distorted leaves.", "No cure. Remove infected plants. Control aphid vectors immediately."),
    "Leaf_scorch": ("medium", "Brown scorched margins and leaf tips due to stress.", "Improve irrigation consistency. Mulch around base of plant."),
    "healthy": ("none", "No disease detected. The plant appears healthy.", "Continue regular watering, fertilization, and routine monitoring."),
}


class EfficientNetB3(nn.Module):
    def __init__(self, num_classes: int = NUM_CLASSES):
        super().__init__()
        self.network = models.efficientnet_b3(weights=None)
        in_features = self.network.classifier[1].in_features
        self.network.classifier = nn.Sequential(
            nn.Dropout(p=0.3, inplace=True),
            nn.Linear(in_features, num_classes),
        )

    def forward(self, xb):
        return self.network(xb)

    def extract_features(self, xb):
        features = self.network.features(xb)
        features = self.network.avgpool(features)
        features = features.flatten(1)
        logits = self.network.classifier(features)
        return features, logits


@dataclass(frozen=True)
class ModelBundle:
    model: EfficientNetB3
    device: torch.device
    anomaly_params: dict | None


def model_status() -> dict[str, bool]:
    return {
        "model_available": MODEL_PATH.exists(),
        "anomaly_params_available": ANOMALY_PARAMS_PATH.exists(),
    }


@lru_cache(maxsize=1)
def load_model_bundle() -> ModelBundle:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EfficientNetB3(NUM_CLASSES)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()
    log.info("Model loaded from %s on %s", MODEL_PATH, device)

    anomaly_params = None
    if ANOMALY_PARAMS_PATH.exists():
        with ANOMALY_PARAMS_PATH.open("rb") as f:
            anomaly_params = pickle.load(f)
        log.info("Anomaly params loaded from %s", ANOMALY_PARAMS_PATH)
    else:
        log.warning("Anomaly params not found at %s", ANOMALY_PARAMS_PATH)

    return ModelBundle(model=model, device=device, anomaly_params=anomaly_params)


def get_disease_info(raw_class: str) -> tuple[str, str, tuple[str, str, str]]:
    plant, condition = raw_class.split("___")
    if condition == "healthy":
        return plant, "Healthy", DISEASE_INFO["healthy"]

    normalized_condition = condition.lower().replace("_", "").replace(" ", "")
    for key, value in DISEASE_INFO.items():
        if key.lower().replace("_", "") in normalized_condition:
            return plant, condition.replace("_", " "), value

    return plant, condition.replace("_", " "), (
        "medium",
        f"Disease detected in {plant}.",
        "Consult an agricultural expert.",
    )
