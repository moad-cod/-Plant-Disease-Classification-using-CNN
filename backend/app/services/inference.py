import numpy as np
import torch
from PIL import Image

from backend.app.core.config import GOAD_EPSILON
from backend.app.schemas.prediction import PredictionResponse, TopPrediction
from backend.app.services.model_loader import CLASSES, ModelBundle, get_disease_info
from backend.app.services.preprocessing import GOAD_TRANSFORM_COUNT, GOAD_TRANSFORMS, INFERENCE_TRANSFORM


def _compute_goad_score(
    pil_image: Image.Image,
    bundle: ModelBundle,
    cluster_centers,
    cov_inv,
    epsilon: float = GOAD_EPSILON,
) -> float:
    log_probs = []
    for transform_index, transform in enumerate(GOAD_TRANSFORMS):
        tensor = transform(pil_image).unsqueeze(0).to(bundle.device)
        with torch.no_grad():
            features, _ = bundle.model.extract_features(tensor)
        feature_vector = features[0].cpu().numpy()
        mahalanobis_distances = np.array(
            [
                float((feature_vector - cluster_centers[candidate]) @ cov_inv @ (feature_vector - cluster_centers[candidate]))
                for candidate in range(GOAD_TRANSFORM_COUNT)
            ]
        )
        negative_distances = -mahalanobis_distances
        exp_distances = np.exp(negative_distances - negative_distances.max())
        numerator = exp_distances[transform_index] + epsilon
        denominator = exp_distances.sum() + GOAD_TRANSFORM_COUNT * epsilon
        log_probs.append(np.log(numerator / denominator))

    return float(-sum(log_probs))


def predict_image(pil_image: Image.Image, bundle: ModelBundle) -> PredictionResponse:
    tensor = INFERENCE_TRANSFORM(pil_image).unsqueeze(0).to(bundle.device)

    bundle.model.eval()
    with torch.no_grad():
        _, logits = bundle.model.extract_features(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
        msp_confidence = float(probabilities.max().item())

    anomaly_params = bundle.anomaly_params
    if anomaly_params and msp_confidence < anomaly_params["msp_threshold"]:
        return PredictionResponse(
            is_anomaly=True,
            reason=f"MSP confidence too low ({msp_confidence * 100:.1f}% < {anomaly_params['msp_threshold'] * 100:.1f}%).",
            confidence=msp_confidence,
            msp_confidence=msp_confidence,
            goad_score=0.0,
            anomaly_checked=True,
            top5=[],
        )

    goad_score = 0.0
    if anomaly_params:
        goad_score = _compute_goad_score(
            pil_image,
            bundle,
            anomaly_params["cluster_centers"],
            anomaly_params["cov_inv"],
            anomaly_params.get("epsilon", GOAD_EPSILON),
        )
        if goad_score > anomaly_params["goad_threshold"]:
            return PredictionResponse(
                is_anomaly=True,
                reason=f"GOAD unusual pattern detected (score {goad_score:.2f} > threshold {anomaly_params['goad_threshold']:.2f}).",
                confidence=msp_confidence,
                msp_confidence=msp_confidence,
                goad_score=goad_score,
                anomaly_checked=True,
                top5=[],
            )

    top_probabilities, top_indices = torch.topk(probabilities, 5)
    top5 = [
        TopPrediction(class_name=CLASSES[index.item()], probability=float(prob.item()))
        for prob, index in zip(top_probabilities.cpu(), top_indices.cpu())
    ]
    plant, condition, (severity, description, treatment) = get_disease_info(top5[0].class_name)

    return PredictionResponse(
        is_anomaly=False,
        plant=plant,
        condition=condition,
        severity=severity,
        description=description,
        treatment=treatment,
        confidence=top5[0].probability,
        msp_confidence=msp_confidence,
        goad_score=goad_score,
        anomaly_checked=anomaly_params is not None,
        top5=top5,
    )
