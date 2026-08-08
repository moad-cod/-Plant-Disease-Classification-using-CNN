from pydantic import BaseModel, Field


class TopPrediction(BaseModel):
    class_name: str = Field(..., serialization_alias="class")
    probability: float


class PredictionResponse(BaseModel):
    is_anomaly: bool
    reason: str = ""
    plant: str | None = None
    condition: str | None = None
    severity: str | None = None
    description: str | None = None
    treatment: str | None = None
    confidence: float
    msp_confidence: float
    goad_score: float
    anomaly_checked: bool
    top5: list[TopPrediction]


class HealthResponse(BaseModel):
    status: str
    model_available: bool
    anomaly_params_available: bool
