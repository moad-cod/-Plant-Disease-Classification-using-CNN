from fastapi import APIRouter

from backend.app.schemas.prediction import HealthResponse
from backend.app.services.model_loader import model_status

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    status = model_status()
    return HealthResponse(status="ok", **status)
