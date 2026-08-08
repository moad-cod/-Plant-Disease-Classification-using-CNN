from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.app.api.health import router as health_router
from backend.app.api.predict import router as predict_router
from backend.app.core.config import API_PREFIX, APP_NAME
from backend.app.core.logging import configure_logging

configure_logging()

app = FastAPI(title=APP_NAME)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health_router, prefix=API_PREFIX)
app.include_router(predict_router, prefix=API_PREFIX)


@app.get("/")
def root() -> dict[str, str]:
    return {"message": APP_NAME, "docs": "/docs"}
