import io
import os
import sys
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# Local dev: model at project root. Docker: model copied to /app or downloaded via MODEL_URL.
# Supports both .keras and .h5 (notebook saves trained_plant_disease_model.h5).
_base = Path(__file__).resolve().parent
_candidates = [
    _base.parent / "plant_disease_v2.keras",
    _base.parent / "trained_plant_disease_model.h5",
    _base / "plant_disease_v2.keras",
    _base / "trained_plant_disease_model.h5",
]
MODEL_PATH = next((p for p in _candidates if p.exists()), _base / "trained_plant_disease_model.h5")

model: tf.keras.Model


def _ensure_model() -> Path:
    """Return path to model file, downloading from MODEL_URL if needed."""
    if MODEL_PATH.exists():
        return MODEL_PATH
    url = os.getenv("MODEL_URL")
    if url:
        dest = _base / "trained_plant_disease_model.h5"
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, dest)
        return dest
    raise RuntimeError(
        f"Model not found at {MODEL_PATH}. "
        "Add the model file to the project root, or set MODEL_URL to download it."
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    path = _ensure_model()
    model = tf.keras.models.load_model(path)
    yield


app = FastAPI(title="Plant Disease Detection API", lifespan=lifespan)

# CORS: use CORS_ORIGINS env (comma-separated) for production, e.g. "https://your-app.vercel.app"
_origins = os.getenv("CORS_ORIGINS", "http://localhost:3000").strip().split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins if o.strip()],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


class PredictResponse(BaseModel):
    cls: str
    confidence: float

    model_config = {"populate_by_name": True}

    def model_dump(self, **kwargs):
        d = super().model_dump(**kwargs)
        d["class"] = d.pop("cls")
        return d


@app.get("/health")
def health():
    """Health check for Railway / load balancers."""
    return {"status": "ok"}


@app.post("/predict", response_model=PredictResponse)
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image.")

    raw = await file.read()
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB").resize(config.IMG_SIZE)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    arr = np.array(img, dtype=np.float32) / 255.0   # normalize to [0, 1]
    scores = model.predict(arr[np.newaxis], verbose=0)[0]
    idx = int(np.argmax(scores))

    return PredictResponse(cls=config.CLASS_NAMES[idx], confidence=round(float(scores[idx]), 4))
