import io
import os
import sys
import urllib.request
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import tensorflow as tf

# gdown handles Google Drive large-file downloads (Drive shows virus-scan page for >100MB)
try:
    import gdown
except ImportError:
    gdown = None
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config

# Local dev: model at project root, Prev Models/, or api/. Docker: often gets Git LFS pointer (invalid).
# Set MODEL_URL to download the real model at startup when the bundled file is missing or invalid.
_base = Path(__file__).resolve().parent
_root = _base.parent
_candidates = [
    _root / "plant_disease_v2.keras",
    _root / "trained_plant_disease_model.h5",
    _root / "Prev Models" / "trained_plant_disease_model.h5",
    _base / "plant_disease_v2.keras",
    _base / "trained_plant_disease_model.h5",
]
MODEL_PATH = next((p for p in _candidates if p.exists()), _base / "trained_plant_disease_model.h5")

# Min size for a valid HDF5 model (~193MB). Git LFS pointer is ~200 bytes.
MIN_MODEL_BYTES = 50_000_000

model: tf.keras.Model


class _LegacyInputLayer(tf.keras.layers.InputLayer):
    """
    Compatibility shim for models saved with old tf.keras that used 'batch_shape'
    in InputLayer config. Newer Keras versions reject this key, so we strip it.
    """

    @classmethod
    def from_config(cls, config):
        config = dict(config)
        config.pop("batch_shape", None)
        return super().from_config(config)


def _load_legacy_model(path: Path) -> tf.keras.Model:
    with tf.keras.utils.custom_object_scope({"InputLayer": _LegacyInputLayer}):
        return tf.keras.models.load_model(path)


def _get_model_path() -> Path:
    """Return path to a valid model file, downloading from MODEL_URL if needed."""
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size >= MIN_MODEL_BYTES:
        return MODEL_PATH
    url = os.getenv("MODEL_URL")
    if url:
        dest = _base / "trained_plant_disease_model.h5"
        dest.parent.mkdir(parents=True, exist_ok=True)
        if "drive.google.com" in url and gdown:
            gdown.download(url, str(dest), quiet=False, fuzzy=True)
        else:
            urllib.request.urlretrieve(url, dest)
        return dest
    raise RuntimeError(
        f"Model not found or invalid (Git LFS pointer?). "
        "Set MODEL_URL to a direct download URL, or ensure the real .h5 file is in the image."
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    path = _get_model_path()
    model = _load_legacy_model(path)
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
    cls: str = Field(serialization_alias="class")
    confidence: float

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)


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

    arr = np.array(img, dtype=np.float32)  # model has built-in Rescaling layer — do NOT divide again
    scores = model.predict(arr[np.newaxis], verbose=0)[0]
    idx = int(np.argmax(scores))

    return PredictResponse(cls=config.CLASS_NAMES[idx], confidence=round(float(scores[idx]), 4))
