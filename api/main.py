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


class _DTypePolicyShim:
    """
    Lets Keras deserialize legacy DTypePolicy configs. Every layer (Conv2D, Dense, etc.)
    has dtype: {'class_name': 'DTypePolicy', 'config': {'name': 'float32'}} in its config;
    registering this in custom_object_scope fixes all of them at once.
    """

    @classmethod
    def from_config(cls, config):
        if not isinstance(config, dict):
            return tf.keras.mixed_precision.Policy("float32")
        inner = config.get("config", config)
        name = inner.get("name", "float32") if isinstance(inner, dict) else "float32"
        return tf.keras.mixed_precision.Policy(name)


def _clean_legacy_config(config: dict) -> dict:
    """
    Normalize legacy Keras layer configs:
    - Convert DTypePolicy dicts to plain dtype strings (e.g. 'float32')
    - Drop obsolete arguments like 'data_format' that newer layers reject.
    """
    cfg = dict(config)
    dtype_cfg = cfg.get("dtype")
    if isinstance(dtype_cfg, dict) and dtype_cfg.get("class_name") == "DTypePolicy":
        cfg["dtype"] = dtype_cfg.get("config", {}).get("name", "float32")
    # These older preprocessing layers used to accept 'data_format' in config;
    # newer versions often do not.
    cfg.pop("data_format", None)
    return cfg


class _LegacyInputLayer(tf.keras.layers.InputLayer):
    """
    Compatibility shim for models saved with old tf.keras that used 'batch_shape'.
    We convert it to batch_input_shape so the first Conv2D gets a defined channel dim.
    """

    @classmethod
    def from_config(cls, config):
        cfg = dict(config)
        batch_shape = cfg.pop("batch_shape", None)
        if batch_shape is not None:
            cfg["batch_input_shape"] = batch_shape
        cfg = _clean_legacy_config(cfg)
        return super().from_config(cfg)


class _LegacyRescaling(tf.keras.layers.Rescaling):
    """
    Compatibility shim for models whose Rescaling layer was saved with a 'dtype'
    config pointing to the now-unknown 'DTypePolicy' object. We map that to a
    plain dtype string (e.g. 'float32') so modern Keras deserialization works.
    """

    @classmethod
    def from_config(cls, config):
        cfg = _clean_legacy_config(config)
        return super().from_config(cfg)


class _LegacyRandomFlip(tf.keras.layers.RandomFlip):
    """
    Compatibility shim for RandomFlip layers saved with legacy dtype policies and
    an obsolete 'data_format' argument.
    """

    @classmethod
    def from_config(cls, config):
        cfg = _clean_legacy_config(config)
        return super().from_config(cfg)


class _LegacyRandomRotation(tf.keras.layers.RandomRotation):
    """
    Compatibility shim for RandomRotation layers saved with legacy dtype policies and
    obsolete 'data_format' argument.
    """

    @classmethod
    def from_config(cls, config):
        cfg = _clean_legacy_config(config)
        return super().from_config(cfg)


class _LegacyRandomZoom(tf.keras.layers.RandomZoom):
    """
    Compatibility shim for RandomZoom layers with legacy dtype/data_format config.
    """

    @classmethod
    def from_config(cls, config):
        cfg = _clean_legacy_config(config)
        return super().from_config(cfg)


def _load_legacy_model(path: Path) -> tf.keras.Model:
    custom_objects = {
        "DTypePolicy": _DTypePolicyShim,
        "InputLayer": _LegacyInputLayer,
        "Rescaling": _LegacyRescaling,
        "RandomFlip": _LegacyRandomFlip,
        "RandomRotation": _LegacyRandomRotation,
        "RandomZoom": _LegacyRandomZoom,
    }
    with tf.keras.utils.custom_object_scope(custom_objects):
        # compile=False skips optimizer deserialization entirely — safe for inference-only use
        return tf.keras.models.load_model(path, compile=False)


# Fallback model URL when bundled file is missing or a Git LFS pointer (~200 bytes).
_DEFAULT_MODEL_URL = "https://drive.google.com/uc?id=1N1eBI4xpBEOGlWcdIEqSak8nF6n4yDJH"


def _get_model_path() -> Path:
    """Return path to a valid model file, downloading from MODEL_URL if needed."""
    if MODEL_PATH.exists() and MODEL_PATH.stat().st_size >= MIN_MODEL_BYTES:
        return MODEL_PATH
    url = os.getenv("MODEL_URL") or _DEFAULT_MODEL_URL
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

# CORS: use CORS_ORIGINS env (comma-separated) to restrict origins in production.
# Defaults to * so the API works out of the box without extra config.
_cors_env = os.getenv("CORS_ORIGINS", "*").strip()
_origins_list = [o.strip() for o in _cors_env.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins_list,
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
