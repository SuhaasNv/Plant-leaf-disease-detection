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
from fastapi import Depends, FastAPI, File, HTTPException, Request, Security, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader
from PIL import Image
from pydantic import BaseModel, ConfigDict, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

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

# ── Rate limiting (slowapi) ───────────────────────────────────────────────────
# Default: 10 predict requests per minute per IP.
# Override with RATE_LIMIT env var, e.g. "20/minute" or "100/hour".
_rate_limit = os.getenv("RATE_LIMIT", "10/minute")
limiter = Limiter(key_func=get_remote_address, default_limits=[_rate_limit])
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)  # type: ignore[arg-type]

# ── API key auth ──────────────────────────────────────────────────────────────
# Set API_KEY env var on Railway. If unset, the endpoint is open (dev mode).
_API_KEY = os.getenv("API_KEY", "").strip() or None
_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def _require_api_key(key: str | None = Security(_api_key_header)) -> None:
    if _API_KEY and key != _API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")

# ── CORS ─────────────────────────────────────────────────────────────────────
# Set CORS_ORIGINS env var (comma-separated). Defaults to * (open).
_cors_env = os.getenv("CORS_ORIGINS", "*").strip()
_origins_list = ["*"] if _cors_env == "*" else [o.strip() for o in _cors_env.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_origins_list,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Input validation constants ────────────────────────────────────────────────
_MAX_FILE_BYTES = 10 * 1024 * 1024   # 10 MB
_MAX_IMAGE_DIM  = 4096               # max width or height in pixels

# Magic-byte signatures for accepted image types
_MAGIC: dict[bytes, str] = {
    b"\xff\xd8\xff": "image/jpeg",
    b"\x89PNG\r\n\x1a\n": "image/png",
    b"RIFF": "image/webp",  # WebP starts with RIFF
    b"GIF8": "image/gif",
}

def _validate_image_bytes(raw: bytes) -> None:
    """Raise 400 if bytes don't look like a supported image."""
    if len(raw) > _MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size is {_MAX_FILE_BYTES // 1_048_576} MB.")
    for magic, _ in _MAGIC.items():
        if raw[:len(magic)] == magic:
            return
    raise HTTPException(status_code=400, detail="Unsupported file type. Only JPEG, PNG, WebP, and GIF are accepted.")


class PredictResponse(BaseModel):
    cls: str = Field(serialization_alias="class")
    confidence: float

    model_config = ConfigDict(populate_by_name=True, serialize_by_alias=True)


# Fraction of pixels that must look plant-like before we even run the model.
_MIN_LEAF_PIXEL_RATIO = 0.08
# Model confidence below this is treated as "not a recognisable leaf".
_MIN_CONFIDENCE = 0.20


def _is_likely_leaf(img: Image.Image) -> bool:
    """
    Fast colour heuristic: checks whether enough pixels fall in ranges
    typical of plant leaves (green for healthy; yellow/brown for diseased).
    No extra dependencies — uses only numpy + PIL which are already loaded.
    """
    arr = np.array(img.convert("RGB"), dtype=np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    # Healthy leaves: green channel clearly dominant
    green = (g > 60) & (g > r * 1.05) & (g > b * 1.05)

    # Diseased / dry leaves: yellow-green or brown tones
    yellow_brown = (r > 80) & (g > 60) & (b < 130) & (r > b * 1.15) & (g > b * 1.05)

    ratio = float(np.sum(green | yellow_brown)) / (arr.shape[0] * arr.shape[1])
    return ratio >= _MIN_LEAF_PIXEL_RATIO


@app.get("/health")
def health():
    """Health check for Railway / load balancers."""
    return {"status": "ok"}


@app.post(
    "/predict",
    response_model=PredictResponse,
    dependencies=[Depends(_require_api_key)],
)
@limiter.limit(_rate_limit)
async def predict(request: Request, file: UploadFile = File(...)):  # noqa: ARG001
    # ── Filename / content-type sanity check ─────────────────────────────────
    filename = (file.filename or "").lower()
    allowed_exts = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
    if not any(filename.endswith(ext) for ext in allowed_exts):
        if not (file.content_type or "").startswith("image/"):
            raise HTTPException(status_code=400, detail="File must be an image (JPEG, PNG, WebP).")

    # ── Read and hard-limit file size ─────────────────────────────────────────
    raw = await file.read()
    _validate_image_bytes(raw)

    # ── Decode image ──────────────────────────────────────────────────────────
    try:
        img = Image.open(io.BytesIO(raw)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image. Please upload a valid image file.")

    # ── Dimension guard ───────────────────────────────────────────────────────
    w, h = img.size
    if w > _MAX_IMAGE_DIM or h > _MAX_IMAGE_DIM:
        raise HTTPException(
            status_code=400,
            detail=f"Image dimensions too large ({w}×{h}). Maximum is {_MAX_IMAGE_DIM}×{_MAX_IMAGE_DIM} pixels.",
        )

    # ── Leaf colour heuristic ─────────────────────────────────────────────────
    if not _is_likely_leaf(img):
        raise HTTPException(
            status_code=422,
            detail="No plant leaf detected. Please upload a clear photo of a leaf.",
        )

    # ── Model inference ───────────────────────────────────────────────────────
    arr = np.array(img.resize(config.IMG_SIZE), dtype=np.float32)
    scores = model.predict(arr[np.newaxis], verbose=0)[0]
    idx = int(np.argmax(scores))
    confidence = round(float(scores[idx]), 4)

    if confidence < _MIN_CONFIDENCE:
        raise HTTPException(
            status_code=422,
            detail="Image not recognised as a plant leaf. Please upload a clear, well-lit leaf photo.",
        )

    return PredictResponse(cls=config.CLASS_NAMES[idx], confidence=confidence)
