"""
Centralized configuration for the Plant Disease Recognition project.

This module defines:
- Data paths
- Image size & batch sizes
- Random seeds for reproducibility
- Model & training hyperparameters
- Data augmentation hyperparameters

Import this in notebooks / scripts instead of hard‑coding values.
"""

from pathlib import Path
from dataclasses import dataclass, asdict, field
from typing import Tuple, Dict, Any


# ---------- Paths ----------

PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
TRAIN_DIR: Path = DATA_DIR / "train"
VALID_DIR: Path = DATA_DIR / "valid"
TEST_DIR: Path = DATA_DIR / "test"


# ---------- Core hyperparameters ----------

IMG_SIZE: Tuple[int, int] = (128, 128)
NUM_CHANNELS: int = 3
NUM_CLASSES: int = 38

# Class names in the same order as image_dataset_from_directory (alphabetical by folder name).
# Used by main.py and should match training/test dataset label indices.
CLASS_NAMES: Tuple[str, ...] = (
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
)

BATCH_SIZE_TRAIN: int = 32
BATCH_SIZE_EVAL: int = 32

EPOCHS: int = 20
LEARNING_RATE: float = 1e-4


# ---------- Reproducibility ----------

SEED: int = 42  # Used for Python, NumPy, TensorFlow, and dataset shuffling


# ---------- Data augmentation hyperparameters ----------

DATA_AUGMENTATION_CONFIG: Dict[str, Any] = {
    "random_flip_mode": "horizontal_and_vertical",
    "random_rotation_factor": 0.1,   # ~ +/- 10 degrees
    "random_zoom_height_factor": 0.1,
    "random_zoom_width_factor": 0.1,
    "random_brightness_factor": 0.1,
}


@dataclass
class ExperimentConfig:
    """
    Lightweight snapshot of the key hyperparameters and settings
    used for a given training run. This is what we persist alongside
    model checkpoints for reproducibility.
    """

    img_size: Tuple[int, int] = IMG_SIZE
    num_channels: int = NUM_CHANNELS
    num_classes: int = NUM_CLASSES
    batch_size_train: int = BATCH_SIZE_TRAIN
    batch_size_eval: int = BATCH_SIZE_EVAL
    epochs: int = EPOCHS
    learning_rate: float = LEARNING_RATE
    seed: int = SEED
    # Use default_factory so each instance gets its own copy of the dict.
    data_augmentation: Dict[str, Any] = field(
        default_factory=lambda: dict(DATA_AUGMENTATION_CONFIG)
    )

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-serializable dictionary representation."""
        return asdict(self)


DEFAULT_EXPERIMENT_CONFIG = ExperimentConfig()

