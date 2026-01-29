"""
Data and preprocessing pipeline utilities for the Plant Disease Recognition project.

This module centralizes:
- Reproducibility controls (random seeds)
- Training / validation / test dataset creation
- Preprocessing (resize + normalization)
- Data augmentation for training
- Inference‑time preprocessing for single images

The goal is to have a single source of truth for how data is handled
so that the training notebooks, evaluation notebooks, and deployed
Streamlit app all behave consistently.
"""

import os
import random
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any

import numpy as np
import tensorflow as tf

import config


# ---------- Reproducibility ----------

def set_global_seed(seed: int = config.SEED) -> None:
    """
    Set random seeds for Python, NumPy, and TensorFlow to improve reproducibility.

    This does not guarantee perfect determinism across all platforms / GPUs,
    but it ensures that:
    - Dataset shuffling
    - Weight initialization
    - Numpy operations
    will be consistent between runs with the same seed.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------- Preprocessing & augmentation layers ----------

def get_preprocessing_layer() -> tf.keras.Sequential:
    """
    Return the core preprocessing layer applied to ALL images
    (train, validation, test, and inference).

    Steps:
    1. Convert raw pixel values from [0, 255] to floating point.
    2. Normalize pixel intensities to [0, 1] via Rescaling(1./255).
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(1.0 / 255.0),
        ],
        name="preprocessing",
    )


def get_data_augmentation_layer() -> tf.keras.Sequential:
    """
    Return the data augmentation pipeline applied ONLY to training images.

    Augmentations:
    - RandomFlip: horizontal and vertical flips to simulate different orientations.
    - RandomRotation: small rotations to improve rotational invariance.
    - RandomZoom: zoom in/out to simulate distance to camera.
    - RandomBrightness: adjust brightness to simulate lighting changes.

    All augmentation parameters are controlled from config.DATA_AUGMENTATION_CONFIG.
    """
    aug_cfg = config.DATA_AUGMENTATION_CONFIG
    return tf.keras.Sequential(
        [
            tf.keras.layers.RandomFlip(aug_cfg["random_flip_mode"]),
            tf.keras.layers.RandomRotation(aug_cfg["random_rotation_factor"]),
            tf.keras.layers.RandomZoom(
                height_factor=aug_cfg["random_zoom_height_factor"],
                width_factor=aug_cfg["random_zoom_width_factor"],
            ),
            tf.keras.layers.RandomBrightness(aug_cfg["random_brightness_factor"]),
        ],
        name="data_augmentation",
    )


# ---------- Dataset builders ----------

def _base_dataset_from_directory(
    directory: Path,
    batch_size: int,
    shuffle: bool,
) -> tf.data.Dataset:
    """
    Internal helper to create a raw tf.data.Dataset from a directory of images.

    It delegates to tf.keras.utils.image_dataset_from_directory, which:
    - Reads images from subdirectories (one subdirectory per class).
    - Resizes them to config.IMG_SIZE.
    - Returns (image_batch, one_hot_labels).
    """
    return tf.keras.utils.image_dataset_from_directory(
        directory,
        labels="inferred",
        label_mode="categorical",
        color_mode="rgb",
        batch_size=batch_size,
        image_size=config.IMG_SIZE,
        shuffle=shuffle,
        interpolation="bilinear",
        seed=config.SEED,
    )


def build_training_dataset() -> tf.data.Dataset:
    """
    Build the full TRAINING pipeline:
    - Load images from data/train.
    - Shuffle and batch.
    - Apply data augmentation.
    - Apply normalization to [0, 1].
    - Enable prefetching for performance.
    """
    raw_ds = _base_dataset_from_directory(
        config.TRAIN_DIR,
        batch_size=config.BATCH_SIZE_TRAIN,
        shuffle=True,
    )

    aug = get_data_augmentation_layer()
    preprocess = get_preprocessing_layer()

    def _map_fn(images, labels):
        images = aug(images, training=True)
        images = preprocess(images, training=False)
        return images, labels

    ds = raw_ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def build_validation_dataset() -> tf.data.Dataset:
    """
    Build the VALIDATION pipeline:
    - Load images from data/valid.
    - Do NOT apply augmentation (we want a stable validation signal).
    - Apply normalization to [0, 1].
    - Enable prefetching for performance.
    """
    raw_ds = _base_dataset_from_directory(
        config.VALID_DIR,
        batch_size=config.BATCH_SIZE_EVAL,
        shuffle=False,
    )

    preprocess = get_preprocessing_layer()

    def _map_fn(images, labels):
        images = preprocess(images, training=False)
        return images, labels

    ds = raw_ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


def build_test_dataset() -> tf.data.Dataset:
    """
    Build the TEST pipeline:
    - Load images from data/test.
    - No augmentation.
    - Apply normalization to [0, 1].
    - Prefetch for performance.
    """
    raw_ds = _base_dataset_from_directory(
        config.TEST_DIR,
        batch_size=1,
        shuffle=False,
    )

    preprocess = get_preprocessing_layer()

    def _map_fn(images, labels):
        images = preprocess(images, training=False)
        return images, labels

    ds = raw_ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    return ds


# ---------- Inference‑time preprocessing ----------

def preprocess_single_image_for_inference(image_path: Path) -> tf.Tensor:
    """
    Preprocess a single image from disk for inference.

    Steps:
    1. Load the image from `image_path`.
    2. Resize to 128x128 (config.IMG_SIZE).
    3. Convert to float32 and add batch dimension.
    4. Normalize to [0, 1] using the same preprocessing layer used in training.

    Returns:
        A 4D tensor of shape (1, H, W, C) ready to pass to model.predict().
    """
    image = tf.keras.preprocessing.image.load_img(
        image_path,
        target_size=config.IMG_SIZE,
    )
    arr = tf.keras.preprocessing.image.img_to_array(image)
    arr = tf.cast(arr, tf.float32)
    arr = tf.expand_dims(arr, axis=0)  # shape: (1, H, W, C)

    preprocess = get_preprocessing_layer()
    arr = preprocess(arr, training=False)
    return arr


# ---------- Experiment logging ----------

def ensure_experiments_dir() -> Path:
    """
    Ensure that an 'experiments' directory exists at the project root.
    This is where we store:
    - Training history
    - Final metrics
    - Model metadata
    """
    exp_dir = config.PROJECT_ROOT / "experiments"
    exp_dir.mkdir(parents=True, exist_ok=True)
    return exp_dir


def build_experiment_run_id(prefix: str = "cnn_baseline") -> str:
    """
    Build a human‑readable, time‑stamped run ID, e.g.:
        cnn_baseline_2026-01-28_15-30-12
    """
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return f"{prefix}_{ts}"


def save_experiment_artifacts(
    run_id: str,
    history: tf.keras.callbacks.History,
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float] | None,
    experiment_config: config.ExperimentConfig | None = None,
) -> None:
    """
    Persist training history, metrics, and metadata to disk as JSON.

    This function is designed to be called from the training notebook
    after training and evaluation have finished.

    Args:
        run_id:         Unique identifier for this run (e.g. from build_experiment_run_id()).
        history:        Keras History object returned by model.fit().
        train_metrics:  Dict with metrics on the training set (e.g. {'accuracy': ..., 'loss': ...}).
        val_metrics:    Dict with metrics on the validation set.
        test_metrics:   Dict with metrics on the test set, or None if not evaluated.
        experiment_config: Optional ExperimentConfig snapshot; if None, DEFAULT_EXPERIMENT_CONFIG is used.
    """
    import json

    exp_dir = ensure_experiments_dir()

    # 1. Training history
    history_path = exp_dir / f"{run_id}_history.json"
    with history_path.open("w", encoding="utf-8") as f:
        json.dump(history.history, f, indent=2)

    # 2. Metrics & metadata
    meta: Dict[str, Any] = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "train_metrics": train_metrics,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }

    cfg = experiment_config or config.DEFAULT_EXPERIMENT_CONFIG
    meta["experiment_config"] = cfg.as_dict()

    meta_path = exp_dir / f"{run_id}_metadata.json"
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

