"""
Canonical data pipeline for the Plant Disease Recognition project.

This module is the single source of truth for preprocessing and dataset creation.
Training, validation, test, and inference all use the same normalization ([0,1])
and, for training only, the same augmentation. That design prevents data leakage
(train-only augmentation) and inference mismatch (same preprocessing at serve time).
"""

import os
import random
from datetime import datetime
from pathlib import Path
from typing import Dict, Any

import numpy as np
import tensorflow as tf

import config


# ---------- Reproducibility ----------

def set_global_seed(seed: int = config.SEED) -> None:
    """
    Set random seeds for Python, NumPy, and TensorFlow (ML lifecycle: all stages).

    Call once at training start. Ensures dataset shuffling, weight init, and
    numpy ops are consistent across runs; does not guarantee full determinism on all GPUs.
    """
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


# ---------- Preprocessing & augmentation layers ----------

def get_preprocessing_layer() -> tf.keras.Sequential:
    """
    Core preprocessing applied to train, validation, test, and inference (no augmentation).

    Rescaling(1/255) only. Shared by all dataset builders and by inference so
    training and serving see the same input distribution.
    """
    return tf.keras.Sequential(
        [
            tf.keras.layers.Rescaling(1.0 / 255.0),
        ],
        name="preprocessing",
    )


def get_data_augmentation_layer() -> tf.keras.Sequential:
    """
    Augmentation pipeline applied ONLY in training (not validation/test/inference).

    RandomFlip, RandomRotation, RandomZoom, RandomBrightness. Kept out of val/test
    to avoid data leakage and to keep metrics comparable; inference sees no augmentation.
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
    Internal: raw dataset from directory (one subdir per class). Resize to IMG_SIZE.
    Callers apply preprocessing (and optionally augmentation) via map().
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
    Training dataset (ML lifecycle: training only). Augmentation ON, then normalize [0,1].

    Order: raw load -> augment -> preprocess. Val/test use the same preprocess, no augment.
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
    Validation dataset (ML lifecycle: validation only). Augmentation OFF, normalize [0,1].

    No augmentation so validation metrics are stable and comparable across runs.
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
    Test dataset (ML lifecycle: testing only). Augmentation OFF, normalize [0,1].

    Same preprocessing as train/val; used for final evaluation and reporting.
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
    Inference-time preprocessing (ML lifecycle: inference). No augmentation, normalize [0,1].

    Same normalization as training so the model sees the same input distribution.
    Returns (1, H, W, C) batch. Streamlit app does equivalent logic in-process.
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
    Persist training history, metrics, and config snapshot to experiments/ (reproducibility).

    Called from the training notebook after fit and evaluate. Config snapshot allows
    exact hyperparameters to be recovered for the run.
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

