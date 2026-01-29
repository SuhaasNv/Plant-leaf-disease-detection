# Codebase Audit — Plant Disease Detection

**Date:** 2026-01-29  
**Scope:** All Python modules, notebooks, config, data pipeline, Streamlit app, README, and requirements.

---

## 1. Executive Summary

The project is a **Plant Disease Recognition** system with a TensorFlow CNN (38 classes), Streamlit app, config/data pipeline modules, and Jupyter notebooks for training and testing. The structure is clear, but there is **duplication** (class names, paths, preprocessing), **no .gitignore**, **unpinned dependencies**, and **README** still refers to model download. Recommendations below are ordered by priority: **Critical**, **High**, **Medium**, **Low**.

---

## 2. File-by-File Audit

### 2.1 `main.py` (Streamlit app)

| Area | Status | Notes |
|------|--------|--------|
| **Imports** | OK | Uses `io`, `os`, `typing`, `numpy`, `streamlit`, `tensorflow`, `PIL`. Type ignores for third-party packages are acceptable. |
| **Model loading** | OK | `@st.cache_resource` loads model once; local file only (no Google Drive). |
| **Image handling** | OK | Uploaded file read via `io.BytesIO` + PIL, resized to (128, 128), converted to RGB. Matches training input size. |
| **Preprocessing** | ⚠️ | App passes **raw pixels [0, 255]** to the model. This is correct **only if** the saved model’s first layer is `Rescaling(1/255)`. If you ever save a model without that layer, inference will be wrong. Align with `data_pipeline.get_preprocessing_layer()` or document assumption. |
| **Class names** | ⚠️ | **Hardcoded** 38-class list in `main.py`. Same list exists implicitly in dataset folder order and in Test notebook. Should live in **one place** (e.g. `config.CLASS_NAMES`) to avoid drift. |
| **Paths** | ⚠️ | `MODEL_PATH`, `image_path` for home are string literals. Consider `config.PROJECT_ROOT` for robustness when run from another directory. |
| **Input validation** | ⚠️ | No check on file type (e.g. allow only image extensions) or file size; very large uploads could cause memory issues. |
| **Error handling** | OK | Model missing and image load errors are caught and surfaced with `st.error`. |
| **`__main__`** | OK | Exits with clear message when run as `python main.py`. |

### 2.2 `config.py`

| Area | Status | Notes |
|------|--------|--------|
| **Structure** | OK | Single source for paths, image size, batch sizes, seed, augmentation, `ExperimentConfig`. |
| **CLASS_NAMES** | ❌ | Not defined. Class names are duplicated in `main.py` and inferred from directories in notebooks. **Recommendation:** Add `CLASS_NAMES: tuple[str, ...]` (or list) in `config.py` and use it in `main.py` and anywhere that maps indices to labels. |
| **Types** | OK | Uses `Path`, `Tuple`, `Dict`, `Any`, and a dataclass. |

### 2.3 `data_pipeline.py`

| Area | Status | Notes |
|------|--------|--------|
| **Seeds** | OK | `set_global_seed()` sets Python, NumPy, TensorFlow seeds. |
| **Preprocessing** | OK | `get_preprocessing_layer()` (Rescaling 1/255) and `get_data_augmentation_layer()` are well defined. |
| **Datasets** | OK | `build_training_dataset`, `build_validation_dataset`, `build_test_dataset` use `config` paths and sizes; prefetch and map are used correctly. |
| **Inference** | OK | `preprocess_single_image_for_inference(image_path: Path)` loads from disk, resizes, normalizes to [0,1]. **Not used by `main.py`** because the app receives an in-memory upload; consider a variant that accepts a numpy array or PIL Image for Streamlit. |
| **Experiment logging** | OK | `save_experiment_artifacts`, `build_experiment_run_id`, `ensure_experiments_dir` are clear and use JSON. |

### 2.4 `split_valid_to_test.py`

| Area | Status | Notes |
|------|--------|--------|
| **Paths** | ⚠️ | Defines its own `PROJECT_ROOT`, `DATA_DIR`, `TRAIN_DIR`, `VALID_DIR`, `TEST_DIR`. Duplicates `config.py`. **Recommendation:** Import from `config` and use `config.VALID_DIR`, `config.TEST_DIR`, etc. |
| **Logic** | OK | Moves a fraction of validation images per class to test; handles edge cases (e.g. 1 image). |
| **Reproducibility** | OK | `random.seed(42)` before shuffle. |

### 2.5 `requirements.txt`

| Area | Status | Notes |
|------|--------|--------|
| **Contents** | OK | Lists `numpy`, `Pillow`, `requests`, `streamlit`, `tensorflow`. |
| **`requests`** | ⚠️ | Not used anywhere after removal of Google Drive download. Can be removed to keep deps minimal. |
| **Pinning** | ❌ | No version pins. For reproducibility, pin major (or exact) versions, e.g. `tensorflow>=2.12,<3`, `streamlit>=1.28`, etc. |

### 2.6 Notebooks

| File | Status | Notes |
|------|--------|--------|
| **Train_plant_disease.ipynb** | ⚠️ | Two training paths: (1) cells using `config` + `data_pipeline` (train/val/test from pipeline), (2) cells using `image_dataset_from_directory` on `data/train`, `data/valid` directly. Same data, but two ways to build datasets; one path uses pipeline augmentation, the other may use model-internal augmentation. Prefer one pipeline (e.g. config + data_pipeline) for consistency. |
| **Test_plant_disease.ipynb** | ⚠️ | Uses `'test'` as directory in `image_dataset_from_directory` (should be `data/test` or `config.TEST_DIR` for consistency). Gets `class_name = validation_set.class_names` from the dataset; order must match training. |

### 2.7 Repo / Docs

| Item | Status | Notes |
|------|--------|--------|
| **.gitignore** | ❌ | Missing. Risk of committing `data/`, `venv/`, `__pycache__/`, `*.h5`, `experiments/*.json`, `.env`, IDE files. |
| **README.md** | ⚠️ | Says the model “is downloaded during the first execution”; app now uses only a **local** model. Update to: place `trained_plant_disease_model.h5` in the project root (or same dir as `main.py`). Also bump suggested Python to 3.9+ (3.6 is EOL). |

---

## 3. Recommendations Summary

### Critical

1. **Single source for class names**  
   - Add `CLASS_NAMES` (tuple or list of 38 strings) to `config.py`, in the **same order** as `image_dataset_from_directory` (alphabetical by class folder name).  
   - Use `config.CLASS_NAMES` in `main.py` and, if needed, in Test notebook so indices always match labels.

### High

2. **Add `.gitignore`**  
   - Ignore: `data/`, `venv/`, `.venv/`, `__pycache__/`, `*.pyc`, `*.h5`, `experiments/`, `.env`, `.idea/`, `.vscode/`, `*.ipynb_checkpoints/`, `*.egg-info/`, `dist/`, `build/`.

3. **Align inference preprocessing with training**  
   - Either: (a) Document in `main.py` that the saved model must include `Rescaling(1/255)` as the first layer, or (b) Use the same normalization as `data_pipeline` (e.g. divide by 255 in the app before `model.predict` if the saved model has no Rescaling).  
   - Optional: Add a `data_pipeline` helper that preprocesses an in-memory image (PIL or numpy) for inference and use it in `main.py`.

4. **README and model instructions**  
   - Update README: model is **not** downloaded; user must place `trained_plant_disease_model.h5` in the project root (or same directory as `main.py`).  
   - Recommend Python 3.9 or higher.

### Medium

5. **Remove unused dependency**  
   - Remove `requests` from `requirements.txt` if no other script uses it.

6. **Pin dependency versions**  
   - Add version constraints to `requirements.txt` (e.g. `tensorflow>=2.12`, `streamlit>=1.28`, `numpy>=1.24`, `Pillow>=9.0`) for reproducible installs.

7. **Use config in `split_valid_to_test.py`**  
   - Import `VALID_DIR`, `TEST_DIR` (and optionally others) from `config` instead of redefining paths.

8. **Unify training pipeline in notebook**  
   - In `Train_plant_disease.ipynb`, use either config + `data_pipeline` **or** raw `image_dataset_from_directory` consistently, and document which one is canonical.

### Low

9. **Streamlit upload limits**  
   - Add `st.file_uploader(..., type=["png", "jpg", "jpeg"])` and optionally check file size and show a warning for very large files.

10. **Paths in main.py**  
    - Use `config.PROJECT_ROOT` for `MODEL_PATH` and home image path so the app works when run from a different cwd (e.g. `streamlit run main.py` from project root).

11. **Test notebook path**  
    - Use `config.TEST_DIR` or `"data/test"` in `Test_plant_disease.ipynb` instead of `'test'` so it works from project root.

---

## 4. Security & Performance

- **Security:** No secrets in code; no unsafe `eval`/exec. File upload is not validated by type/size (see recommendation 9).  
- **Performance:** Model is loaded once via `@st.cache_resource`. Image is resized to 128×128 before prediction, which is appropriate. No obvious bottlenecks.

---

## 5. Next Steps

1. Implement **Critical** and **High** items (CLASS_NAMES in config + usage in main, .gitignore, README, optional preprocessing note/alignment).  
2. Apply **Medium** items (requirements cleanup, config in split script, notebook consistency) as you touch those files.  
3. Apply **Low** items when improving UX and robustness.
