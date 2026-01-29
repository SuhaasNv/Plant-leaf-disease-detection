# 🌿 Plant Disease Recognition System

A production-ready deep learning system for identifying plant diseases from leaf images using Convolutional Neural Networks (CNN). This project provides a complete pipeline from data preprocessing to model training, evaluation, and deployment via a Streamlit web application.

**Accuracy:** ~95% on 38 disease classes across 14 crop types

---

## 📋 Table of Contents

- [Features](#-features)
- [Project Overview](#-project-overview)
- [Architecture](#-architecture)
- [Directory Structure](#-directory-structure)
- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
  - [Training the Model](#training-the-model)
  - [Running the Web App](#running-the-web-app)
  - [Testing the Model](#testing-the-model)
- [Dataset](#-dataset)
- [Model Architecture](#-model-architecture)
- [Supported Diseases](#-supported-diseases)
- [Project Structure](#-project-structure)
- [Reproducibility](#-reproducibility)
- [Performance](#-performance)
- [Contributing](#-contributing)
- [Author](#-author)
- [License](#-license)

---

## ✨ Features

- **38 Disease Classes:** Detects diseases across 14 crop types (Apple, Corn, Tomato, Potato, Grape, Peach, Pepper, Strawberry, Blueberry, Soybean, Raspberry, Squash, Cherry, Orange)
- **High Accuracy:** ~95% test accuracy on a balanced dataset
- **Production-Ready Pipeline:** Centralized configuration and data pipeline for consistent preprocessing
- **Reproducible Training:** Seed control, experiment logging, and config snapshots
- **User-Friendly Interface:** Modern Streamlit web app with intuitive UI
- **Fast Inference:** Model caching for instant predictions
- **Comprehensive Testing:** Evaluation notebooks with confusion matrices and classification reports

---

## 🎯 Project Overview

This system uses a CNN to classify plant leaf images into 38 categories (healthy and diseased states). The project follows best practices for ML engineering:

- **Single Source of Truth:** All preprocessing and augmentation handled by `data_pipeline.py`
- **Centralized Configuration:** Hyperparameters, paths, and class names in `config.py`
- **Modular Design:** Separate modules for data, config, and application logic
- **Experiment Tracking:** Automatic logging of training history and metrics

---

## 🏗️ Architecture

```
┌─────────────────┐
│   Data Pipeline │  ← Single source of truth for preprocessing
│ (data_pipeline) │     - Normalization [0,1]
└────────┬────────┘     - Augmentation (train only)
         │
         ▼
┌─────────────────┐
│  Training       │  ← Train_plant_disease.ipynb
│  (Notebook)     │     - Uses canonical pipeline
└────────┬────────┘     - Saves model + experiment logs
         │
         ▼
┌─────────────────┐
│  Saved Model    │  ← trained_plant_disease_model.h5
│  (.h5 format)   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Streamlit App  │  ← main.py
│  (Inference)    │     - Loads model once (cached)
└─────────────────┘     - Normalizes input to [0,1]
```

---

## 📁 Directory Structure

```
Plant-leaf-disease-detection/
├── config.py                 # Centralized configuration
├── data_pipeline.py          # Data preprocessing & augmentation
├── main.py                   # Streamlit web application
├── split_valid_to_test.py    # Utility to split validation/test sets
├── Train_plant_disease.ipynb # Training notebook (canonical pipeline)
├── Test_plant_disease.ipynb  # Evaluation & testing notebook
├── requirements.txt          # Python dependencies
├── .gitignore               # Git ignore rules
├── AUDIT.md                 # Codebase audit report
├── README.md                # This file
│
├── data/                    # Dataset (not in repo)
│   ├── train/               # Training images (70%)
│   ├── valid/               # Validation images (15%)
│   └── test/                # Test images (15%)
│
├── experiments/             # Experiment logs (auto-generated)
│   ├── *_history.json       # Training history
│   └── *_metadata.json      # Metrics & config snapshots
│
└── trained_plant_disease_model.h5  # Saved model (after training)
```

---

## 💻 Installation

### Prerequisites

- **Python 3.9+** ([Download](https://www.python.org/downloads/))
- **pip** (included with Python)
- **Git** (optional, for cloning)

### Step-by-Step Setup

1. **Clone or download the repository:**
   ```bash
   git clone <repository_url>
   cd Plant-leaf-disease-detection
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   
   # Activate virtual environment
   # On macOS/Linux:
   source venv/bin/activate
   # On Windows:
   venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Prepare the dataset:**
   - Organize your images in `data/train/`, `data/valid/`, and `data/test/` directories
   - Each subdirectory should be named after the class (e.g., `data/train/Apple___Apple_scab/`)
   - See [Dataset](#-dataset) section for details

---

## 🚀 Quick Start

### Running the Web App (Inference Only)

If you already have a trained model:

1. Place `trained_plant_disease_model.h5` in the project root
2. Run the Streamlit app:
   ```bash
   streamlit run main.py
   ```
3. Open your browser to the URL shown (typically `http://localhost:8501`)
4. Navigate to **Disease Recognition** and upload a leaf image

### Training Your Own Model

See [Training the Model](#training-the-model) section below.

---

## 📖 Usage

### Training the Model

1. **Open the training notebook:**
   ```bash
   jupyter notebook Train_plant_disease.ipynb
   ```

2. **Run cells in order:**
   - **Cell 1:** Imports
   - **Cell 2:** Build datasets using canonical pipeline (`train_ds`, `val_ds`, `test_ds`)
   - **Cell 3:** Define CNN architecture
   - **Cell 4:** Train with early stopping and learning rate scheduling
   - **Cell 5:** Evaluate and save experiment artifacts

3. **Save the model:**
   - After training, run the save cell to create `trained_plant_disease_model.h5`

4. **View experiment logs:**
   - Check `experiments/` for training history and metrics JSON files

**Note:** The notebook uses the **canonical pipeline** (`config.py` + `data_pipeline.py`) as the single source of truth for preprocessing. This ensures consistency between training and inference.

### Running the Web App

1. **Start the application:**
   ```bash
   streamlit run main.py
   ```

2. **Use the interface:**
   - **Home:** Overview and instructions
   - **About:** Dataset and team information
   - **Disease Recognition:** Upload an image and get predictions

3. **Upload requirements:**
   - Supported formats: PNG, JPG, JPEG
   - Recommended size: < 50 MB (warnings shown for larger files)
   - Image is automatically resized to 128×128 pixels

### Testing the Model

1. **Open the test notebook:**
   ```bash
   jupyter notebook Test_plant_disease.ipynb
   ```

2. **Run evaluation cells:**
   - Load the trained model
   - Evaluate on test set
   - Generate confusion matrix and classification report

---

## 📊 Dataset

- **Total Images:** ~87,867 RGB images
- **Classes:** 38 (healthy + diseased states)
- **Split:**
  - Training: 61,490 images (70%)
  - Validation: 13,164 images (15%)
  - Test: 13,213 images (15%)

**Dataset Structure:**
```
data/
├── train/
│   ├── Apple___Apple_scab/
│   ├── Apple___Black_rot/
│   ├── ...
├── valid/
│   └── (same class structure)
└── test/
    └── (same class structure)
```

**Preprocessing:**
- Images resized to 128×128 pixels
- Normalized to [0, 1] range
- Training augmentation: random flips, rotations, zoom, brightness adjustments

---

## 🧠 Model Architecture

The CNN consists of:

1. **Convolutional Blocks (3):**
   - Conv2D layers (32, 64, 128 filters)
   - MaxPooling2D after each block
   - ReLU activation

2. **Regularization:**
   - Dropout (0.3) after convolutional layers

3. **Dense Layers:**
   - Flatten layer
   - Dense(256, ReLU)
   - Dense(38, Softmax) - output layer

**Training Configuration:**
- Optimizer: Adam (learning rate: 1e-4)
- Loss: Categorical cross-entropy
- Callbacks: Early stopping, learning rate reduction
- Epochs: Up to 20 (with early stopping)

**Input:** 128×128×3 RGB images, normalized to [0, 1]  
**Output:** 38-class probability distribution

---

## 🌾 Supported Diseases

The model detects diseases across **14 crop types**:

| Crop | Diseases Detected |
|------|-------------------|
| **Apple** | Apple Scab, Black Rot, Cedar Apple Rust, Healthy |
| **Corn** | Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy |
| **Tomato** | Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites, Target Spot, Yellow Leaf Curl Virus, Mosaic Virus, Healthy |
| **Potato** | Early Blight, Late Blight, Healthy |
| **Grape** | Black Rot, Esca (Black Measles), Leaf Blight, Healthy |
| **Peach** | Bacterial Spot, Healthy |
| **Pepper** | Bacterial Spot, Healthy |
| **Strawberry** | Leaf Scorch, Healthy |
| **Squash** | Powdery Mildew, Healthy |
| **Blueberry, Cherry, Orange, Raspberry, Soybean** | Healthy (and disease states where applicable) |

**Total: 38 classes** (see `config.CLASS_NAMES` for complete list)

---

## 🔧 Project Structure

### Core Modules

**`config.py`**
- Centralized hyperparameters (image size, batch size, learning rate)
- Path definitions (data directories, project root)
- Class names tuple (38 classes in alphabetical order)
- Data augmentation configuration
- Experiment config dataclass for reproducibility

**`data_pipeline.py`**
- `set_global_seed()`: Reproducibility control
- `build_training_dataset()`: Training set with augmentation
- `build_validation_dataset()`: Validation set (no augmentation)
- `build_test_dataset()`: Test set (no augmentation)
- `save_experiment_artifacts()`: Log training history and metrics

**`main.py`**
- Streamlit web application
- Model loading with caching (`@st.cache_resource`)
- Image preprocessing for inference
- UI with Home, About, and Disease Recognition pages

### Notebooks

**`Train_plant_disease.ipynb`**
- Uses canonical pipeline (cells 1-5)
- Model definition, training, evaluation
- Experiment logging
- Model saving

**`Test_plant_disease.ipynb`**
- Model evaluation on test set
- Confusion matrix visualization
- Classification report generation

---

## 🔬 Reproducibility

The project ensures reproducibility through:

1. **Seed Control:**
   - Python, NumPy, TensorFlow seeds set via `data_pipeline.set_global_seed()`
   - Dataset shuffling uses the same seed

2. **Configuration Snapshots:**
   - Each training run saves hyperparameters in `experiments/*_metadata.json`
   - Includes image size, batch size, learning rate, augmentation settings

3. **Experiment Logging:**
   - Training history saved as JSON
   - Final metrics (train/val/test) recorded
   - Timestamped run IDs for tracking

4. **Canonical Pipeline:**
   - Single source of truth for preprocessing
   - Training and inference use the same normalization

**To reproduce results:**
- Use the same seed (default: 42)
- Use the same config values
- Run cells in `Train_plant_disease.ipynb` sequentially

---

## 📈 Performance

**Model Performance (on test set):**
- **Accuracy:** ~95%
- **Training Accuracy:** ~95%
- **Validation Accuracy:** ~95%

**Training Details:**
- Training time: ~6-8 hours (on CPU/GPU depending on hardware)
- Model size: ~17M parameters
- Inference time: < 1 second per image

**Note:** Performance may vary based on hardware and dataset quality.

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. **Fork the repository**
2. **Create a feature branch:**
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. **Make your changes:**
   - Follow existing code style
   - Add comments for complex logic
   - Update documentation as needed
4. **Test your changes:**
   - Ensure training notebook runs successfully
   - Test the Streamlit app
   - Verify no linter errors
5. **Commit and push:**
   ```bash
   git commit -m "Add: description of changes"
   git push origin feature/your-feature-name
   ```
6. **Submit a pull request**

**Guidelines:**
- Keep changes focused and well-documented
- Maintain backward compatibility where possible
- Update `AUDIT.md` if making architectural changes

---

## 👤 Author

**Vijaya Suhaas Nadukooru**

This project was developed as part of a portfolio to demonstrate end-to-end ML engineering capabilities, from data preprocessing to model deployment.

---

## 📄 License

This project is open source and available for educational and research purposes. Please ensure compliance with dataset licenses if using external datasets.

---

## 🙏 Acknowledgments

- Dataset: Plant Village dataset (or similar public dataset)
- TensorFlow team for the deep learning framework
- Streamlit for the web application framework
- Open source community for tools and libraries

---

## 📝 Notes

- **Model File:** The trained model (`trained_plant_disease_model.h5`) is not included in the repository due to size. Train your own model using the provided notebook.
- **Data:** The dataset is not included. Organize your images according to the structure described in the [Dataset](#-dataset) section.
- **GPU:** Training benefits from GPU acceleration but works on CPU (slower).
- **Production:** For production deployment, consider:
  - Model optimization (quantization, pruning)
  - API deployment (FastAPI, Flask)
  - Containerization (Docker)
  - Cloud deployment (AWS, GCP, Azure)

---

## 🔗 Related Resources

- [TensorFlow Documentation](https://www.tensorflow.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Plant Disease Detection Research](https://www.kaggle.com/datasets/abdallahalidev/plantvillage-dataset)

---

**Last Updated:** January 2026  
**Version:** 1.0.0
