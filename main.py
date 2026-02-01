"""
Streamlit UI for plant disease inference. Thin layer: ML logic (preprocessing, model)
delegates to config + data_pipeline assumptions; this file handles upload, display, and
result rendering only.
"""
import io
import os
from pathlib import Path
from typing import Optional

import numpy as np  # type: ignore[import-untyped]
import streamlit as st  # type: ignore[import-untyped]
import tensorflow as tf  # type: ignore[import-untyped]
from PIL import Image  # type: ignore[import-untyped]

import config

st.set_page_config(
    page_icon="Plant.png",
    page_title="Plant Disease Detection",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Preprocessing contract: canonical pipeline normalizes to [0,1] in the dataset; the saved
# model (from Train notebook cell 3) has no Rescaling layer, so inference must normalize
# input to [0,1] before predict. See model_prediction() below.

# Only Prev Models (no project-root fallback so we never load an old .h5 from root).
_PREV_MODEL = config.PROJECT_ROOT / "Prev Models" / "trained_plant_disease_model.h5"
_PREV_MODEL_CWD = Path.cwd() / "Prev Models" / "trained_plant_disease_model.h5"
# Fallback: downloaded model (e.g. from Google Drive on Streamlit Cloud).
_DOWNLOADED_MODEL = Path.cwd() / "trained_plant_disease_model.h5"


def _get_model_url_from_secrets() -> Optional[str]:
    """Return MODEL_URL from Streamlit secrets or env (for Streamlit Cloud / Google Drive)."""
    try:
        url = st.secrets.get("MODEL_URL", "")
        if url:
            return url
    except Exception:
        pass
    return os.environ.get("MODEL_URL") or None


def _extract_drive_id(url: str) -> Optional[str]:
    """Extract Google Drive file ID from share or uc link."""
    s = url.strip()
    if "/file/d/" in s:
        start = s.find("/file/d/") + 7
        end = s.find("/", start)
        if end == -1:
            end = s.find("?", start)
        if end == -1:
            end = len(s)
        return s[start:end]
    if "id=" in s:
        start = s.find("id=") + 3
        end = s.find("&", start)
        if end == -1:
            end = len(s)
        return s[start:end].strip()
    return None


def _download_model_to(path: Path, model_url: str) -> tuple[bool, str]:
    """Download model from URL (e.g. Google Drive) to path. Returns (success, error_message)."""
    try:
        import gdown
        path.parent.mkdir(parents=True, exist_ok=True)
        url = model_url.strip()
        if "drive.google.com" in url:
            file_id = _extract_drive_id(url)
            if file_id:
                gdown.download(id=file_id, output=str(path), quiet=True)
            else:
                gdown.download(url=url, output=str(path), quiet=True, fuzzy=True)
        else:
            gdown.download(url=url, output=str(path), quiet=True)
        return (path.exists(), "")
    except Exception as e:
        return (False, str(e))


def _resolve_model_path() -> Optional[str]:
    """Use Prev Models file if present; else download from MODEL_URL (Streamlit Cloud) if set."""
    if _PREV_MODEL.exists():
        return str(_PREV_MODEL)
    if _PREV_MODEL_CWD.exists():
        return str(_PREV_MODEL_CWD)
    if _DOWNLOADED_MODEL.exists():
        return str(_DOWNLOADED_MODEL)

    model_url = _get_model_url_from_secrets()
    if model_url:
        with st.spinner("Downloading model from Google Drive…"):
            ok, err = _download_model_to(_DOWNLOADED_MODEL, model_url)
            if ok:
                return str(_DOWNLOADED_MODEL)
        st.error(
            "Model download failed. Check: (1) MODEL_URL in Streamlit secrets is the full share link "
            "(e.g. https://drive.google.com/file/d/FILE_ID/view?usp=sharing). "
            "(2) File is shared so **Anyone with the link** can view. "
            "(3) For large files, use the direct link: https://drive.google.com/uc?id=FILE_ID"
        )
        if err:
            st.caption(f"Error: {err}")
    return None


@st.cache_resource
def load_model(path: str):
    """Load model once per path (Prev Models only); cache so we don't reload every click."""
    if not path or not os.path.exists(path):
        return None
    return tf.keras.models.load_model(path)


def model_prediction(test_image) -> Optional[int]:
    """
    Run inference on an uploaded image. Returns class index or None on error.

    Preprocessing must match training: resize to IMG_SIZE, normalize to [0,1].
    File type/size checks (in the uploader) reduce invalid inputs; we still guard
    on load errors. Label mapping uses config.CLASS_NAMES for consistency.
    """
    model_path = _resolve_model_path()
    if model_path is None:
        st.error(
            "Model file not found. Put trained_plant_disease_model.h5 in Prev Models, "
            "or set MODEL_URL in Streamlit secrets (e.g. Google Drive direct link) for deployment."
        )
        return None
    model = load_model(model_path)
    if model is None:
        st.error("Failed to load model.")
        return None
    try:
        test_image.seek(0)  # reset stream so each Predict click reads the same upload (otherwise read() returns empty after first time)
        img_bytes = test_image.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image = image.resize(config.IMG_SIZE)
        input_arr = np.array(image, dtype=np.float32)
        # Prev Models (95% run) was likely saved from the notebook's *other* model that has Rescaling(1/255) as first layer—it expects raw [0,255]. Canonical model has no Rescaling and expects [0,1].
        if not (model.layers and "rescaling" in model.layers[0].name.lower()):
            input_arr = input_arr / 255.0
        input_arr = np.array([input_arr])
    except Exception as e:
        st.error(f"Could not load image: {e}")
        return None
    predictions = model.predict(input_arr, verbose=0)
    return int(np.argmax(predictions))

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg" if os.path.exists("home_page.jpeg") else "Plant.png"
    st.image(image_path, use_container_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.

    ### Diseases We Can Detect
    Our model is trained to detect various plant diseases, including but not limited to:
    - **Apple Diseases:** Apple Scab, Black Rot, Cedar Apple Rust, Healthy
    - **Corn Diseases:** Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy
    - **Tomato Diseases:** Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites (Two-spotted Spider Mite), Target Spot, Yellow Leaf Curl Virus, Tomato Mosaic Virus, Healthy
    - **Potato Diseases:** Early Blight, Late Blight, Healthy
    - **Grape Diseases:** Black Rot, Esca (Black Measles), Leaf Blight (Isariopsis Leaf Spot), Healthy
    - **Peach Diseases:** Bacterial Spot, Healthy
    - **Strawberry Diseases:** Leaf Scorch, Healthy
    - **Pepper Diseases:** Bacterial Spot, Healthy
    - **Blueberry Diseases:** Healthy
    - **Soybean Diseases:** Healthy
    - **Raspberry Diseases:** Healthy
    - **Squash Diseases:** Powdery Mildew, Healthy
    """)



# About Page
elif app_mode == "About":
    st.header("About")
    st.markdown("""
    #### About Dataset
    This dataset consists of approximately 87,867 RGB images of healthy and diseased crop leaves, categorized into 38 different classes. 
    The dataset is divided as follows:
    - **Training Set:** 61,490 images (70%)
    - **Validation Set:** 13,164 images (15%)
    - **Test Set:** 13,213 images (15%)

    #### Project Team
    This project is developed by:

    - **Vijaya Suhaas Nadukooru**

    Our team is dedicated to creating an efficient and accurate plant disease recognition system to help in protecting crops and ensuring a healthier harvest.
    """)



# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    model_path = _resolve_model_path()
    if model_path:
        st.caption(f"Using model: **{model_path}**")
    else:
        st.warning(
            "No model file found. Add trained_plant_disease_model.h5 to Prev Models, "
            "or set MODEL_URL in Streamlit secrets for cloud deployment."
        )
    test_image = st.file_uploader("Choose an Image:")
    
    if test_image:
        st.image(test_image, use_container_width=True)
    
        # Predict button
        if st.button("Predict"):
            result_index = model_prediction(test_image)

            if result_index is not None:
                if 0 <= result_index < len(config.CLASS_NAMES):
                    label = config.CLASS_NAMES[result_index]
                    st.success(f"Model predicts: **{label}**")
                    st.caption(f"(class index {result_index})")
                else:
                    st.error(f"Invalid prediction index: {result_index}")
            else:
                st.error("Prediction failed, please check the model file.")
    
    # Add warning message
    # st.warning("⚠️ The model is currently under production and may make mistakes. Please use with caution.")


if __name__ == "__main__":
    import sys

    print("This is a Streamlit app. Run it with:", file=sys.stderr)
    print("  streamlit run main.py", file=sys.stderr)
    sys.exit(1)
