"""
Streamlit UI for plant disease inference. Thin layer: ML logic (preprocessing, model)
delegates to config + data_pipeline assumptions; this file handles upload, display, and
result rendering only.
"""
import io
import os
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

MODEL_PATH = "trained_plant_disease_model.h5"

# Preprocessing contract: canonical pipeline normalizes to [0,1] in the dataset; the saved
# model (from Train notebook cell 3) has no Rescaling layer, so inference must normalize
# input to [0,1] before predict. See model_prediction() below.


@st.cache_resource
def load_model():
    """Load model once per session; @st.cache_resource avoids reload on every prediction."""
    if not os.path.exists(MODEL_PATH):
        return None
    return tf.keras.models.load_model(MODEL_PATH)


def model_prediction(test_image) -> Optional[int]:
    """
    Run inference on an uploaded image. Returns class index or None on error.

    Preprocessing must match training: resize to IMG_SIZE, normalize to [0,1].
    File type/size checks (in the uploader) reduce invalid inputs; we still guard
    on load errors. Label mapping uses config.CLASS_NAMES for consistency.
    """
    model = load_model()
    if model is None:
        st.error(f"Model file not found: {MODEL_PATH}. Place the trained model in this directory.")
        return None
    try:
        img_bytes = test_image.read()
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        image = image.resize((128, 128))
        input_arr = np.array(image, dtype=np.float32) / 255.0  # match canonical pipeline [0,1]
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
    # Input validation (e.g. accepted types, max size) rejects bad uploads early and avoids
    # unnecessary model loads; UI remains thin and delegates all ML to pipeline contract.
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
