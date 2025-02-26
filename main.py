import os
import requests
import tensorflow as tf
import numpy as np
import streamlit as st



st.set_page_config(
    page_icon="Plant.png",
    page_title="Plant Disease Detection",
    layout="wide",  # This ensures wide layout
    initial_sidebar_state="expanded"  # Makes sure sidebar is expanded by default
)

# Google Drive file ID and file path
FILE_ID = "1Z8oMCp1XR3sFq2iK034RCnTj_2RbqRKN"
MODEL_PATH = "trained_plant_disease_model.h5"
MODEL_URL = f"https://drive.google.com/uc?export=download&id={FILE_ID}"

# Function to download the model if not present
def download_model():
    if not os.path.exists(MODEL_PATH):
        st.info("Downloading the model, please wait...")
        response = requests.get(MODEL_URL, stream=True)
        if response.status_code == 200:
            with open(MODEL_PATH, 'wb') as file:
                for chunk in response.iter_content(chunk_size=8192):
                    file.write(chunk)
            st.success("Model downloaded successfully!")
        else:
            st.error("Failed to download the model. Please check the URL or model file.")
            return False
    return True

# TensorFlow Model Prediction
def model_prediction(test_image):
    # Ensure the model is downloaded
    if not download_model():
        return None

    # Load the model
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
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

    - **Bhaswanth Kumar Bonthala**

    Our team is dedicated to creating an efficient and accurate plant disease recognition system to help in protecting crops and ensuring a healthier harvest.
    """)



# Disease Recognition Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    
    if test_image:
        st.image(test_image, use_container_width=True)
    
        # Predict button
        if st.button("Predict"):
            result_index = model_prediction(test_image)

            if result_index is not None:
                # Class names for the predictions
                class_name = [
                    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
                    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
                    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
                    'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
                    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
                    'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 
                    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 
                    'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 
                    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 
                    'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 
                    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                    'Tomato___healthy'
                ]
                st.success(f"Model is predicting it's a {class_name[result_index]}")
            else:
                st.error("Prediction failed, please check the model file.")
    
    # Add warning message
    #st.warning("⚠️ The model is currently under production and may make mistakes. Please use with caution.")
