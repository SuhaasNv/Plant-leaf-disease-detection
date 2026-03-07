import json
import re

CLASS_NAMES = [
    "Apple___Apple_scab", "Apple___Black_rot", "Apple___Cedar_apple_rust", "Apple___healthy",
    "Blueberry___healthy", "Cherry_(including_sour)___Powdery_mildew", "Cherry_(including_sour)___healthy",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
    "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy", "Grape___Black_rot",
    "Grape___Esca_(Black_Measles)", "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)", "Grape___healthy",
    "Orange___Haunglongbing_(Citrus_greening)", "Peach___Bacterial_spot", "Peach___healthy",
    "Pepper,_bell___Bacterial_spot", "Pepper,_bell___healthy", "Potato___Early_blight",
    "Potato___Late_blight", "Potato___healthy", "Raspberry___healthy", "Soybean___healthy",
    "Squash___Powdery_mildew", "Strawberry___Leaf_scorch", "Strawberry___healthy",
    "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
    "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot", "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot", "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy"
]

def format_label(raw_str):
    s = raw_str.replace("___", " — ")
    s = s.replace("_", " ")
    # capitalize first letter of each word
    s = " ".join([word.capitalize() for word in s.split()])
    return s

data = {}
for c in CLASS_NAMES:
    if "healthy" in c.lower():
        data[c] = {
            "treatment": ["No treatment needed.", "Continue regular maintenance."],
            "prevention": ["Ensure proper watering and fertilization.", "Monitor regularly for pests and diseases."]
        }
    else:
        data[c] = {
            "treatment": ["Remove and destroy infected plant parts.", "Apply appropriate fungicide/pesticide if severe."],
            "prevention": ["Ensure adequate spacing for good airflow.", "Avoid overhead watering to keep leaves dry.", "Rotate crops annually."]
        }

# Customize a few specific ones
data["Tomato___Early_blight"] = {
    "treatment": ["Remove infected leaves", "Use copper fungicide"],
    "prevention": ["Avoid overhead watering", "Improve airflow"]
}

with open("api/treatments.json", "w") as f:
    json.dump(data, f, indent=4)
