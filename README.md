# Plant Disease Recognition System

This repository contains the code for a Plant Disease Recognition System using TensorFlow and Streamlit. The system is capable of identifying diseases in various plant leaves through image analysis.

## Project Overview

The Plant Disease Recognition System aims to provide an efficient and accurate method for detecting diseases in plant leaves. This system uses:
*   A Convolutional Neural Network (CNN) built with TensorFlow for image classification.
*   Streamlit to create an interactive web application for easy user access.

The project includes:

*   **Training:** A Jupyter Notebook (`Train_plant_disease.ipynb`) for training the CNN model using a large dataset of plant leaf images.
*   **Testing:** A Jupyter Notebook (`Test_plant_disease.ipynb`) for testing the trained model's performance.
*   **Application:** A Python script (`main.py`) to run the Streamlit application for real-time disease detection using the trained model.
*   **Deployment:** Instructions for running the Streamlit application locally.
*   **Dataset:** The notebook mentions using a dataset of approximately 87,867 RGB images of healthy and diseased crop leaves, categorized into 38 different classes.
*   **Requirements:** A `requirements.txt` to install necessary python packages.

## Table of Contents

*   [Project Overview](#project-overview)
*   [Directory Structure](#directory-structure)
*   [Getting Started](#getting-started)
    *   [Prerequisites](#prerequisites)
    *   [Installation](#installation)
    *   [Running the Application](#running-the-application)
    *  [Downloading the Model](#downloading-the-model)
*   [Usage](#usage)
*   [Dataset](#dataset)
*   [Model Details](#model-details)
*   [Diseases Supported](#diseases-supported)

   
## Getting Started

### Prerequisites

Before you begin, ensure you have the following installed:

*   **Python 3.9 or higher:** You can download it from the [official Python website](https://www.python.org/downloads/).
*   **pip:** Python's package installer (usually included with Python).
*   **TensorFlow:** Used for training and using the CNN model. You can find installation instructions on the [TensorFlow website](https://www.tensorflow.org/install).
*   **Streamlit:** Used for creating the web application. Instructions are available on the [Streamlit website](https://streamlit.io/installation).

### Installation

1.  **Clone the repository:**

    ```bash
    git clone <repository_url>
    cd plant-disease-recognition
    ```
    (Replace `<repository_url>` with the actual URL of your repository).

2.  **Create a virtual environment (recommended):**

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Linux/macOS
    venv\Scripts\activate  # On Windows
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Model (local only)

The app uses a **local** trained model; it does not download the model. Place the file `trained_plant_disease_model.h5` in the same directory as `main.py` (project root). Train the model using `Train_plant_disease.ipynb` and save it, or copy an existing trained model there.

### Running the Application

To run the Streamlit app, execute the following command in your terminal:

```bash
streamlit run main.py
```

## Usage
--Open the application in your web browser.

--Use the navigation sidebar to select between "Home," "About," or "Disease Recognition."

--The Home page provides an overview of the project, its functionalities, and instructions on how to use the system.

--The About page gives details on the dataset and the development team.

On the Disease Recognition page:

--Upload an image of a plant leaf using the file uploader.

--Press the "Predict" button to analyze the image.

--The system will output the predicted disease name, if a disease is detected.


## Dataset
The dataset used in this project is a collection of approximately 87,867 RGB images of both healthy and diseased crop leaves. These images are categorized into 38 different classes. The dataset is split into:

--Training Set: 61,490 images (70%)

--Validation Set: 13,164 images (15%)

--Test Set: 13,213 images (15%)

## Model Details:
The core of this project is a Convolutional Neural Network (CNN) model built with TensorFlow. The architecture consists of the following:

Three convolutional blocks with relu activation functions.

Max pooling layers after each convolutional block.

A dropout layer to prevent overfitting.

A fully connected layer followed by an output layer with 38 neurons (corresponding to the 38 classes in our dataset) and a softmax activation function.

The model uses the Adam optimizer and categorical_crossentropy loss.

## Diseases Supported:
The system is trained to detect various plant diseases, including but not limited to:

Apple Diseases: Apple Scab, Black Rot, Cedar Apple Rust, Healthy

Corn Diseases: Cercospora Leaf Spot, Common Rust, Northern Leaf Blight, Healthy

Tomato Diseases: Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria Leaf Spot, Spider Mites (Two-spotted Spider Mite), Target Spot, Yellow Leaf Curl Virus, Tomato Mosaic Virus, Healthy

Potato Diseases: Early Blight, Late Blight, Healthy

Grape Diseases: Black Rot, Esca (Black Measles), Leaf Blight (Isariopsis Leaf Spot), Healthy

Peach Diseases: Bacterial Spot, Healthy

Strawberry Diseases: Leaf Scorch, Healthy

Pepper Diseases: Bacterial Spot, Healthy

Blueberry Diseases: Healthy

Soybean Diseases: Healthy

Raspberry Diseases: Healthy

Squash Diseases: Powdery Mildew, Healthy

## Contributing
Contributions to the project are welcome! To contribute:

Fork the repository.

Create a new branch for your feature or bug fix.

Make your changes and commit them.

Push your changes to your fork.

Submit a pull request.
