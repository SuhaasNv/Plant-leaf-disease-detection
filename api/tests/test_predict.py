import os
import pytest
from unittest.mock import patch, MagicMock
import numpy as np
import config
from fastapi import HTTPException

# Test prediction confidence boundaries
confidence_cases = [
    # (score, expected_status, expected_error_msg)
    (0.95, 200, None),
    (0.85, 200, None),
    (0.60, 200, None),
    (0.45, 200, None),
    (0.25, 200, None),
    (0.21, 200, None),
    (0.19, 422, "Image not recognised as a plant leaf"),
    (0.15, 422, "Image not recognised as a plant leaf"),
    (0.10, 422, "Image not recognised as a plant leaf"),
    (0.01, 422, "Image not recognised as a plant leaf"),
]

@pytest.mark.parametrize("score, expected_status, expected_error", confidence_cases)
def test_predict_confidence_thresholds(client, generate_image_bytes, mock_keras_and_tflite, score, expected_status, expected_error):
    img = generate_image_bytes(color="green")
    files = {"file": ("leaf.jpg", img, "image/jpeg")}
    
    # Mock prediction output
    dummy_scores = np.zeros((1, 38), dtype=np.float32)
    dummy_scores[0, 0] = score
    mock_keras_and_tflite.predict.return_value = dummy_scores
    
    response = client.post("/predict", files=files)
    assert response.status_code == expected_status
    if expected_error:
        assert expected_error in response.json()["detail"]

# Test class name mapping for all 38 classes under 5 different confidence values (190 test cases)
@pytest.mark.parametrize("class_idx", range(38))
@pytest.mark.parametrize("confidence", [1.0, 0.85, 0.70, 0.50, 0.30])
def test_predict_all_classes_mapping(client, generate_image_bytes, mock_keras_and_tflite, class_idx, confidence):
    img = generate_image_bytes(color="green")
    files = {"file": ("leaf.jpg", img, "image/jpeg")}
    
    # Mock model to return the specific confidence for this specific class index
    dummy_scores = np.zeros((1, 38), dtype=np.float32)
    dummy_scores[0, class_idx] = confidence
    mock_keras_and_tflite.predict.return_value = dummy_scores
    
    expected_class = config.CLASS_NAMES[class_idx]
    
    response = client.post("/predict", files=files)
    assert response.status_code == 200
    
    data = response.json()
    predictions = data["predictions"]
    
    assert predictions[0]["label"] == expected_class
    assert predictions[0]["confidence"] == confidence
    assert isinstance(predictions[0]["treatment"], list)
    assert isinstance(predictions[0]["prevention"], list)

# Test model loader status handling
def test_predict_models_unavailable(client, generate_image_bytes):
    img = generate_image_bytes(color="green")
    files = {"file": ("leaf.jpg", img, "image/jpeg")}
    
    # Temporarily set both model references to None
    with patch("api.main.model", None), patch("api.main.tflite_model", None):
        response = client.post("/predict", files=files)
        assert response.status_code == 503
        assert "Prediction models are currently unavailable" in response.json()["detail"]

def test_predict_tflite_fallback(client, generate_image_bytes):
    img = generate_image_bytes(color="green")
    files = {"file": ("leaf.jpg", img, "image/jpeg")}
    
    mock_interpreter = MagicMock()
    mock_interpreter.get_input_details.return_value = [{'index': 0}]
    mock_interpreter.get_output_details.return_value = [{'index': 0}]
    
    dummy_scores = np.zeros((1, 38), dtype=np.float32)
    dummy_scores[0, 0] = 0.98
    mock_interpreter.get_tensor.return_value = dummy_scores
    
    # Set model=None, tflite_model=interpreter
    with patch("api.main.model", None), patch("api.main.tflite_model", mock_interpreter):
        response = client.post("/predict", files=files)
        assert response.status_code == 200
        data = response.json()
        assert data["predictions"][0]["label"] == config.CLASS_NAMES[0]
        assert data["predictions"][0]["confidence"] == 0.98
        mock_interpreter.set_tensor.assert_called_once()
        mock_interpreter.invoke.assert_called_once()
