import io
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

# Add parent directory of api to sys.path so config can be imported
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

@pytest.fixture(autouse=True)
def mock_keras_and_tflite():
    """Mock the model loading and prediction in main.py to run in isolation."""
    mock_model = MagicMock()
    # Mock model.predict to return dummy probabilities (38 classes)
    dummy_scores = np.zeros((1, 38), dtype=np.float32)
    dummy_scores[0, 0] = 0.95  # Class 0 has high confidence by default
    mock_model.predict.return_value = dummy_scores
    
    # Mock layers
    mock_layer = MagicMock()
    mock_layer.scale = 1.0
    mock_model.layers = [mock_layer]

    mock_interpreter = MagicMock()
    mock_interpreter.get_input_details.return_value = [{'index': 0}]
    mock_interpreter.get_output_details.return_value = [{'index': 0}]
    mock_interpreter.get_tensor.return_value = dummy_scores

    with patch("tensorflow.keras.models.load_model", return_value=mock_model, create=True), \
         patch("tensorflow.lite.Interpreter", return_value=mock_interpreter, create=True), \
         patch("api.main.model", mock_model), \
         patch("api.main.tflite_model", None):
        yield mock_model

@pytest.fixture(autouse=True)
def clean_environment():
    """Temporarily mock env vars to ensure clean test state."""
    old_env = dict(os.environ)
    os.environ["API_KEY"] = ""
    os.environ["RATE_LIMIT"] = "1000/minute"  # Disable rate limit for tests
    os.environ["GEMINI_API_KEY"] = ""
    os.environ["OPENAI_API_KEY"] = ""
    yield
    os.environ.clear()
    os.environ.update(old_env)

@pytest.fixture
def client():
    # Force reload main.py to respect environment changes and mocks
    import api.main as main
    main.limiter.enabled = False
    with TestClient(main.app) as c:
        yield c

@pytest.fixture
def generate_image_bytes():
    """Helper to generate mock image bytes with specified properties."""
    def _generate(width=150, height=150, color="green", low_quality=False):
        # Create an image using PIL
        if color == "green":
            # Healthy leaf green: RGB(80, 180, 80)
            img = Image.new("RGB", (width, height), (80, 180, 80))
        elif color == "yellow_brown":
            # Diseased yellow-brown: RGB(150, 100, 40)
            img = Image.new("RGB", (width, height), (150, 100, 40))
        elif color == "blue":
            # Non-plant blue color: RGB(30, 30, 200)
            img = Image.new("RGB", (width, height), (30, 30, 200))
        else:
            img = Image.new("RGB", (width, height), (0, 0, 0))

        if low_quality:
            # Low contrast/brightness image (very dark, almost black)
            img = Image.new("RGB", (width, height), (5, 5, 5))
        else:
            # Draw patterns to guarantee non-zero variance for image quality checks
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            line_color = (200, 250, 200) if color == "green" else (220, 180, 100) if color == "yellow_brown" else (100, 100, 255)
            rect_color = (40, 120, 40) if color == "green" else (100, 60, 20) if color == "yellow_brown" else (10, 10, 150)
            draw.line([(0, 0), (width, height)], fill=line_color, width=10)
            draw.rectangle([(width // 4, height // 4), (width // 2, height // 2)], fill=rect_color)

        out = io.BytesIO()
        img.save(out, format="JPEG")
        return out.getvalue()
    return _generate
