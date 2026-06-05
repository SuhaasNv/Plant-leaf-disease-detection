import pytest
from PIL import Image
import io
import numpy as np
from unittest.mock import patch
from api.main import _validate_image_bytes, _is_likely_leaf

# Parameterized cases for magic bytes
magic_byte_cases = [
    (b"\xff\xd8\xff\xe0", None),  # Valid JPEG
    (b"\x89PNG\r\n\x1a\n", None),  # Valid PNG
    (b"RIFF\x00\x00\x00\x00WEBP", None),  # Valid WebP
    (b"GIF89a", None),  # Valid GIF
    (b"PK\x03\x04", "Invalid file format. Please upload a PNG or JPG image."),  # Zip file
    (b"%PDF-1.4", "Invalid file format. Please upload a PNG or JPG image."),  # PDF file
    (b"plain text", "Invalid file format. Please upload a PNG or JPG image."),  # Plain text
]

@pytest.mark.parametrize("file_bytes, expected_error", magic_byte_cases)
def test_validate_image_bytes_formats(file_bytes, expected_error):
    result = _validate_image_bytes(file_bytes)
    assert result == expected_error

def test_validate_image_bytes_size_limit():
    # 10 MB is limit, generate 11 MB bytes
    large_bytes = b"\xff\xd8\xff" + b"\x00" * (10 * 1024 * 1024 + 100)
    result = _validate_image_bytes(large_bytes)
    assert result == "File too large. Maximum size is 10 MB."

# Parameterized cases for leaf color heuristic
color_cases = [
    ((80, 180, 80), True),       # Bright Green - Healthy Leaf color
    ((150, 100, 40), True),      # Yellow Brown - Diseased Leaf color
    ((30, 30, 200), False),      # Dark Blue - Not a leaf color
    ((255, 0, 0), False),        # Pure Red - Not a leaf color
    ((240, 240, 240), False),    # Off-white background - Not a leaf color
]

@pytest.mark.parametrize("rgb_color, expected_leaf", color_cases)
def test_is_likely_leaf_heuristics(rgb_color, expected_leaf):
    img = Image.new("RGB", (100, 100), rgb_color)
    assert _is_likely_leaf(img) == expected_leaf

# Parameterized cases for dimensions and quality checks on API level
quality_scenarios = [
    # (width, height, color, low_quality, expected_status, expected_error_msg)
    (100, 100, "green", False, 400, "Image quality too low. Please upload a clear close-up leaf photo."),  # Too small
    (5000, 500, "green", False, 400, "Image dimensions too large"),  # Width too large
    (500, 5000, "green", False, 400, "Image dimensions too large"),  # Height too large
    (300, 300, "green", True, 400, "Image quality too low. Please upload a clear close-up leaf photo."),  # Low contrast/dark
    (300, 300, "blue", False, 422, "No plant leaf detected. Please upload a clear photo of a leaf."),  # Not a leaf color
]

@pytest.mark.parametrize("w, h, col, low_q, status, err_snippet", quality_scenarios)
def test_predict_quality_guards(client, generate_image_bytes, w, h, col, low_q, status, err_snippet):
    img = generate_image_bytes(width=w, height=h, color=col, low_quality=low_q)
    files = {"file": ("leaf.jpg", img, "image/jpeg")}
    response = client.post("/predict", files=files)
    
    assert response.status_code == status
    detail = response.json().get("detail") or response.json().get("error")
    assert err_snippet in str(detail)
