import pytest
from unittest.mock import patch

# Parameterized test for authenticating requests
auth_scenarios = [
    # (API_KEY env, request header, expected status)
    ("", None, 200),  # Open mode - no key set, no header passed -> OK
    ("", "random-key", 200),  # Open mode - key passed but ignored -> OK
    ("secret-key", None, 401),  # Key set on server, none provided -> Unauthorized
    ("secret-key", "wrong-key", 401),  # Key set on server, invalid key provided -> Unauthorized
    ("secret-key", "secret-key", 200),  # Key set on server, valid key provided -> OK
]

@pytest.mark.parametrize("server_key, client_key_header, expected_status", auth_scenarios)
def test_api_key_auth_predict(client, generate_image_bytes, server_key, client_key_header, expected_status):
    # Setup test file
    img = generate_image_bytes(color="green")
    files = {"file": ("leaf.jpg", img, "image/jpeg")}
    
    headers = {}
    if client_key_header is not None:
        headers["X-API-Key"] = client_key_header
        
    with patch("api.main._API_KEY", server_key or None):
        response = client.post("/predict", files=files, headers=headers)
        assert response.status_code == expected_status
