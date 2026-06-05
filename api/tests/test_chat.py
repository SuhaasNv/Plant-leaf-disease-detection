import pytest
from unittest.mock import patch, MagicMock

chat_scenarios = [
    # (gemini_env, openai_env, gemini_header, openai_header, expected_status)
    ("", "", "", "", 503),  # No keys at all -> Service Unavailable
    ("gkey", "", "", "", 200),  # Gemini in env -> OK
    ("", "okey", "", "", 200),  # OpenAI in env -> OK
    ("", "", "gkey-header", "", 200),  # Gemini in header -> OK
    ("", "", "", "okey-header", 200),  # OpenAI in header -> OK
]

@pytest.mark.parametrize("gemini_env, openai_env, gemini_header, openai_header, expected_status", chat_scenarios)
def test_chat_key_configurations(client, gemini_env, openai_env, gemini_header, openai_header, expected_status):
    headers = {}
    if gemini_header:
        headers["x-gemini-key"] = gemini_header
    if openai_header:
        headers["x-openai-key"] = openai_header

    # Mock LLM helper functions to avoid actual API requests
    with patch("api.main.os.getenv") as mock_getenv, \
         patch("api.main._call_gemini") as mock_gemini, \
         patch("api.main._call_openai") as mock_openai:
         
        # Mock env vars return values
        mock_getenv.side_effect = lambda key, default=None: {
            "GEMINI_API_KEY": gemini_env,
            "OPENAI_API_KEY": openai_env,
            "RATE_LIMIT": "1000/minute"
        }.get(key, default)

        mock_gemini.return_value = ("Gemini response", None)
        mock_openai.return_value = ("OpenAI response", None)

        body = {
            "predictions": [{"label": "Apple___Apple_scab", "confidence": 0.9}],
            "message": "What is this?"
        }

        response = client.post("/chat", json=body, headers=headers)
        assert response.status_code == expected_status
        if expected_status == 200:
            data = response.json()
            assert "reply" in data
            assert data["reply"] in ["Gemini response", "OpenAI response"]

def test_chat_gemini_fails_openai_fallback(client):
    headers = {"x-gemini-key": "gkey", "x-openai-key": "okey"}
    
    with patch("api.main._call_gemini") as mock_gemini, \
         patch("api.main._call_openai") as mock_openai:
         
        # Gemini fails with error, OpenAI succeeds
        mock_gemini.return_value = (None, "Quota exceeded")
        mock_openai.return_value = ("OpenAI response fallback", None)

        body = {
            "predictions": [{"label": "Apple___Apple_scab", "confidence": 0.9}],
            "message": "What is this?"
        }

        response = client.post("/chat", json=body, headers=headers)
        assert response.status_code == 200
        assert response.json()["reply"] == "OpenAI response fallback"
        mock_gemini.assert_called_once()
        mock_openai.assert_called_once()

def test_chat_both_fail_returns_502(client):
    headers = {"x-gemini-key": "gkey", "x-openai-key": "okey"}
    
    with patch("api.main._call_gemini") as mock_gemini, \
         patch("api.main._call_openai") as mock_openai:
         
        # Both return errors
        mock_gemini.return_value = (None, "Gemini unavailable")
        mock_openai.return_value = (None, "OpenAI down")

        body = {
            "predictions": [{"label": "Apple___Apple_scab", "confidence": 0.9}],
            "message": "What is this?"
        }

        response = client.post("/chat", json=body, headers=headers)
        assert response.status_code == 502
        assert "Both Gemini and OpenAI chat services failed" in response.json()["detail"]
        assert "Gemini unavailable" in response.json()["detail"]
        assert "OpenAI down" in response.json()["detail"]
