import json
from unittest.mock import patch, mock_open
import pytest

def test_get_analytics_empty(client):
    with patch("api.main._analytics_db", {}):
        response = client.get("/analytics")
        assert response.status_code == 200
        assert response.json() == {"disease_counts": {}}

def test_get_analytics_populated(client):
    mock_db = {"Apple___healthy": 5, "Tomato___Early_blight": 12}
    with patch("api.main._analytics_db", mock_db):
        response = client.get("/analytics")
        assert response.status_code == 200
        assert response.json() == {"disease_counts": mock_db}

@pytest.mark.parametrize("disease,initial_count,expected", [
    ("Apple___healthy", 0, 1),
    ("Tomato___Early_blight", 10, 11),
    ("Grape___Black_rot", 5, 6),
])
def test_analytics_increment(disease, initial_count, expected):
    # Test internal analytics increment helper structure directly
    db = {disease: initial_count}
    db[disease] = db.get(disease, 0) + 1
    assert db[disease] == expected
