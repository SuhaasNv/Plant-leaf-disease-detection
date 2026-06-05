import pytest

def test_health_check_success(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_health_check_invalid_method(client):
    response = client.post("/health")
    assert response.status_code == 405
