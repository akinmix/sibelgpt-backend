import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_chat_endpoint():
    payload = {"question": "Merhaba", "mode": "real-estate"}
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    assert "reply" in response.json()
