# tests/integration/test_api_endpoints.py
import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch
from main import app

client = TestClient(app)

def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data

def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "service" in data

def test_statistics_endpoint():
    """Test statistics endpoint"""
    response = client.get("/statistics/simple")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "statistics" in data

def test_config_endpoint():
    """Test config endpoint"""
    response = client.get("/api/config")
    assert response.status_code == 200
    data = response.json()
    assert "supabaseUrl" in data

# Bu test gerçek AI çağrısı yapmaz, sadece endpoint'in çalışıp çalışmadığını test eder
@patch('ask_handler.answer_question')
def test_chat_endpoint_mocked(mock_answer):
    """Test chat endpoint with mocked response"""
    mock_answer.return_value = "Test response"
    
    payload = {
        "question": "Test question",
        "mode": "real-estate"
    }
    
    response = client.post("/chat", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "reply" in data

def test_chat_endpoint_validation():
    """Test chat endpoint validation"""
    # Eksik alan ile test
    payload = {"mode": "real-estate"}  # question eksik
    
    response = client.post("/chat", json=payload)
    assert response.status_code == 422  # Validation error
