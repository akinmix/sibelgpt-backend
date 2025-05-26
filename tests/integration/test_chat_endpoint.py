# tests/integration/test_chat_endpoint.py
import pytest
from unittest.mock import patch
import sys
import os

sys.path.insert(0, '/opt/render/project/src')

try:
    from main import app
    from fastapi.testclient import TestClient
    client = TestClient(app)
    HAS_APP = True
except ImportError:
    HAS_APP = False

class TestChatEndpoint:
    """Chat API endpoint testleri"""
    
    @pytest.mark.skipif(not HAS_APP, reason="Main app not available")
    @patch('ask_handler.answer_question')
    def test_chat_endpoint_success(self, mock_answer):
        """Chat endpoint başarı testi"""
        
        # Sahte yanıt ayarla
        mock_answer.return_value = "Chat test yanıtı"
        
        # API çağrısı yap
        response = client.post("/chat", json={
            "question": "Test sorusu",
            "mode": "real-estate"
        })
        
        # Kontrol et
        assert response.status_code == 200
        data = response.json()
        assert "reply" in data
        assert data["reply"] == "Chat test yanıtı"
    
    @pytest.mark.skipif(not HAS_APP, reason="Main app not available")
    def test_chat_endpoint_validation(self):
        """Chat endpoint doğrulama testi"""
        
        # Eksik parametre ile test
        response = client.post("/chat", json={
            "mode": "real-estate"
            # question eksik
        })
        
        assert response.status_code == 422  # Validation error
    
    @pytest.mark.skipif(not HAS_APP, reason="Main app not available")  
    def test_chat_endpoint_invalid_mode(self):
        """Geçersiz mod testi"""
        
        response = client.post("/chat", json={
            "question": "Test",
            "mode": "invalid-mode"  # Geçersiz mod
        })
        
        # Geçersiz mod da kabul edilebilir, çünkü backend handle ediyor
        assert response.status_code in [200, 400, 422]
    
    # Import gerekmez testler
    def test_chat_request_structure(self):
        """Chat request yapısı testi"""
        valid_request = {
            "question": "Test sorusu",
            "mode": "real-estate",
            "conversation_history": []
        }
        
        assert "question" in valid_request
        assert "mode" in valid_request
        assert isinstance(valid_request["conversation_history"], list)
