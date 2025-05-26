# tests/conftest.py
import pytest
from unittest.mock import AsyncMock, MagicMock
from fastapi.testclient import TestClient
from main import app

@pytest.fixture
def client():
    """FastAPI test client"""
    return TestClient(app)

@pytest.fixture
def sample_property():
    """Sample property data"""
    return {
        "ilan_id": "P123456",
        "baslik": "Test Daire",
        "fiyat": "1000000",
        "ilce": "Kadıköy",
        "mahalle": "Moda",
        "oda_sayisi": "3+1"
    }
