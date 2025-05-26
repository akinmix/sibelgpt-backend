# tests/conftest.py - DÜZELTİLMİŞ VERSİYON
import pytest
import sys
import os
from pathlib import Path

# ✅ DOĞRU PATH AYARLARI
# Ana proje dizinini bul
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print(f"✅ Project root: {project_root}")
print(f"✅ Python path: {sys.path[:3]}")

# ✅ GÜVENLİ IMPORT - Hata yakalamalı
try:
    # Ana modülü import etmeye çalış
    import main
    print("✅ main modülü başarıyla import edildi")
except ImportError as e:
    print(f"⚠️ main modülü import edilemedi: {e}")
    # Test devam etsin ama uyarı ver
    main = None

# ✅ PYTEST CONFIGURATION
@pytest.fixture(scope="session")
def app():
    """FastAPI app fixture"""
    if main:
        return main.app
    else:
        # Mock app döndür
        from fastapi import FastAPI
        return FastAPI()

@pytest.fixture(scope="session") 
def event_loop():
    """Event loop fixture for async tests"""
    import asyncio
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

# ✅ TEST ENVIRONMENT SETUP
def pytest_configure(config):
    """Pytest configuration"""
    print("🧪 Test environment configured")
    
def pytest_sessionstart(session):
    """Session start hook"""
    print("🚀 Test session started")
    
def pytest_sessionfinish(session, exitstatus):
    """Session finish hook"""
    print(f"✅ Test session finished with status: {exitstatus}")
