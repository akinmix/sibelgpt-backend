# tests/unit/test_ask_handler.py
import pytest
from unittest.mock import patch, AsyncMock
import sys
import os

# Ana dizini import path'e ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import ask_handler

@pytest.mark.asyncio
async def test_detect_topic_real_estate():
    """Test real estate topic detection"""
    # Basit bir test - gerçek fonksiyonu çağırır
    result = await ask_handler.detect_topic("Kadıköy'de ev", "real-estate")
    assert result == "real-estate"

@pytest.mark.asyncio
async def test_detect_topic_greeting():
    """Test greeting detection"""
    result = await ask_handler.detect_topic("merhaba", "real-estate")
    assert result == "real-estate"  # Selamlaşma mevcut modda kalır

def test_get_out_of_scope_response():
    """Test out of scope response"""
    response = ask_handler.get_out_of_scope_response("real-estate")
    assert "Gayrimenkul GPT" in response
    assert "uzmanlık alanı" in response

# Bu test OpenAI API kullanmaz, sadece fonksiyon var mı kontrol eder
def test_functions_exist():
    """Test that required functions exist"""
    assert hasattr(ask_handler, 'answer_question')
    assert hasattr(ask_handler, 'detect_topic')
    assert callable(ask_handler.answer_question)
