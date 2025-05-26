# tests/unit/test_search_handler.py
import pytest
from unittest.mock import patch, AsyncMock
import search_handler

def test_format_search_results_empty():
    """Test formatting empty search results"""
    result = search_handler.format_search_results([])
    assert "bulunamadı" in result

def test_format_search_results_with_data():
    """Test formatting with mock data"""
    mock_results = [{
        "title": "Test Title",
        "link": "https://example.com",
        "snippet": "Test snippet"
    }]
    result = search_handler.format_search_results(mock_results)
    assert "Test Title" in result
    assert "example.com" in result
    assert "Test snippet" in result

# Bu test gerçek API çağrısı yapmaz, sadece fonksiyon var mı kontrol eder
def test_functions_exist():
    """Test that required functions exist"""
    assert hasattr(search_handler, 'search_google')
    assert hasattr(search_handler, 'web_search_answer')
    assert callable(search_handler.search_google)
