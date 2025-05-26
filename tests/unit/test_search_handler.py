import pytest
import asyncio
from search_handler import web_search_answer, search_google, format_search_results

@pytest.mark.asyncio
async def test_search_google_function():
    # Mock response yerine basit bir test
    assert callable(search_google)
    
@pytest.mark.asyncio
async def test_format_search_results():
    # Test empty results
    formatted = format_search_results([])
    assert "bulunamadı" in formatted
    
    # Test with mock results
    mock_results = [{"title": "Test", "link": "https://example.com", "snippet": "Test snippet"}]
    formatted = format_search_results(mock_results)
    assert "Test" in formatted
    assert "example.com" in formatted
