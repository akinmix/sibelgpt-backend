# tests/unit/test_property_search_handler.py
import pytest
from unittest.mock import patch, MagicMock
import property_search_handler

def test_is_property_search_query_positive():
    """Test property search query detection - positive cases"""
    assert property_search_handler.is_property_search_query("Kadıköy'de satılık daire") == True
    assert property_search_handler.is_property_search_query("3+1 ev arıyorum") == True
    assert property_search_handler.is_property_search_query("villa bul") == True

def test_is_property_search_query_negative():
    """Test property search query detection - negative cases"""
    assert property_search_handler.is_property_search_query("Hava durumu") == False
    assert property_search_handler.is_property_search_query("Bitcoin fiyatı") == False

def test_format_property_listings_empty():
    """Test formatting empty listings"""
    result = property_search_handler.format_property_listings([])
    assert "Hiç ilan bulunamadı" in result

def test_format_property_listings_with_data():
    """Test formatting with sample data"""
    sample_data = [{
        "ilan_id": "P123456",
        "baslik": "Test Daire",
        "fiyat": "1000000",
        "ilce": "Kadıköy",
        "mahalle": "Moda"
    }]
    result = property_search_handler.format_property_listings(sample_data)
    assert "Test Daire" in result
    assert "P123456" in result
    assert "PDF İndir" in result
