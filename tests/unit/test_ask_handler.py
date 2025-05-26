# tests/unit/test_ask_handler.py
import pytest
import os
import sys

# Ana dizini import path'e ekle
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import ask_handler

# Basit bir unit test
def test_detect_topic_exists():
    """Test if detect_topic function exists"""
    assert hasattr(ask_handler, 'detect_topic'), "detect_topic function should exist"

# Async test
@pytest.mark.asyncio
async def test_check_if_property_listing_query():
    """Test if check_if_property_listing_query behaves as expected"""
    # Bu fonksiyon sadece varlığını kontrol ediyor, gerçek işlevselliği daha sonra test edilecek
    assert hasattr(ask_handler, 'check_if_property_listing_query'), "check_if_property_listing_query function should exist"
