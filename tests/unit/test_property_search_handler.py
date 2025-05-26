import pytest
import asyncio
from property_search_handler import is_property_search_query, search_properties

def test_is_property_search_query():
    # Test property search queries
    assert is_property_search_query("Kadıköy'de satılık daire")
    assert is_property_search_query("3+1 ev arıyorum")
    
    # Test non-property queries
    assert not is_property_search_query("Hava durumu nasıl")
    assert not is_property_search_query("Bitcoin fiyatı")
 
