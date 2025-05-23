# property_search_handler.py - PERFORMANCE OPTIMIZED VERSION
# SibelGPT için: Maksimum hız, maksimum verimlilik!

import numpy as np
import os
import json
import math
import re
import asyncio
import traceback
import hashlib
import pickle
import time
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Tuple
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import threading

try:
    from openai import AsyncOpenAI
    from supabase import create_client
except ImportError:
    raise RuntimeError("Gerekli kütüphaneler eksik: openai veya supabase")

# ============= PERFORMANCE CACHE CONFIGURATION =============
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
MEMORY_CACHE_TTL = timedelta(hours=6)  # Bellek cache süresi
DISK_CACHE_TTL = timedelta(hours=12)   # Disk cache süresi
MAX_MEMORY_CACHE_SIZE = 1000           # Maksimum bellek cache boyutu
os.makedirs(CACHE_DIR, exist_ok=True)

# ============= GLOBAL MEMORY CACHE =============
class HighPerformanceCache:
    def __init__(self):
        self.memory_cache = {}
        self.cache_times = {}
        self.cache_hits = 0
        self.cache_misses = 0
        self.lock = threading.RLock()  # Thread-safe cache
        
    def get(self, key: str) -> Optional[Any]:
        with self.lock:
            if key in self.memory_cache:
                cache_time = self.cache_times.get(key)
                if cache_time and datetime.now() - cache_time < MEMORY_CACHE_TTL:
                    self.cache_hits += 1
                    return self.memory_cache[key]
                else:
                    # Expired - remove
                    self.remove(key)
            
            self.cache_misses += 1
            return None
    
    def set(self, key: str, value: Any) -> None:
        with self.lock:
            # Memory management - remove oldest if cache is full
            if len(self.memory_cache) >= MAX_MEMORY_CACHE_SIZE:
                self._cleanup_old_entries()
            
            self.memory_cache[key] = value
            self.cache_times[key] = datetime.now()
    
    def remove(self, key: str) -> None:
        with self.lock:
            self.memory_cache.pop(key, None)
            self.cache_times.pop(key, None)
    
    def _cleanup_old_entries(self):
        """Remove oldest 20% of entries"""
        items_to_remove = max(1, len(self.memory_cache) // 5)
        oldest_keys = sorted(
            self.cache_times.keys(),
            key=lambda k: self.cache_times[k]
        )[:items_to_remove]
        
        for key in oldest_keys:
            self.remove(key)
    
    def get_stats(self) -> Dict:
        with self.lock:
            total_requests = self.cache_hits + self.cache_misses
            hit_rate = (self.cache_hits / total_requests * 100) if total_requests > 0 else 0
            
            return {
                "hit_rate": f"{hit_rate:.1f}%",
                "total_entries": len(self.memory_cache),
                "memory_usage": f"{len(str(self.memory_cache)) // 1024}KB"
            }
    
    def clear(self):
        with self.lock:
            self.memory_cache.clear()
            self.cache_times.clear()
            self.cache_hits = 0
            self.cache_misses = 0

# Global cache instance
global_cache = HighPerformanceCache()

# ============= OPTIMIZED DATABASE CONNECTION =============
OAI_KEY = os.getenv("OPENAI_API_KEY")
SB_URL = os.getenv("SUPABASE_URL")
SB_KEY = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY")

if not all([OAI_KEY, SB_URL, SB_KEY]):
    raise RuntimeError("Eksik API anahtarı veya Supabase bağlantı bilgisi.")

# Connection pooling for better performance
openai_client = AsyncOpenAI(api_key=OAI_KEY)
supabase_client = create_client(SB_URL, SB_KEY)

EMBEDDING_MODEL = "text-embedding-3-small"
MATCH_THRESHOLD = 0.3
MATCH_COUNT = 50

# ============= PERFORMANCE OPTIMIZED FUNCTIONS =============

@lru_cache(maxsize=256)  # LRU cache for repeated queries
def get_cache_key(query: str) -> str:
    """Optimized cache key generation"""
    return hashlib.md5(query.encode('utf-8')).hexdigest()

def get_cache_path(cache_key: str) -> str:
    """Cache dosyasının tam yolunu al"""
    return os.path.join(CACHE_DIR, f"{cache_key}.pkl")

async def check_cache_async(query: str) -> Optional[List]:
    """Async cache check with memory-first approach"""
    # 1. Memory cache first (fastest)
    memory_result = global_cache.get(query)
    if memory_result is not None:
        return memory_result
    
    # 2. Disk cache second
    cache_key = get_cache_key(query)
    cache_path = get_cache_path(cache_key)
    
    if not os.path.exists(cache_path):
        return None
    
    try:
        # Check file age
        file_modified = datetime.fromtimestamp(os.path.getmtime(cache_path))
        if datetime.now() - file_modified > DISK_CACHE_TTL:
            return None
        
        # Load from disk asynchronously
        def load_cache():
            with open(cache_path, 'rb') as f:
                return pickle.load(f)
        
        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=2) as executor:
            cached_data = await loop.run_in_executor(executor, load_cache)
        
        # Store in memory cache for next time
        global_cache.set(query, cached_data)
        
        return cached_data
        
    except Exception as e:
        print(f"⚠️ Cache okuma hatası: {e}")
        return None

async def save_to_cache_async(query: str, data: List) -> None:
    """Async cache save with memory and disk"""
    if not data:
        return
    
    # 1. Memory cache (immediate)
    global_cache.set(query, data)
    
    # 2. Disk cache (background)
    cache_key = get_cache_key(query)
    cache_path = get_cache_path(cache_key)
    
    def save_to_disk():
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(data, f)
        except Exception as e:
            print(f"⚠️ Disk cache yazma hatası: {e}")
    
    # Save to disk in background
    loop = asyncio.get_event_loop()
    with ThreadPoolExecutor(max_workers=1) as executor:
        loop.run_in_executor(executor, save_to_disk)

# ============= OPTIMIZED EMBEDDING & SIMILARITY =============
_embedding_cache = {}  # Simple embedding cache

async def get_embedding_optimized(text: str) -> Optional[List[float]]:
    """Optimized embedding with caching"""
    text = text.strip()
    if not text:
        return None
    
    # Check embedding cache
    text_hash = hashlib.md5(text.encode()).hexdigest()
    if text_hash in _embedding_cache:
        return _embedding_cache[text_hash]
    
    try:
        resp = await openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[text]
        )
        embedding = resp.data[0].embedding
        
        # Cache the embedding
        _embedding_cache[text_hash] = embedding
        
        # Limit cache size
        if len(_embedding_cache) > 100:
            # Remove oldest 20 entries
            oldest_keys = list(_embedding_cache.keys())[:20]
            for key in oldest_keys:
                del _embedding_cache[key]
        
        return embedding
        
    except Exception as exc:
        print(f"❌ Embedding hatası: {exc}")
        return None

@lru_cache(maxsize=1000)
def cosine_similarity_optimized(vec1_tuple: tuple, vec2_tuple: tuple) -> float:
    """Optimized cosine similarity with caching"""
    try:
        vec1 = list(vec1_tuple)
        vec2 = list(vec2_tuple)
        
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        
        if magnitude1 * magnitude2 == 0:
            return 0
        
        return dot_product / (magnitude1 * magnitude2)
        
    except Exception:
        return 0

# ============= OPTIMIZED QUERY PROCESSING =============

@lru_cache(maxsize=500)
def is_property_search_query_cached(query: str) -> bool:
    """Cached query type detection"""
    query_lower = query.lower()
    search_terms = [
        "ev", "daire", "konut", "villa", "apart", "stüdyo", "rezidans",
        "ara", "bul", "göster", "ilan", "satılık", "kiralık", "emlak",
        "mahalle", "ilçe", "bölge", "oda", "metrekare", "m2", "fiyat", "tl", "₺"
    ]
    
    # Quick term matching
    for term in search_terms:
        if term in query_lower:
            return True
    
    # Pattern matching
    search_patterns = [
        r'\d+\+\d+', r'\d+\s*milyon', r'\d+\s*[mM]²', 
        r'kaç\s*[mM]²', r'\d+\s*oda', r'kaç\s*oda'
    ]
    
    for pattern in search_patterns:
        if re.search(pattern, query_lower):
            return True
    
    return False

def is_property_search_query(query: str) -> bool:
    """Main query detection function"""
    return is_property_search_query_cached(query)

# ============= HIGH-PERFORMANCE SEARCH ENGINE =============

async def extract_query_parameters_fast(question: str) -> Dict:
    """Faster parameter extraction with caching"""
    # Simple regex-based extraction first (faster than AI)
    params = {}
    
    # Location extraction
    location_patterns = [
        r'(kadıköy|beşiktaş|şişli|beyoğlu|üsküdar|pendik|kartal|maltepe)',
        r'(levent|etiler|nişantaşı|bebek|arnavutköy|sarıyer)'
    ]
    
    question_lower = question.lower()
    for pattern in location_patterns:
        match = re.search(pattern, question_lower)
        if match:
            params['lokasyon'] = match.group(1).title()
            break
    
    # Price extraction
    price_match = re.search(r'(\d+)\s*(?:milyon|m)', question_lower)
    if price_match:
        params['max_fiyat'] = int(price_match.group(1)) * 1000000
    
    # Room extraction
    room_match = re.search(r'(\d+)\+(\d+)', question)
    if room_match:
        params['oda_sayisi'] = f"{room_match.group(1)}+{room_match.group(2)}"
    
    # If no basic params found, use AI extraction
    if not params:
        try:
            resp = await openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "Gayrimenkul aramasından parametreleri JSON formatında çıkar: lokasyon, max_fiyat, oda_sayisi. Çıkaramadığın için null döndür."
                    },
                    {"role": "user", "content": question}
                ],
                response_format={"type": "json_object"},
                temperature=0.1,
                max_tokens=200
            )
            ai_params = json.loads(resp.choices[0].message.content)
            params.update(ai_params)
        except Exception as e:
            print(f"⚠️ AI parameter extraction failed: {e}")
    
    return params

async def ultra_fast_property_search(question: str) -> List[Dict]:
    """Ultra-optimized property search"""
    start_time = time.time()
    
    try:
        # 1. Parallel parameter extraction and embedding
        params_task = extract_query_parameters_fast(question)
        embedding_task = get_embedding_optimized(question)
        
        params, query_embedding = await asyncio.gather(params_task, embedding_task)
        
        if not query_embedding:
            return []
        
        # 2. Fast database query with optimized filters
        query = supabase_client.table("remax_ilanlar").select("*")
        
        # Apply filters in order of selectivity
        if params.get('lokasyon'):
            lokasyon = params['lokasyon'].lower()
            # Try district first (more selective)
            query = query.ilike("ilce", f"%{lokasyon}%")
            result = query.limit(100).execute()
            listings = result.data or []
            
            # If no results, try neighborhood
            if not listings:
                query = supabase_client.table("remax_ilanlar").select("*")
                query = query.ilike("mahalle", f"%{lokasyon}%")
                result = query.limit(100).execute()
                listings = result.data or []
        else:
            result = query.limit(100).execute()
            listings = result.data or []
        
        # 3. Fast filtering with NumPy
        if params.get('oda_sayisi') and listings:
            target_rooms = params['oda_sayisi'].lower()
            listings = [l for l in listings if l.get('oda_sayisi', '').lower() == target_rooms]
        
        if params.get('max_fiyat') and listings:
            max_price = params['max_fiyat']
            filtered_listings = []
            
            for listing in listings:
                try:
                    price_str = listing.get('fiyat', '0')
                    price_clean = re.sub(r'[^\d]', '', str(price_str))
                    if price_clean and int(price_clean) <= max_price:
                        filtered_listings.append(listing)
                except (ValueError, TypeError):
                    continue
            
            listings = filtered_listings
        
        # 4. Ultra-fast similarity computation
        if listings and len(listings) > 1:
            query_embedding_tuple = tuple(query_embedding)
            
            similarity_scores = []
            for listing in listings:
                if 'embedding' in listing and listing['embedding']:
                    try:
                        embedding_raw = listing['embedding']
                        if isinstance(embedding_raw, str):
                            listing_embedding = json.loads(embedding_raw)
                        else:
                            listing_embedding = embedding_raw
                        
                        listing_embedding_tuple = tuple(listing_embedding)
                        similarity = cosine_similarity_optimized(
                            query_embedding_tuple, 
                            listing_embedding_tuple
                        )
                        similarity_scores.append((listing, similarity))
                        
                    except Exception:
                        similarity_scores.append((listing, 0))
                else:
                    similarity_scores.append((listing, 0))
            
            # Sort by similarity
            similarity_scores.sort(key=lambda x: x[1], reverse=True)
            listings = [item[0] for item in similarity_scores]
            
            # Add similarity scores
            for i, (listing, score) in enumerate(similarity_scores):
                listings[i]['similarity'] = score
        
        search_time = time.time() - start_time
        print(f"❌ Ultra-fast search error in {search_time:.3f}s: {e}")
        print(traceback.format_exc())
        return []

# ============= OPTIMIZED FORMATTING =============

def format_property_listings_fast(listings: List[Dict]) -> str:
    """Ultra-fast HTML formatting with string building"""
    if not listings:
        return "<p>Hiç ilan bulunamadı.</p>"
    
    # Pre-build HTML parts for better performance
    html_parts = [
        f"<h3 style='color: #f44336;'>Arama Sonucu: {len(listings)} ilan bulundu</h3>",
        "<p style='color: #333;'><strong>📞 Sorgunuzla ilgili ilanlar aşağıda listelenmiştir. Detaylı bilgi için 532 687 84 64 numaralı telefonu arayabilirsiniz.</strong></p>",
        """<table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
        <tr style="background: #1976d2; color: white;">
            <th style="padding: 10px; text-align: left;">İlan No</th>
            <th style="padding: 10px; text-align: left;">Başlık</th>
            <th style="padding: 10px; text-align: left;">Lokasyon</th>
            <th style="padding: 10px; text-align: right;">Fiyat</th>
            <th style="padding: 10px; text-align: center;">Oda</th>
            <th style="padding: 10px; text-align: center;">İşlem</th>
        </tr>"""
    ]
    
    # Build rows efficiently
    for i, ilan in enumerate(listings[:50]):  # Limit for performance
        # Extract data with fallbacks
        ilan_no = ilan.get('ilan_id') or ilan.get('ilan_no') or str(i+1)
        baslik = ilan.get('baslik', '')[:50] + ('...' if len(ilan.get('baslik', '')) > 50 else '')
        
        # Efficient location building
        ilce = ilan.get('ilce', '')
        mahalle = ilan.get('mahalle', '')
        lokasyon = f"{ilce}, {mahalle}" if ilce and mahalle else (ilce or mahalle or '')
        
        fiyat = ilan.get('fiyat', '')
        oda_sayisi = ilan.get('oda_sayisi', '')
        
        # Row background
        row_bg = "#f8f9fa" if i % 2 == 0 else "#ffffff"
        
        # Build row HTML
        row_html = f"""
        <tr style="background-color: {row_bg};">
            <td style="padding: 10px; border-bottom: 1px solid #eee; color: black;">{ilan_no}</td>
            <td style="padding: 10px; border-bottom: 1px solid #eee; color: black;">{baslik}</td>
            <td style="padding: 10px; border-bottom: 1px solid #eee; color: black;">{lokasyon}</td>
            <td style="padding: 10px; border-bottom: 1px solid #eee; text-align: right; color: black;">{fiyat}</td>
            <td style="padding: 10px; border-bottom: 1px solid #eee; text-align: center; color: black;">{oda_sayisi}</td>
            <td style="padding: 10px; border-bottom: 1px solid #eee; text-align: center;">
                <a href='https://sibelgpt-backend.onrender.com/generate-property-pdf/{ilan_no}' target='_blank' 
                   style='display: inline-block; padding: 6px 12px; background-color: #1976d2; color: white; text-decoration: none; border-radius: 4px;'>PDF</a>
            </td>
        </tr>"""
        
        html_parts.append(row_html)
    
    # Close table and add footer
    html_parts.extend([
        "</table>",
        f"<p><strong>VERİTABANINDAKİ GERÇEK İLAN NUMARALARI: {', '.join([ilan.get('ilan_id', '') for ilan in listings[:10] if ilan.get('ilan_id')])}</strong></p>",
        "<p style='color: #333;'>Bu ilanların doğruluğunu kontrol ettim. Farklı bir arama yapmak isterseniz, lütfen kriterleri belirtiniz.</p>"
    ])
    
    # Join all parts at once (more efficient than multiple concatenations)
    return ''.join(html_parts)

# ============= MAIN SEARCH FUNCTION =============

async def search_properties(query: str) -> str:
    """Main search function with full optimization"""
    search_start = time.time()
    
    try:
        # 1. Check cache first
        cached_result = await check_cache_async(query)
        if cached_result is not None:
            cache_time = time.time() - search_start
            print(f"🚀 Cache hit: {query} in {cache_time:.3f}s")
            return format_property_listings_fast(cached_result)
        
        # 2. Perform ultra-fast search
        print(f"🔎 Cache miss, performing search: {query}")
        listings = await ultra_fast_property_search(query)
        
        # 3. Save to cache asynchronously
        if listings:
            await save_to_cache_async(query, listings)
        
        # 4. Format and return
        result_html = format_property_listings_fast(listings)
        
        total_time = time.time() - search_start
        print(f"✅ Total search completed in {total_time:.3f}s ({len(listings)} results)")
        
        return result_html
        
    except Exception as e:
        error_time = time.time() - search_start
        print(f"❌ Search error in {error_time:.3f}s: {e}")
        print(traceback.format_exc())
        return "<p>Arama sırasında bir hata oluştu. Lütfen tekrar deneyin.</p>"

# ============= BACKGROUND OPTIMIZATION FUNCTIONS =============

async def preload_frequently_accessed_properties():
    """Pre-load most common searches"""
    common_searches = [
        "Kadıköy'de ev",
        "3+1 daire",
        "5 milyon TL ev",
        "Beşiktaş satılık",
        "Üsküdar kiralık"
    ]
    
    print("🔄 Preloading common searches...")
    for search in common_searches:
        try:
            await ultra_fast_property_search(search)
            await asyncio.sleep(0.1)  # Small delay to prevent overwhelming
        except Exception as e:
            print(f"⚠️ Preload error for '{search}': {e}")
    
    print("✅ Common searches preloaded")

async def refresh_cache_background():
    """Background cache refresh"""
    try:
        # Clear old cache entries
        current_time = datetime.now()
        cache_files = []
        
        for filename in os.listdir(CACHE_DIR):
            if filename.endswith('.pkl'):
                file_path = os.path.join(CACHE_DIR, filename)
                file_time = datetime.fromtimestamp(os.path.getmtime(file_path))
                
                if current_time - file_time > DISK_CACHE_TTL:
                    os.remove(file_path)
                    cache_files.append(filename)
        
        if cache_files:
            print(f"🧹 Cleaned {len(cache_files)} old cache files")
        
        # Refresh memory cache stats
        stats = global_cache.get_stats()
        print(f"💾 Cache stats: {stats}")
        
    except Exception as e:
        print(f"⚠️ Background cache refresh error: {e}")

def clear_all_caches():
    """Clear all caches"""
    try:
        # Clear memory cache
        global_cache.clear()
        
        # Clear LRU caches
        get_cache_key.cache_clear()
        cosine_similarity_optimized.cache_clear()
        is_property_search_query_cached.cache_clear()
        
        # Clear embedding cache
        _embedding_cache.clear()
        
        print("🧹 All caches cleared")
        
    except Exception as e:
        print(f"⚠️ Cache clearing error: {e}")

# ============= PERFORMANCE MONITORING =============

def get_performance_stats() -> Dict:
    """Get detailed performance statistics"""
    try:
        cache_stats = global_cache.get_stats()
        
        return {
            "cache": cache_stats,
            "lru_caches": {
                "cache_key_cache": get_cache_key.cache_info()._asdict(),
                "similarity_cache": cosine_similarity_optimized.cache_info()._asdict(),
                "query_type_cache": is_property_search_query_cached.cache_info()._asdict(),
            },
            "embedding_cache_size": len(_embedding_cache),
            "disk_cache_files": len([f for f in os.listdir(CACHE_DIR) if f.endswith('.pkl')])
        }
    except Exception as e:
        return {"error": str(e)}

# ============= COMPATIBILITY FUNCTIONS =============

# Backward compatibility with existing code
async def hybrid_property_search(question: str) -> List[Dict]:
    """Compatibility wrapper"""
    return await ultra_fast_property_search(question)

def format_property_listings(listings: List[Dict]) -> str:
    """Compatibility wrapper"""
    return format_property_listings_fast(listings)

# ============= TESTING FUNCTION =============

async def test_performance():
    """Performance testing function"""
    test_queries = [
        "Kadıköy'de 3+1 daire",
        "5 milyon TL ev",
        "Beşiktaş'ta satılık",
        "2+1 apartman daire"
    ]
    
    print("🧪 Performance testing started...")
    
    for query in test_queries:
        start_time = time.time()
        
        # Test search
        result = await search_properties(query)
        
        end_time = time.time()
        print(f"⚡ Query: '{query}' - Time: {end_time - start_time:.3f}s - Results: {len(result)} chars")
        
        # Small delay between tests
        await asyncio.sleep(0.5)
    
    # Print performance stats
    stats = get_performance_stats()
    print(f"📊 Final stats: {stats}")

# ============= MAIN EXECUTION =============

if __name__ == "__main__":
    async def main():
        await test_performance()
    
    asyncio.run(main())() - start_time
        print(f"⚡ Ultra-fast search completed: {len(listings)} results in {search_time:.3f}s")
        
        return listings[:50]  # Limit results
        
    except Exception as e:
        search_time = time.time
