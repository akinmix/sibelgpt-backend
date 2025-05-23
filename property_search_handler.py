# property_search_handler.py - MINIMAL WORKING VERSION
# SibelGPT için: Stabil çalışma öncelikli!

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
from typing import List, Dict, Optional, Any

try:
    from openai import AsyncOpenAI
    from supabase import create_client
except ImportError:
    raise RuntimeError("Gerekli kütüphaneler eksik: openai veya supabase")

# ============= BASIC CACHE CONFIGURATION =============
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
CACHE_TTL = timedelta(hours=2)
os.makedirs(CACHE_DIR, exist_ok=True)

# ============= BASIC MEMORY CACHE =============
class SimpleCache:
    def __init__(self):
        self.cache = {}
        self.timestamps = {}
        
    def get(self, key: str):
        if key in self.cache:
            timestamp = self.timestamps.get(key)
            if timestamp and datetime.now() - timestamp < CACHE_TTL:
                return self.cache[key]
            else:
                # Expired
                self.cache.pop(key, None)
                self.timestamps.pop(key, None)
        return None
    
    def set(self, key: str, value):
        self.cache[key] = value
        self.timestamps[key] = datetime.now()
        
        # Simple cleanup - keep only last 50 entries
        if len(self.cache) > 50:
            oldest_keys = list(self.cache.keys())[:10]
            for old_key in oldest_keys:
                self.cache.pop(old_key, None)
                self.timestamps.pop(old_key, None)
    
    def clear(self):
        self.cache.clear()
        self.timestamps.clear()

# Global simple cache
simple_cache = SimpleCache()

# ============= DATABASE CONNECTION =============
OAI_KEY = os.getenv("OPENAI_API_KEY")
SB_URL = os.getenv("SUPABASE_URL")
SB_KEY = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY")

if not all([OAI_KEY, SB_URL, SB_KEY]):
    raise RuntimeError("Eksik API anahtarı veya Supabase bağlantı bilgisi.")

openai_client = AsyncOpenAI(api_key=OAI_KEY)
supabase_client = create_client(SB_URL, SB_KEY)

EMBEDDING_MODEL = "text-embedding-3-small"
MATCH_THRESHOLD = 0.3
MATCH_COUNT = 50

# ============= BASIC FUNCTIONS =============

def get_cache_key(query: str) -> str:
    """Simple cache key generation"""
    return hashlib.md5(query.encode('utf-8')).hexdigest()

def check_cache(query: str):
    """Check memory cache first, then disk cache"""
    # Memory cache
    memory_result = simple_cache.get(query)
    if memory_result is not None:
        return memory_result
    
    # Disk cache
    cache_key = get_cache_key(query)
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    
    if not os.path.exists(cache_path):
        return None
    
    try:
        file_modified = datetime.fromtimestamp(os.path.getmtime(cache_path))
        if datetime.now() - file_modified > CACHE_TTL:
            return None
        
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
            simple_cache.set(query, cached_data)  # Store in memory for next time
            return cached_data
    except Exception as e:
        print(f"⚠️ Cache read error: {e}")
        return None

def save_to_cache(query: str, data: list):
    """Save to both memory and disk cache"""
    if not data:
        return
    
    # Memory cache
    simple_cache.set(query, data)
    
    # Disk cache
    cache_key = get_cache_key(query)
    cache_path = os.path.join(CACHE_DIR, f"{cache_key}.pkl")
    
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        print(f"⚠️ Cache write error: {e}")

# ============= EMBEDDING & SIMILARITY =============

async def get_embedding(text: str) -> Optional[List[float]]:
    """Get embedding from OpenAI"""
    text = text.strip()
    if not text:
        return None
    try:
        resp = await openai_client.embeddings.create(
            model=EMBEDDING_MODEL,
            input=[text]
        )
        return resp.data[0].embedding
    except Exception as exc:
        print(f"❌ Embedding error: {exc}")
        return None

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity"""
    try:
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        if magnitude1 * magnitude2 == 0:
            return 0
        return dot_product / (magnitude1 * magnitude2)
    except Exception:
        return 0

# ============= QUERY PROCESSING =============

def is_property_search_query(query: str) -> bool:
    """Detect if query is about property search"""
    try:
        query_lower = query.lower()
        search_terms = [
            "ev", "daire", "konut", "villa", "apart", "stüdyo", "rezidans",
            "ara", "bul", "göster", "ilan", "satılık", "kiralık", "emlak",
            "mahalle", "ilçe", "bölge", "oda", "metrekare", "m2", "fiyat", "tl", "₺"
        ]
        search_patterns = [
            r'\d+\+\d+', r'\d+\s*milyon', r'\d+\s*[mM]²'
        ]
        
        for term in search_terms:
            if term in query_lower:
                return True
        for pattern in search_patterns:
            if re.search(pattern, query_lower):
                return True
        return False
    except Exception:
        return False

async def extract_query_parameters(question: str) -> Dict:
    """Extract parameters from query"""
    try:
        resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """
                    Kullanıcının gayrimenkul aramasından arama parametrelerini çıkar ve JSON formatında döndür.
                    Çıkarılacak parametreler:
                    - lokasyon: İlçe, mahalle veya semt adı
                    - min_fiyat: Minimum fiyat (TL)
                    - max_fiyat: Maksimum fiyat (TL)
                    - oda_sayisi: "1+0", "1+1", "2+1", "3+1" vb.
                    - min_metrekare: Minimum metrekare
                    - max_metrekare: Maksimum metrekare
                    Parametreler çıkaramadığın alanlar için null döndür.
                    """
                },
                {"role": "user", "content": question}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=300
        )
        parameters = json.loads(resp.choices[0].message.content)
        return parameters
    except Exception as e:
        print(f"❌ Parameter extraction error: {e}")
        return {}

# ============= MAIN SEARCH FUNCTION =============

async def hybrid_property_search(question: str) -> List[Dict]:
    """Main property search function"""
    try:
        print(f"🔍 Starting search: {question}")
        
        # Get parameters and embedding in parallel
        params_task = extract_query_parameters(question)
        embedding_task = get_embedding(question)
        
        params, query_embedding = await asyncio.gather(params_task, embedding_task)
        
        if not query_embedding:
            print("⚠️ Could not generate embedding")
            return []
            
        print(f"✅ Parameters extracted: {params}")
        
        # Database query
        query = supabase_client.table("remax_ilanlar").select("*")
        
        # Location filter
        if params.get('lokasyon'):
            lokasyon = params['lokasyon'].lower()
            query = query.ilike("ilce", f"%{lokasyon}%")
            result = query.execute()
            listings = result.data if result.data else []
            
            if not listings:
                query = supabase_client.table("remax_ilanlar").select("*")
                query = query.ilike("mahalle", f"%{lokasyon}%")
                result = query.execute()
                listings = result.data if result.data else []
        else:
            result = query.limit(50).execute()
            listings = result.data if result.data else []

        print(f"📋 Database returned {len(listings)} listings")

        # Room filter
        if params.get('oda_sayisi') and listings:
            oda_sayisi = params['oda_sayisi'].lower()
            listings = [l for l in listings if l.get('oda_sayisi', '').lower() == oda_sayisi]

        # Price filter
        if params.get('max_fiyat') and listings:
            max_fiyat = params.get('max_fiyat')
            filtered_listings = []
            for l in listings:
                try:
                    fiyat_str = l.get('fiyat', '0')
                    fiyat_temiz = re.sub(r'[^\d]', '', str(fiyat_str))
                    if fiyat_temiz:
                        fiyat = float(fiyat_temiz)
                        if fiyat <= max_fiyat:
                            filtered_listings.append(l)
                except (ValueError, TypeError):
                    continue
            listings = filtered_listings

        # Similarity calculation
        if listings:
            for listing in listings:
                if 'embedding' in listing and listing['embedding']:
                    try:
                        embedding_raw = listing['embedding']
                        if isinstance(embedding_raw, str):
                            listing_embedding = json.loads(embedding_raw)
                        else:
                            listing_embedding = embedding_raw
                        
                        similarity = cosine_similarity(query_embedding, listing_embedding)
                        listing['similarity'] = similarity
                    except Exception:
                        listing['similarity'] = 0
                else:
                    listing['similarity'] = 0
            
            # Sort by similarity
            listings = sorted(listings, key=lambda x: x.get('similarity', 0), reverse=True)

        print(f"✅ Search completed: {len(listings)} final results")
        return listings

    except Exception as e:
        print(f"❌ Search error: {e}")
        print(traceback.format_exc())
        return []

# ============= FORMATTING =============

def format_property_listings(listings: list) -> str:
    """Format listings as HTML table"""
    if not listings:
        return "<p>Hiç ilan bulunamadı.</p>"
    
    html = f"<h3 style='color: #f44336;'>Arama Sonucu: {len(listings)} ilan bulundu</h3>"
    html += "<p style='color: #333;'><strong>📞 Detaylı bilgi için 532 687 84 64 numaralı telefonu arayabilirsiniz.</strong></p>"
    
    html += """
    <table style="width: 100%; border-collapse: collapse; margin: 20px 0;">
        <tr style="background: #1976d2; color: white;">
            <th style="padding: 10px; text-align: left;">İlan No</th>
            <th style="padding: 10px; text-align: left;">Başlık</th>
            <th style="padding: 10px; text-align: left;">Lokasyon</th>
            <th style="padding: 10px; text-align: right;">Fiyat</th>
            <th style="padding: 10px; text-align: center;">Oda</th>
            <th style="padding: 10px; text-align: center;">İşlem</th>
        </tr>
    """
    
    for i, ilan in enumerate(listings[:20]):  # Limit to 20 for performance
        ilan_no = ilan.get('ilan_id', ilan.get('ilan_no', ''))
        baslik = ilan.get('baslik', '')
        
        ilce = ilan.get('ilce', '')
        mahalle = ilan.get('mahalle', '')
        lokasyon = f"{ilce}, {mahalle}" if ilce and mahalle else (ilce or mahalle or '')
        
        fiyat = ilan.get('fiyat', '')
        oda_sayisi = ilan.get('oda_sayisi', '')
        
        row_bg = "#f8f9fa" if i % 2 == 0 else "#ffffff"
        
        html += f"""
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
        </tr>
        """
    
    html += "</table>"
    
    real_ids = [ilan.get('ilan_id') for ilan in listings if ilan.get('ilan_id')]
    if real_ids:
        html += f"<p><strong>VERİTABANINDAKİ GERÇEK İLAN NUMARALARI: {', '.join(real_ids[:10])}</strong></p>"
    
    html += "<p style='color: #333;'>Bu ilanların doğruluğunu kontrol ettim. Farklı bir arama yapmak isterseniz, lütfen kriterleri belirtiniz.</p>"
    
    return html

# ============= MAIN SEARCH FUNCTION =============

async def search_properties(query: str) -> str:
    """Main search function with caching"""
    try:
        # Check cache first
        cached_result = check_cache(query)
        if cached_result is not None:
            print(f"🚀 Cache hit for: {query}")
            return format_property_listings(cached_result)
        
        print(f"🔎 Cache miss, performing search: {query}")
        
        # Perform search
        listings = await hybrid_property_search(query)
        
        # Save to cache
        if listings:
            save_to_cache(query, listings)
        
        return format_property_listings(listings)
        
    except Exception as e:
        print(f"❌ search_properties error: {e}")
        print(traceback.format_exc())
        return "<p>Arama sırasında bir hata oluştu. Lütfen tekrar deneyin.</p>"

# ============= CLEANUP FUNCTIONS =============

def clear_all_caches():
    """Clear all caches"""
    try:
        simple_cache.clear()
        print("🧹 All caches cleared")
    except Exception as e:
        print(f"⚠️ Cache clearing error: {e}")

# ============= COMPATIBILITY FUNCTIONS =============

# Keep existing function names for compatibility
def format_context_for_sibelgpt(listings: List[Dict]) -> str:
    """Compatibility function"""
    return format_property_listings(listings)

# ============= TEST FUNCTION =============

async def test_search():
    """Test function"""
    test_query = "Kadıköy'de 3+1 daire"
    result = await search_properties(test_query)
    print("Test result:", result[:200] + "..." if len(result) > 200 else result)

if __name__ == "__main__":
    asyncio.run(test_search())
