# property_search_handler.py
# SibelGPT Emlak İlan Arama Modülü

import numpy as np
import os
import json
import math
import re
import asyncio
import traceback
from typing import List, Dict, Optional, Tuple, Any

# Supabase ve OpenAI bağlantıları için gerekli importlar
try:
    from openai import AsyncOpenAI
    from supabase import create_client
except ImportError:
    raise RuntimeError("Gerekli kütüphaneler eksik: openai veya supabase")

# ── Ortam Değişkenleri ─────────────────────────────────────
OAI_KEY = os.getenv("OPENAI_API_KEY")
SB_URL = os.getenv("SUPABASE_URL")
SB_KEY = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY")

if not all([OAI_KEY, SB_URL, SB_KEY]):
    raise RuntimeError("Eksik API anahtarı veya Supabase bağlantı bilgisi.")

# İstemcileri oluştur
openai_client = AsyncOpenAI(api_key=OAI_KEY)
supabase_client = create_client(SB_URL, SB_KEY)

# ── Ayarlar ────────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-small"
MATCH_THRESHOLD = 0.3
MATCH_COUNT = 50

# ── Temel Fonksiyonlar ───────────────────────────────────

async def get_embedding(text: str) -> Optional[List[float]]:
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
        print(f"❌ Embedding hatası: {exc}")
        return None

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    if magnitude1 * magnitude2 == 0:
        return 0
    return dot_product / (magnitude1 * magnitude2)

def is_property_search_query(query: str) -> bool:
    query_lower = query.lower()
    search_terms = [
        "ev", "daire", "konut", "villa", "apart", "stüdyo", "rezidans",
        "ara", "bul", "göster", "ilan", "satılık", "kiralık", "emlak",
        "mahalle", "ilçe", "bölge", "oda", "metrekare", "m2", "fiyat", "tl", "₺"
    ]
    search_patterns = [
        r'\d+\+\d+', r'\d+\s*milyon', r'\d+\s*[mM]²', r'kaç\s*[mM]²', r'\d+\s*oda',
        r'kaç\s*oda', r'kadar\s*fiyat', r'bütçe[m\s]', r'en\s*ucuz', r'en\s*pahalı',
        r'ara[nıtyp]*(m?a)', r'bul[a-z]*(m?a)', r'göster[a-z]*(m?e)',
    ]
    for term in search_terms:
        if term in query_lower:
            return True
    for pattern in search_patterns:
        if re.search(pattern, query_lower):
            return True
    return False

# ── Parametre Çıkarma ve SQL Oluşturma ────────────────────

async def extract_query_parameters(question: str) -> Dict:
    try:
        print(f"🔍 Sorgudan parametreler çıkarılıyor: {question}")
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
                    - bulundugu_kat: Kat bilgisi
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
        print(f"✅ Çıkarılan parametreler: {parameters}")
        return parameters
    except Exception as e:
        print(f"❌ Parametre çıkarma hatası: {e}")
        print(traceback.format_exc())
        return {}

# ── Hibrit Arama Fonksiyonu ───────────────────────────────

async def hybrid_property_search(question: str) -> List[Dict]:
    try:
        params = await extract_query_parameters(question)
        print(f"🔍 Çıkarılan parametreler: {params}")

        query = supabase_client.table("remax_ilanlar").select("*")
        # Lokasyon filtresi
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

        # Oda sayısı filtresi
        if params.get('oda_sayisi') and listings:
            oda_sayisi = params['oda_sayisi'].lower()
            listings = [l for l in listings if l.get('oda_sayisi', '').lower() == oda_sayisi]

        # Max fiyat filtresi
        if params.get('max_fiyat') and listings:
            max_fiyat = params.get('max_fiyat')
            filtered_listings = []
            for l in listings:
                try:
                    fiyat_str = l.get('fiyat', '0')
                    fiyat_temiz = re.sub(r'[^0-9.]', '', fiyat_str.replace(',', '.'))
                    if fiyat_temiz:
                        fiyat = float(fiyat_temiz)
                        if fiyat <= max_fiyat:
                            filtered_listings.append(l)
                except (ValueError, TypeError):
                    filtered_listings.append(l)
            listings = filtered_listings

        print(f"📋 Veritabanı sorgusu {len(listings)} ilan buldu")

        # 3. Embedding ile benzerlik skoru
        query_embedding = await get_embedding(question)

        if query_embedding and listings:
            query_embedding_np = np.array(query_embedding, dtype=np.float32)
            for listing in listings:
                if 'embedding' in listing and listing['embedding']:
                    embedding_raw = listing['embedding']
                    if isinstance(embedding_raw, str):
                        try:
                            listing_embedding = json.loads(embedding_raw)
                        except Exception as e:
                            print("Embedding JSON decode hatası:", e)
                            listing_embedding = []
                    else:
                        listing_embedding = embedding_raw
                    listing_embedding_np = np.array(listing_embedding, dtype=np.float32)
                    similarity = cosine_similarity(query_embedding_np, listing_embedding_np)
                    listing['similarity'] = similarity
                else:
                    listing['similarity'] = 0
            listings = sorted(listings, key=lambda x: x.get('similarity', 0), reverse=True)

        print(f"✅ Hibrit arama sonuçları: {len(listings)} ilan bulundu")
        ilan_ids = [listing.get('ilan_id') for listing in listings[:10] if listing.get('ilan_id')]
        print(f"🏷️ Bulunan ilk 10 ilan ID: {ilan_ids}")

        return listings

    except Exception as e:
        print(f"❌ Hibrit arama hatası: {e}")
        print(traceback.format_exc())
        return []

# ── İlan Formatlama Fonksiyonu (değişmedi) ────────────────
# ... (Senin mevcut format_property_listings fonksiyonunu buraya ekle, yukarıda doğru ve uzun haliyle aynen bırakabilirsin.)

# ── Ana Arama Fonksiyonu (değişmedi) ─────────────────────
# ... (search_properties ve test_search fonksiyonları yukarıdaki gibi aynı kalabilir.)

# ── Dosya doğrudan çalıştırılırsa ────────────────────────
if __name__ == "__main__":
    asyncio.run(test_search())
