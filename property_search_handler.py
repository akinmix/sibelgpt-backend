# property_search_handler.py
# SibelGPT için: Maksimum log, maksimum sağlamlık!

import numpy as np
import os
import json
import math
import re
import asyncio
import traceback
import hashlib
import pickle
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any

try:
    from openai import AsyncOpenAI
    from supabase import create_client
except ImportError:
    raise RuntimeError("Gerekli kütüphaneler eksik: openai veya supabase")

# ---- Cache Mekanizması Ayarları ----
CACHE_DIR = os.path.join(os.path.dirname(__file__), "cache")
CACHE_TTL = timedelta(hours=2)  # 2 saat süreyle önbellekte tut
os.makedirs(CACHE_DIR, exist_ok=True)

# ---- Ortam Değişkenleri ve API Bağlantıları ----
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

# ---- Cache İşlemleri için Fonksiyonlar ----
def get_cache_key(query: str) -> str:
    """Sorgu için benzersiz bir cache anahtarı oluştur"""
    return hashlib.md5(query.encode('utf-8')).hexdigest()

def get_cache_path(cache_key: str) -> str:
    """Cache dosyasının tam yolunu al"""
    return os.path.join(CACHE_DIR, f"{cache_key}.pkl")

def check_cache(query: str) -> list:
    """Cache'te ilgili sorgu sonucu var mı kontrol et, varsa döndür"""
    cache_key = get_cache_key(query)
    cache_path = get_cache_path(cache_key)
    
    if not os.path.exists(cache_path):
        return None
    
    try:
        # Cache dosyasının yaşını kontrol et
        file_modified = datetime.fromtimestamp(os.path.getmtime(cache_path))
        if datetime.now() - file_modified > CACHE_TTL:
            print(f"🕒 Cache süresi dolmuş: {query}")
            return None
        
        # Cache'ten oku
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
            print(f"✅ Cache'ten sonuç alındı: {query}")
            return cached_data
    except Exception as e:
        print(f"⚠️ Cache okuma hatası: {e}")
        return None

def save_to_cache(query: str, data: list) -> None:
    """Sorgu sonucunu cache'e kaydet"""
    if not data:
        return  # Boş sonuçları önbelleğe alma
    
    cache_key = get_cache_key(query)
    cache_path = get_cache_path(cache_key)
    
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump(data, f)
            print(f"💾 Sonuç cache'e kaydedildi: {query}")
    except Exception as e:
        print(f"⚠️ Cache yazma hatası: {e}")

# --- Embedding çekme ---
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
        print(traceback.format_exc())
        return None

# --- Benzerlik hesaplama ---
def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    try:
        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        magnitude1 = math.sqrt(sum(a * a for a in vec1))
        magnitude2 = math.sqrt(sum(b * b for b in vec2))
        if magnitude1 * magnitude2 == 0:
            return 0
        return dot_product / (magnitude1 * magnitude2)
    except Exception as exc:
        print(f"❌ Cosine similarity hatası: {exc}")
        print(traceback.format_exc())
        return 0

# --- Sorgu tipi anlama ---
def is_property_search_query(query: str) -> bool:
    try:
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
    except Exception as exc:
        print(f"❌ Sorgu analiz hatası: {exc}")
        print(traceback.format_exc())
        return False

# --- Parametre Çıkarma ---
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

# --- Ana Arama Fonksiyonu ---
async def hybrid_property_search(question: str) -> List[Dict]:
    try:
        # Parametreleri çıkarma ve embedding oluşturmayı paralel yap
        params_task = extract_query_parameters(question)
        embedding_task = get_embedding(question)
        
        # Her iki görevi de bekleyelim
        params, query_embedding = await asyncio.gather(params_task, embedding_task)
        
        print(f"🔍 Çıkarılan parametreler: {params}")
        
        if not query_embedding:
            print("⚠️ Embedding oluşturulamadı!")
            return []
            
        # Veritabanı sorgusuna devam
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
                    fiyat_temiz = re.sub(r'[^\d]', '', fiyat_str)
                    print(f"↪️ temiz fiyat: {fiyat_temiz!r}")
                    fiyat = float(fiyat_temiz)
                    if fiyat <= max_fiyat:
                        filtered_listings.append(l)
                except (ValueError, TypeError) as err:
                    print(f"Fiyat float dönüştürme hatası: {fiyat_str!r} -> {fiyat_temiz!r} ({err})")
                    print(traceback.format_exc())
            listings = filtered_listings

        print(f"📋 Veritabanı sorgusu {len(listings)} ilan buldu")

        # Embedding ile benzerlik skoru
        if listings:
            query_embedding_np = np.array(query_embedding, dtype=np.float32)
            for listing in listings:
                if 'embedding' in listing and listing['embedding']:
                    embedding_raw = listing['embedding']
                    if isinstance(embedding_raw, str):
                        try:
                            listing_embedding = json.loads(embedding_raw)
                        except Exception as e:
                            print("Embedding JSON decode hatası:", e)
                            print(traceback.format_exc())
                            listing_embedding = []
                    else:
                        listing_embedding = embedding_raw
                    try:
                        listing_embedding_np = np.array(listing_embedding, dtype=np.float32)
                        similarity = cosine_similarity(query_embedding_np, listing_embedding_np)
                        listing['similarity'] = similarity
                    except Exception as emb_err:
                        print(f"Benzerlik hesaplama hatası: {emb_err}")
                        print(traceback.format_exc())
                        listing['similarity'] = 0
                else:
                    listing['similarity'] = 0
            listings = sorted(listings, key=lambda x: x.get('similarity', 0), reverse=True)

        print(f"✅ Hibrit arama sonuçları: {len(listings)} ilan bulundu")
        
        if listings:
            ilan_ids = [listing.get('ilan_id') for listing in listings[:10] if listing.get('ilan_id')]
            print(f"🏷️ Bulunan ilk 10 ilan ID: {ilan_ids}")

        return listings

    except Exception as e:
        print(f"❌ Hibrit arama hatası: {e}")
        print(traceback.format_exc())
        return []

def format_property_listings(listings: list) -> str:
    """İlan sonuçlarını HTML tabloya çevir"""
    if not listings:
        return "<p>Hiç ilan bulunamadı.</p>"
    
    # Başlık: Arama sonuçları sayısı
    html = f"<h3 style='color: #f44336;'>Arama Sonucu: {len(listings)} ilan bulundu</h3>"
    
    # Telefon bilgisi - metin rengini belirtelim
    html += "<p style='color: #333;'><strong>📞 Sorgunuzla ilgili ilanlar aşağıda listelenmiştir. Detaylı bilgi için 532 687 84 64 numaralı telefonu arayabilirsiniz.</strong></p>"
    
    # Tablo başlangıcı
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
    
    # Satırları ekle
    for i, ilan in enumerate(listings[:50]):
        # Temel veri alanlarını al
        ilan_no = ilan.get('ilan_id', ilan.get('ilan_no', ''))
        baslik = ilan.get('baslik', '')
        
        # Lokasyon bilgisini birleştir
        ilce = ilan.get('ilce', '')
        mahalle = ilan.get('mahalle', '')
        lokasyon = f"{ilce}, {mahalle}" if ilce and mahalle else (ilce or mahalle or '')
        
        fiyat = ilan.get('fiyat', '')
        oda_sayisi = ilan.get('oda_sayisi', '')
        
        # PDF butonu
        pdf_link = f"<a href='https://sibelgpt-backend.onrender.com/generate-property-pdf/{ilan_no}' target='_blank' style='display: inline-block; padding: 6px 12px; background-color: #1976d2; color: white; text-decoration: none; border-radius: 4px;'>PDF</a>"
        
        # Satır arka plan rengi
        row_bg = "#f8f9fa" if i % 2 == 0 else "#ffffff"
        
        # BURADA METİN RENGİNİ AÇIKÇA BELİRTİYORUZ: color: black;
        html += f"""
        <tr style="background-color: {row_bg};">
            <td style="padding: 10px; border-bottom: 1px solid #eee; color: black;">{ilan_no}</td>
            <td style="padding: 10px; border-bottom: 1px solid #eee; color: black;">{baslik}</td>
            <td style="padding: 10px; border-bottom: 1px solid #eee; color: black;">{lokasyon}</td>
            <td style="padding: 10px; border-bottom: 1px solid #eee; text-align: right; color: black;">{fiyat}</td>
            <td style="padding: 10px; border-bottom: 1px solid #eee; text-align: center; color: black;">{oda_sayisi}</td>
            <td style="padding: 10px; border-bottom: 1px solid #eee; text-align: center;">{pdf_link}</td>
        </tr>
        """
    
    # Tabloyu kapat
    html += "</table>"
    
    # Kapanış metni
    html += "<p style='color: #333;'>Bu ilanların doğruluğunu kontrol ettim. Farklı bir arama yapmak isterseniz, lütfen kriterleri belirtiniz.</p>"
    
    return html

# --- Ana arama fonksiyonu: Dışarıdan çağrılır ---
async def search_properties(query: str) -> str:
    try:
        # Önce cache'i kontrol et
        cached_listings = check_cache(query)
        if cached_listings is not None:
            print(f"🚀 Önbellekten hızlı yanıt: {query}")
            return format_property_listings(cached_listings)
        
        print(f"🔎 Önbellekte bulunamadı, arama yapılıyor: {query}")
        # Cache'te yoksa normal aramayı yap
        listings = await hybrid_property_search(query)
        
        # Sonuçları cache'e kaydet
        save_to_cache(query, listings)
        
        return format_property_listings(listings)
    except Exception as e:
        print(f"❌ search_properties hatası: {e}")
        print(traceback.format_exc())
        return "<p>Bir hata oluştu.</p>"

# --- Kendi başına test etmek istersen ---
async def test_search():
    soru = "Kadıköy'de 20 milyona kadar 3+1 daire"
    html = await search_properties(soru)
    print(html)

if __name__ == "__main__":
    asyncio.run(test_search())
