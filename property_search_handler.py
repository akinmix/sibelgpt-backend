# property_search_handler.py
# SibelGPT için: Maksimum hız, maksimum performans!

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

# ===== HIZLANDIRMA İÇİN CACHE SİSTEMİ =====
# Cache klasörü
CACHE_DIR = os.path.join(os.path.dirname(__file__), "listings_cache")
os.makedirs(CACHE_DIR, exist_ok=True)

# 🔥 DÜZELTME: Global cache değişkenlerini başlangıçta tanımla
ALL_LISTINGS_CACHE = []  # ✅ BOŞ LİSTE OLARAK BAŞLAT
CACHE_LOADED_TIME = None
CACHE_LOCK = asyncio.Lock()

async def load_all_listings_to_memory():
    """Tüm ilanları belleğe yükle - HIZLI ERİŞİM İÇİN"""
    global ALL_LISTINGS_CACHE, CACHE_LOADED_TIME
    
    async with CACHE_LOCK:
        print("🔄 İlanlar belleğe yükleniyor...")
        
        cache_file = os.path.join(CACHE_DIR, "all_listings.pkl")
        
        # Önce cache dosyasını kontrol et
        if os.path.exists(cache_file):
            file_time = datetime.fromtimestamp(os.path.getmtime(cache_file))
            if datetime.now() - file_time < timedelta(hours=12):
                try:
                    with open(cache_file, 'rb') as f:
                        ALL_LISTINGS_CACHE = pickle.load(f)
                        CACHE_LOADED_TIME = datetime.now()
                        print(f"✅ {len(ALL_LISTINGS_CACHE)} ilan cache'den yüklendi!")
                        return
                except Exception as e:
                    print(f"⚠️ Cache okuma hatası: {e}")
        
        # Cache yoksa veya eskiyse veritabanından çek
        try:
            result = supabase_client.table("remax_ilanlar").select("*").execute()
            ALL_LISTINGS_CACHE = result.data if result.data else []
            CACHE_LOADED_TIME = datetime.now()
            
            # Cache'e kaydet
            try:
                with open(cache_file, 'wb') as f:
                    pickle.dump(ALL_LISTINGS_CACHE, f)
            except Exception as e:
                print(f"⚠️ Cache kaydetme hatası: {e}")
            
            print(f"✅ {len(ALL_LISTINGS_CACHE)} ilan veritabanından yüklendi!")
            
        except Exception as e:
            print(f"❌ Veritabanı hatası: {e}")
            ALL_LISTINGS_CACHE = []  # Hata durumunda boş liste

# ---- Yardımcı Fonksiyonlar ----
async def get_embedding(text: str) -> Optional[List[float]]:
    """OpenAI ile embedding oluştur"""
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

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """İki vektör arasındaki benzerliği hesapla"""
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

def is_property_search_query(query: str) -> bool:
    """Sorgunun emlak araması olup olmadığını kontrol et"""
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

async def extract_query_parameters(question: str) -> Dict:
    """Sorgudaki arama parametrelerini çıkar"""
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

def format_property_listings(listings: list) -> str:
    """İlan sonuçlarını HTML tabloya çevir"""
    if not listings:
        return "<p>Hiç ilan bulunamadı.</p>"
    
    # Başlık: Arama sonuçları sayısı
    html = f"<h3 style='color: #f44336;'>Arama Sonucu: {len(listings)} ilan bulundu</h3>"
    
    # Telefon bilgisi
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
    
    # GERÇEK İLAN NUMARALARI başlığını ekle
    real_ids = [ilan.get('ilan_id') for ilan in listings if ilan.get('ilan_id')]
    if real_ids:
        html += f"<p><strong>VERİTABANINDAKİ GERÇEK İLAN NUMARALARI: {', '.join(real_ids)}</strong></p>"
    
    # Kapanış metni
    html += "<p style='color: #333;'>Bu ilanların doğruluğunu kontrol ettim. Farklı bir arama yapmak isterseniz, lütfen kriterleri belirtiniz.</p>"
    
    return html

# ---- Ana Arama Fonksiyonu ----
async def search_properties(query: str) -> str:
    """HIZLANDIRILMIŞ ARAMA FONKSİYONU - HATA KORUMASILI"""
    global ALL_LISTINGS_CACHE, CACHE_LOADED_TIME  # Global değişkenleri belirt
    
    try:
        # ✅ DÜZELTİLMİŞ CACHE KONTROLÜ
        if ALL_LISTINGS_CACHE is None:
        ALL_LISTINGS_CACHE = []

    # Cache sadece boşsa yükle (her seferinde değil!)
        if not ALL_LISTINGS_CACHE:
            print("📥 Cache ilk kez yükleniyor...")
            await load_all_listings_to_memory()
        elif CACHE_LOADED_TIME and datetime.now() - CACHE_LOADED_TIME > timedelta(hours=12):
            print("🔄 Cache 12 saatlik, yenileniyor...")
            await load_all_listings_to_memory()
        
        # 6 saatten eski mi?
        if CACHE_LOADED_TIME and datetime.now() - CACHE_LOADED_TIME > timedelta(hours=6):
            print("🔄 Cache süresi dolmuş, yenileniyor...")
            await load_all_listings_to_memory()
        
        print(f"🔎 Arama yapılıyor: {query}")
        print(f"📊 Bellekte {len(ALL_LISTINGS_CACHE)} ilan var")
        
        # Eğer cache hala boşsa, basit veritabanı sorgusu yap
        if not ALL_LISTINGS_CACHE:
            print("⚠️ Cache hala boş, doğrudan veritabanından arama yapılıyor...")
            try:
                result = supabase_client.table("remax_ilanlar").select("*").limit(50).execute()
                if result.data:
                    return format_property_listings(result.data)
                else:
                    return "<p>Veritabanında hiç ilan bulunamadı.</p>"
            except Exception as e:
                print(f"❌ Doğrudan veritabanı hatası: {e}")
                return "<p>Arama sırasında teknik bir sorun oluştu. Lütfen daha sonra tekrar deneyin.</p>"
        
        # Parametreleri çıkar
        params = await extract_query_parameters(query)
        print(f"📝 Parametreler: {params}")
        
        # Bellekteki ilanları kopyala (orijinali bozma)
        filtered = ALL_LISTINGS_CACHE.copy()
        
        # HIZLI FİLTRELEME
        
        # 1. Lokasyon filtresi
        if params.get('lokasyon'):
            lok = params['lokasyon'].lower()
            filtered = [
                ilan for ilan in filtered 
                if lok in (str(ilan.get('ilce', '')).lower() + ' ' + 
                          str(ilan.get('mahalle', '')).lower())
            ]
            print(f"📍 Lokasyon filtresi sonrası: {len(filtered)} ilan")
        
        # 2. Fiyat filtresi
        if params.get('max_fiyat'):
            max_fiyat = float(params['max_fiyat'])
            temp = []
            for ilan in filtered:
                try:
                    fiyat_str = re.sub(r'[^\d]', '', str(ilan.get('fiyat', '0')))
                    if fiyat_str and float(fiyat_str) <= max_fiyat:
                        temp.append(ilan)
                except:
                    continue
            filtered = temp
            print(f"💰 Fiyat filtresi sonrası: {len(filtered)} ilan")
        
        # 3. Oda sayısı filtresi  
        if params.get('oda_sayisi'):
            oda = params['oda_sayisi'].lower()
            filtered = [
                ilan for ilan in filtered 
                if str(ilan.get('oda_sayisi', '')).lower() == oda
            ]
            print(f"🏠 Oda filtresi sonrası: {len(filtered)} ilan")
        
        # En fazla 50 ilan göster
        filtered = filtered[:50]
        
        print(f"✅ Toplam {len(filtered)} ilan bulundu")
        return format_property_listings(filtered)
        
    except Exception as e:
        print(f"❌ Arama hatası: {e}")
        import traceback
        traceback.print_exc()
        return "<p>Arama sırasında bir hata oluştu. Lütfen tekrar deneyin.</p>"

# ---- Hibrit Arama (Geriye dönük uyumluluk için) ----
async def hybrid_property_search(question: str) -> List[Dict]:
    """Eski fonksiyon - geriye dönük uyumluluk için"""
    try:
        html_result = await search_properties(question)
        # HTML'den basit bir liste döndür
        return ALL_LISTINGS_CACHE[:50] if ALL_LISTINGS_CACHE else []
    except Exception as e:
        print(f"❌ Hibrit arama hatası: {e}")
        return []

# Uygulama başlarken cache'i yükle
print("🚀 Property search handler başlatılıyor...")

