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
    """Verilen metni vektör embedding'e dönüştürür."""
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
    """İki vektör arasındaki kosinüs benzerliğini hesaplar."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    magnitude1 = math.sqrt(sum(a * a for a in vec1))
    magnitude2 = math.sqrt(sum(b * b for b in vec2))
    
    if magnitude1 * magnitude2 == 0:
        return 0
    
    return dot_product / (magnitude1 * magnitude2)

def is_property_search_query(query: str) -> bool:
    """Sorgunun bir ilan araması olup olmadığını tespit eder."""
    query_lower = query.lower()
    
    # İlan arama ifadeleri
    search_terms = [
        "ev", "daire", "konut", "villa", "apart", "stüdyo", "rezidans", 
        "ara", "bul", "göster", "ilan", "satılık", "kiralık", "emlak",
        "mahalle", "ilçe", "bölge", "oda", "metrekare", "m2", "fiyat", "tl", "₺"
    ]
    
    # İlan arama soru kalıpları
    search_patterns = [
        r'\d+\+\d+',                  # 1+1, 2+1, 3+1, 4+1, 4+2 vb.
        r'\d+\s*milyon',              # 1 milyon, 2 milyon vb.
        r'\d+\s*[mM]²',               # 100 m², 150m² vb.
        r'kaç\s*[mM]²',               # kaç m² vb.
        r'\d+\s*oda',                 # 2 oda, 3 oda vb.
        r'kaç\s*oda',                 # kaç oda vb.
        r'kadar\s*fiyat',             # kadar fiyat
        r'bütçe[m\s]',                # bütçem, bütçe vb.
        r'en\s*ucuz',                 # en ucuz
        r'en\s*pahalı',               # en pahalı
        r'ara[nıtyp]*(m?a)',          # arayım, arat, aratma vb.
        r'bul[a-z]*(m?a)',            # bulalım, bul, bulma vb.
        r'göster[a-z]*(m?e)',         # göster, gösterme vb.
    ]
    
    # Terim kontrolü
    for term in search_terms:
        if term in query_lower:
            return True
    
    # Kalıp kontrolü
    for pattern in search_patterns:
        if re.search(pattern, query_lower):
            return True
    
    return False

# ── Parametre Çıkarma ve SQL Oluşturma ────────────────────

async def extract_query_parameters(question: str) -> Dict:
    """Kullanıcının sorusundan arama parametrelerini çıkarır."""
    try:
        print(f"🔍 Sorgudan parametreler çıkarılıyor: {question}")
        
        # OpenAI API'yi kullanarak parametreleri çıkar
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

                    Örnek:
                    Soru: "Kadıköy'de 2 milyon TL'ye kadar 2+1 daire"
                    Yanıt: {"lokasyon": "Kadıköy", "max_fiyat": 2000000, "oda_sayisi": "2+1"}

                    Parametreler çıkaramadığın alanlar için null döndür. Mesela kullanıcı oda sayısı belirtmediyse "oda_sayisi": null şeklinde.
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
        # Hata durumunda boş bir sözlük döndür
        return {}

# ── Hibrit Arama Fonksiyonu ───────────────────────────────

async def hybrid_property_search(question: str) -> List[Dict]:
    """Hibrit arama yöntemi ile emlak ilanlarını arar."""
    try:
        # 1. Sorgudan parametreleri çıkar
        params = await extract_query_parameters(question)
        
        print(f"🔍 Çıkarılan parametreler: {params}")
        
        # 2. Basit SQL sorgusu oluştur
        # Supabase'in filtreleme fonksiyonlarını kullanarak sorgu yapalım
        query = supabase_client.table("remax_ilanlar").select("*")
        
        # Lokasyon filtresi
        if params.get('lokasyon'):
            lokasyon = params['lokasyon'].lower()
            # İlçe'de ara
            query = query.ilike("ilce", f"%{lokasyon}%")
            
            # Sonuçları çek
            result = query.execute()
            listings = result.data
            
            # Eğer ilçede sonuç bulunamadıysa, mahallede ara
            if not listings:
                query = supabase_client.table("remax_ilanlar").select("*")
                query = query.ilike("mahalle", f"%{lokasyon}%")
                result = query.execute()
                listings = result.data
            
        else:
            # Lokasyon yoksa tüm ilanları getir (limit ile)
            result = query.limit(50).execute()
            listings = result.data if result.data else []
            
        # Oda sayısı filtresi
        if params.get('oda_sayisi') and listings:
            # Oda sayısı filtresini memory'de yapalım
            oda_sayisi = params['oda_sayisi'].lower()
            listings = [l for l in listings if l.get('oda_sayisi', '').lower() == oda_sayisi]
        
        # Max fiyat filtresi (eğer varsa)
        if params.get('max_fiyat') and listings:
            max_fiyat = params.get('max_fiyat')
            # Basit bir yaklaşımla memory'de filtreleyelim
            filtered_listings = []
            for l in listings:
                try:
                    fiyat_str = l.get('fiyat', '0')
                    # Rakam ve nokta dışındaki karakterleri kaldır
                    fiyat_temiz = re.sub(r'[^0-9.]', '', fiyat_str.replace(',', '.'))
                    if fiyat_temiz:
                        fiyat = float(fiyat_temiz)
                        if fiyat <= max_fiyat:
                            filtered_listings.append(l)
                except (ValueError, TypeError):
                    # Hatalı fiyat verisi varsa, ilanı dahil et
                    filtered_listings.append(l)
            
            listings = filtered_listings
            
        print(f"📋 Veritabanı sorgusu {len(listings)} ilan buldu")
        
        # 3. Embedding'i hesapla ve sonuçları sırala
        query_embedding = await get_embedding(question)
        
        if query_embedding and listings:
            # Her listing için benzerlik skoru hesapla
            for listing in listings:
                if 'embedding' in listing and listing['embedding']:
                    listing_embedding = listing['embedding']
                    
                    # Kosinüs benzerliği hesapla
                    query_embedding = np.array(query_embedding, dtype=np.float32)
                    listing_embedding = np.array(listing_embedding, dtype=np.float32)
                    similarity = cosine_similarity(query_embedding, listing_embedding)
                    listing['similarity'] = similarity
                else:
                    listing['similarity'] = 0
            
            # Benzerliklere göre sırala
            listings = sorted(listings, key=lambda x: x.get('similarity', 0), reverse=True)
        
        print(f"✅ Hibrit arama sonuçları: {len(listings)} ilan bulundu")
        
        # 4. İlan ID'leri listesini oluştur (loglamak için)
        ilan_ids = [listing.get('ilan_id') for listing in listings[:10] if listing.get('ilan_id')]
        print(f"🏷️ Bulunan ilk 10 ilan ID: {ilan_ids}")
        
        return listings
        
    except Exception as e:
        print(f"❌ Hibrit arama hatası: {e}")
        print(traceback.format_exc())
        return []

# ── İlan Formatlama Fonksiyonu ───────────────────────────

def format_property_listings(listings: List[Dict]) -> str:
    """İlanları HTML tablo olarak formatlar."""
    if not listings:
        return "<p>🔍 Bu kriterlere uygun ilan bulunamadı. Lütfen farklı arama kriterleri deneyiniz.</p>"
    
    # Locale ayarı
    try:
        import locale
        try:
            locale.setlocale(locale.LC_ALL, 'tr_TR.UTF-8')
        except locale.Error:
            try:
                locale.setlocale(locale.LC_ALL, 'tr_TR')
            except locale.Error:
                pass
    except ImportError:
        pass
    
    MAX_LISTINGS_TO_SHOW = 20
    listings_to_format = listings[:MAX_LISTINGS_TO_SHOW]
    
    output = "<p><strong>📞 Sorgunuzla ilgili ilanlar aşağıdaki tabloda listelenmiştir. Detaylı bilgi için 532 687 84 64 numaralı telefonu arayabilirsiniz.</strong></p>"
    
    # Responsive ve modern tablo stili
    output += """
    <style>
    .property-table {
        width: 100%;
        border-collapse: collapse;
        margin: 20px 0;
        font-family: 'Segoe UI', Arial, sans-serif;
        box-shadow: 0 2px 15px rgba(0,0,0,0.1);
        border-radius: 8px;
        overflow: hidden;
    }
    
    .property-table th {
        background-color: #1976d2;
        color: white;
        padding: 12px 15px;
        text-align: left;
        font-weight: 600;
    }
    
    .property-table tr {
        border-bottom: 1px solid #dddddd;
    }
    
    .property-table tr:nth-of-type(even) {
        background-color: #f3f3f3;
    }
    
    .property-table tr:last-of-type {
        border-bottom: 2px solid #1976d2;
    }
    
    .property-table td {
        padding: 12px 15px;
        vertical-align: top;
    }
    
    .property-table .btn-pdf {
        display: inline-block;
        padding: 6px 12px;
        background: #1976d2;
        color: white;
        border: none;
        border-radius: 25px;
        cursor: pointer;
        font-size: 13px;
        font-weight: 500;
        text-decoration: none;
        transition: all 0.3s ease;
    }
    
    .property-table .btn-pdf:hover {
        background: #115293;
        transform: translateY(-1px);
        box-shadow: 0 2px 5px rgba(0,0,0,0.2);
    }
    
    @media screen and (max-width: 600px) {
        .property-table {
            display: block;
            overflow-x: auto;
        }
    }
    </style>
    """
    
    # Tablo başlangıcı
    output += """
    <table class="property-table">
        <thead>
            <tr>
                <th>Detaylar</th>
                <th>Lokasyon</th>
                <th>Fiyat</th>
                <th>Özellikler</th>
                <th>İşlem</th>
            </tr>
        </thead>
        <tbody>
    """
    
    # Tablo içeriği
    for i, listing in enumerate(listings_to_format, start=1):
        ilan_no = listing.get('ilan_id', f"ilan-{i}")
        baslik = listing.get('baslik', '(başlık yok)')
        
        # Lokasyon bilgisi
        ilce = listing.get('ilce', '')
        mahalle = listing.get('mahalle', '')
        lokasyon = listing.get('lokasyon', '')
        if ilce and mahalle:
            lokasyon_str = f"{ilce} / {mahalle}"
        elif ilce:
            lokasyon_str = ilce
        elif mahalle:
            lokasyon_str = mahalle
        elif lokasyon:
            lokasyon_str = lokasyon
        else:
            lokasyon_str = "(lokasyon bilgisi yok)"
        
        # Fiyat bilgisi
        fiyat = listing.get('fiyat', '')
        if fiyat:
            try:
                fiyat_str = fiyat
                if isinstance(fiyat, (int, float)):
                    fiyat_str = f"{fiyat:,.0f} ₺".replace(',', '.').replace('.', ',')
            except:
                fiyat_str = str(fiyat)
        else:
            fiyat_str = "(fiyat bilgisi yok)"
        
        # Özellikler
        ozellikler = []
        
        # Oda sayısı
        oda_sayisi = listing.get('oda_sayisi', '')
        if oda_sayisi:
            ozellikler.append(oda_sayisi)
        
        # Metrekare
        metrekare = listing.get('metrekare', '')
        if metrekare:
            if not str(metrekare).endswith('m²'):
                ozellikler.append(f"{metrekare} m²")
            else:
                ozellikler.append(str(metrekare))
        
        # Kat bilgisi
        kat = listing.get('bulundugu_kat', '')
        if kat:
            try:
                kat_int = int(kat)
                if kat_int == 0:
                    ozellikler.append("Giriş Kat")
                elif kat_int < 0:
                    ozellikler.append(f"Bodrum Kat ({kat_int})")
                else:
                    ozellikler.append(f"{kat_int}. Kat")
            except:
                ozellikler.append(str(kat))
        
        ozellikler_str = " | ".join(ozellikler) if ozellikler else "(özellik bilgisi yok)"
        
        # Tablo satırı
        output += f"""
        <tr>
            <td>
                <strong>{baslik}</strong><br>
                <small>İlan No: {ilan_no}</small>
            </td>
            <td>{lokasyon_str}</td>
            <td><strong>{fiyat_str}</strong></td>
            <td>{ozellikler_str}</td>
            <td>
                <a href="https://sibelgpt-backend.onrender.com/generate-property-pdf/{ilan_no}" target="_blank" class="btn-pdf">
                    <i class="fas fa-file-pdf" style="margin-right: 5px;"></i> PDF İndir
                </a>
            </td>
        </tr>
        """
    
    # Tablo sonu
    output += """
        </tbody>
    </table>
    """
    
    # İlan ID'leri (debugging için, isterseniz kaldırabilirsiniz)
    real_ids = [listing.get('ilan_id') for listing in listings_to_format if listing.get('ilan_id')]
    output += f"<p><strong>VERİTABANINDAKİ GERÇEK İLAN NUMARALARI: {', '.join(real_ids)}</strong></p>"
    
    output += "<p>Farklı bir arama yapmak isterseniz, lütfen kriterleri belirtiniz.</p>"
    
    return output

# ── Ana Arama Fonksiyonu ─────────────────────────────────

async def search_properties(question: str) -> str:
    """Ana ilan arama fonksiyonu."""
    try:
        print(f"🔎 İlan araması: '{question}'")
        
        # İlan araması olup olmadığını kontrol et
        if not is_property_search_query(question):
            print("⚠️ Bu bir ilan araması değil!")
            return "<p>Bu sorgu bir ilan araması olarak tespit edilmedi. Lütfen ilan araması için daha spesifik bir soru sorun.</p>"
        
        # Hibrit arama yap
        listings = await hybrid_property_search(question)
        
        # Sonuçları formatla
        formatted_results = format_property_listings(listings)
        
        return formatted_results
    
    except Exception as e:
        print(f"❌ İlan arama hatası: {e}")
        print(traceback.format_exc())
        return "<p>İlan araması sırasında bir hata oluştu. Lütfen daha sonra tekrar deneyin.</p>"

# ── Test Fonksiyonu ──────────────────────────────────────

async def test_search():
    """Modülü test etmek için basit bir fonksiyon."""
    test_query = input("Test sorgusu girin: ")
    result = await search_properties(test_query)
    print("\n========== ARAMA SONUÇLARI ==========\n")
    print(result)

# ── Eğer dosya doğrudan çalıştırılıyorsa ─────────────────

if __name__ == "__main__":
    asyncio.run(test_search())
