import os
import asyncio 
import json
import re
from typing import List, Dict, Any, Optional

# Gerekli kütüphaneleri import et
from openai import AsyncOpenAI
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("UYARI: supabase-py kütüphanesi yüklü değil. Veritabanı işlemleri çalışmayacak.")

# ---- Ortam Değişkenleri ve Bağlantılar ----
OAI_KEY = os.getenv("OPENAI_API_KEY")
SB_URL  = os.getenv("SUPABASE_URL")
SB_ANON_KEY = os.getenv("SUPABASE_KEY") # Güvenli olduğunu teyit etmiştik.

if not all([OAI_KEY, SB_URL, SB_ANON_KEY]):
    raise RuntimeError("Eksik API anahtarı veya Supabase bağlantı bilgisi (URL ve KEY).")

openai_client = AsyncOpenAI(api_key=OAI_KEY)
supabase: Optional[Client] = None
if SUPABASE_AVAILABLE:
    supabase = create_client(SB_URL, SB_ANON_KEY)

# ---- Ayarlar ----
EMBEDDING_MODEL = "text-embedding-3-small"
MATCH_THRESHOLD = 0.4  # Daha alakalı sonuçlar için eşik
MATCH_COUNT = 25

# ==============================================================================
# ==================== HIZLI VE DOĞRU ARAMA MİMARİSİ (v2) ======================
# ==============================================================================

async def get_embedding(text: str) -> Optional[List[float]]:
    """Metin için OpenAI embedding'i oluşturur."""
    try:
        resp = await openai_client.embeddings.create(model=EMBEDDING_MODEL, input=[text.strip()])
        return resp.data[0].embedding
    except Exception as e:
        print(f"❌ Embedding hatası: {e}")
        return None

async def extract_filters_from_query(question: str) -> Dict:
    """Sorgudan SADECE yapısal filtreleri çıkarır (Hızlı ve Akıllı Versiyon)."""
    print(f"🔍 Akıllı filtre çıkarma işlemi başlatıldı: {question}")
    try:
        resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """Sen bir emlak arama asistanısın. Kullanıcının sorgusundan SADECE şu filtreleri JSON olarak çıkar: "min_fiyat", "max_fiyat", "oda_sayisi", ve "lokasyon" (TÜM ilçe/mahalle adları). 'ilce'/'mahalle' diye ayırma, sadece 'lokasyon' kullan. Örnek: "kadıköyde 5 milyona kadar 2+1 daire" -> {"max_fiyat": 5000000, "oda_sayisi": "2+1", "lokasyon": "Kadıköy"}. Örnek 2: "erenköy bostancı civarı" -> {"lokasyon": "Erenköy Bostancı"}. Sadece bulabildiklerini ekle."""},
                {"role": "user", "content": question}
            ],
            response_format={"type": "json_object"}, temperature=0.0, max_tokens=200
        )
        filters = json.loads(resp.choices[0].message.content)
        print(f"✅ Çıkarılan akıllı filtreler: {filters}")
        return filters
    except Exception as e:
        print(f"❌ Filtre çıkarma hatası: {e}")
        return {}

async def hybrid_search_listings(question: str) -> List[Dict]:
    """Supabase'de HIZLI hibrit arama yapar (v2 - Akıllı Lokasyon)."""
    if not supabase:
        print("❌ Supabase istemcisi mevcut değil. Arama yapılamıyor.")
        return []
        
    filters = await extract_filters_from_query(question)
    query_embedding = await get_embedding(question)
    if not query_embedding:
        return []
        
    try:
        print("⚡️ Supabase'de v2 hibrit arama yapılıyor...")
        rpc_params = {
            "query_embedding": query_embedding,
            "match_threshold": MATCH_THRESHOLD,
            "match_count": MATCH_COUNT,
            "p_max_fiyat": filters.get("max_fiyat"),
            "p_min_fiyat": filters.get("min_fiyat"),
            "p_oda_sayisi": filters.get("oda_sayisi"),
            "p_lokasyon": filters.get("lokasyon")
        }
        rpc_params = {k: v for k, v in rpc_params.items() if v is not None}

        # Supabase RPC çağrısı
        response = await asyncio.to_thread(
            supabase.rpc("search_listings_hybrid", rpc_params).execute
        )
        
        listings = response.data if hasattr(response, 'data') and response.data else []
        print(f"✅ v2 Hibrit arama tamamlandı. {len(listings)} ilan bulundu.")
        return listings
        
    except Exception as e:
        print(f"❌ Supabase RPC ('search_listings_hybrid') hatası: {e}")
        import traceback
        traceback.print_exc()
        return []

def format_listings_to_html(listings: List[Dict]) -> str:
    """İlan listesini şık bir HTML'e dönüştürür."""
    if not listings:
        return "<p>🔍 Üzgünüm, belirttiğiniz kriterlere uygun bir ilan bulamadım. Lütfen arama kriterlerinizi değiştirerek tekrar deneyin.</p>"

    def format_price(val):
        try:
            num = float(val)
            return f"{num:,.0f} ₺".replace(',', 'X').replace('.', ',').replace('X', '.')
        except (ValueError, TypeError):
            return str(val or 'N/A')

    html_parts = [
        f"<h3 style='color: #4dabf7;'>İşte sorgunuza en uygun {len(listings)} ilan:</h3>",
        "<ul style='list-style-type: none; padding: 0;'>"
    ]
    
    for ilan in listings:
        ilan_no = ilan.get('ilan_id', 'N/A')
        baslik = ilan.get('baslik', 'Başlık Yok')
        lokasyon = f"{ilan.get('ilce', '')}, {ilan.get('mahalle', '')}".strip(", ")
        fiyat = format_price(ilan.get('fiyat_numeric') or ilan.get('fiyat'))
        oda_sayisi = ilan.get('oda_sayisi', '')
        metrekare = f"{ilan.get('metrekare')} m²" if ilan.get('metrekare') else ''

        pdf_button = f"<a href='https://sibelgpt-backend.onrender.com/generate-property-pdf/{ilan_no}' target='_blank' style='display: inline-block; margin-top: 8px; padding: 6px 12px; background-color: #e53935; color: white; text-decoration: none; border-radius: 4px; font-size: 13px;'>📄 PDF Görüntüle</a>"
        
        html_parts.append(f"""
        <li style='background: rgba(40, 40, 40, 0.6); border-left: 4px solid #4dabf7; padding: 15px; margin-bottom: 12px; border-radius: 8px;'>
            <strong style='font-size: 16px; color: #ffffff;'>{baslik}</strong><br>
            <span style='font-size: 14px; color: #cccccc;'>📍 {lokasyon}  |  🏠 {oda_sayisi} ({metrekare})</span><br>
            <span style='font-size: 15px; font-weight: bold; color: #81c784;'>💰 {fiyat}</span>
            {pdf_button}
        </li>
        """)
        
    html_parts.append("</ul><p>Daha fazla detay veya farklı bir arama için lütfen belirtin.</p>")
    return "\n".join(html_parts)

async def check_if_property_listing_query(question: str) -> bool:
    """Sorunun ilan araması gerektirip gerektirmediğini tespit eder."""
    try:
        resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system","content": """Bu soruyu analiz et ve sadece "Evet" veya "Hayır" yanıtı ver. İLAN ARAMASI GEREKTİREN SORULAR (Evet): "Kadıköy'de satılık daire bul/ara/göster", "20 milyona kadar 3+1 daire arıyorum", "Beşiktaş'ta ev var mı?", "Maltepe'de villa göster/listele". İLAN ARAMASI GEREKTİRMEYEN SORULAR (Hayır): "Ev alırken nelere dikkat etmeliyim?", "Konut kredisi nasıl alınır?", "Tapu işlemleri nasıl yapılır?", "Emlak vergisi ne kadar?". Sadece "Evet" veya "Hayır" yanıtı ver."""},
                {"role": "user", "content": question}
            ],
            temperature=0.0, max_tokens=10
        )
        is_listing_query = "evet" in resp.choices[0].message.content.strip().lower()
        print(f"📊 İlan araması tespiti: {is_listing_query}")
        return is_listing_query
    except Exception as e:
        print(f"❌ İlan araması tespiti hatası: {e}")
        return False

# ==============================================================================
# ================= ANA SORGULAMA FONKSİYONU (NİHAİ HAL) ======================
# ==============================================================================

async def answer_question(question: str, mode: str = "real-estate", conversation_history: List = None) -> Dict[str, Any]:
    print(f"🚀 YENİ HIZLI SORGULAMA SİSTEMİ BAŞLADI - Soru: {question[:50]}..., Mod: {mode}")
    
    response_data = {"reply": "", "is_listing_response": False}

    # Sadece Gayrimenkul modunda özel arama mantığı çalışır
    if mode == 'real-estate':
        is_listing_query = await check_if_property_listing_query(question)
        
        if is_listing_query:
            print("🏠 İlan araması tespit edildi -> HIZLI HİBRİT ARAMA KULLANILIYOR!")
            response_data["is_listing_response"] = True # Avatar için sinyal
            
            try:
                listings = await hybrid_search_listings(question)
                response_data["reply"] = format_listings_to_html(listings)
                return response_data
            except Exception as e:
                print(f"❌ Hızlı arama yolu hatası: {e}")
                response_data["reply"] = "İlanları ararken bir sorunla karşılaştım. Lütfen daha sonra tekrar deneyin."
                return response_data

    # EĞER İLAN ARAMASI DEĞİLSE veya FARKLI BİR MOD İSE, GENEL GPT YOLU KULLANILIR.
    # Bu bölüm, projenizin orijinalindeki genel sohbet mantığıdır.
    # Şimdilik basit tutuyoruz, daha sonra eski kodunuzdaki karmaşık prompt'lar buraya eklenebilir.
    print(f"📚 Genel bilgi sorusu veya farklı mod ({mode}). Standart GPT kullanılacak.")
    try:
        messages = [
            {"role": "system", "content": f"Sen SibelGPT'sin. '{mode}' modunda bir uzmansın. Kullanıcının sorusuna cevap ver."},
        ]
        if conversation_history:
            messages.extend(conversation_history[-5:]) # Son 5 mesajı al
        
        messages.append({"role": "user", "content": question})

        resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=1500
        )
        response_data["reply"] = resp.choices[0].message.content.strip()

    except Exception as e:
        print(f"❌ Genel GPT yanıt hatası: {e}")
        response_data["reply"] = "Üzgünüm, bu soruya cevap verirken bir sorun oluştu."

    return response_data
