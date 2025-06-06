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
SB_ANON_KEY = os.getenv("SUPABASE_KEY") # Güvenli olduğunu teyit ettik.

if not all([OAI_KEY, SB_URL, SB_ANON_KEY]):
    raise RuntimeError("Eksik API anahtarı veya Supabase bağlantı bilgisi (URL ve ANON_KEY).")

openai_client = AsyncOpenAI(api_key=OAI_KEY)
supabase: Optional[Client] = None
if SUPABASE_AVAILABLE:
    supabase = create_client(SB_URL, SB_ANON_KEY)

# ---- Ayarlar ----
EMBEDDING_MODEL = "text-embedding-3-small"
MATCH_THRESHOLD = 0.4  # Eşiği biraz artırarak daha alakalı sonuçlar hedefliyoruz.
MATCH_COUNT = 25

# ==============================================================================
# ==================== YENİ VE HIZLI ARAMA MİMARİSİ =============================
# ==============================================================================

async def get_embedding(text: str) -> Optional[List[float]]:
    """Metin için embedding oluşturur."""
    try:
        resp = await openai_client.embeddings.create(model=EMBEDDING_MODEL, input=[text.strip()])
        return resp.data[0].embedding
    except Exception as e:
        print(f"❌ Embedding hatası: {e}")
        return None

async def extract_filters_from_query(question: str) -> Dict:
    """Sorgudan SADECE yapısal filtreleri çıkarır (Hızlı GPT-4o-mini çağrısı)."""
    print(f"🔍 Hızlı filtre çıkarma işlemi başlatıldı: {question}")
    try:
        resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """Sen bir emlak arama asistanısın. Kullanıcının sorgusundan SADECE şu gayrimenkul filtrelerini JSON formatında çıkar: "min_fiyat": Sayısal minimum fiyat, "max_fiyat": Sayısal maksimum fiyat, "oda_sayisi": String (ör: "2+1"), "ilce": String, "mahalle": String. Sadece bulabildiğin filtreleri ekle. Örnek: "kadıköyde 5 milyona kadar 2+1 daire" -> {"max_fiyat": 5000000, "oda_sayisi": "2+1", "ilce": "Kadıköy"}"""},
                {"role": "user", "content": question}
            ],
            response_format={"type": "json_object"}, temperature=0.0, max_tokens=200
        )
        filters = json.loads(resp.choices[0].message.content)
        print(f"✅ Çıkarılan filtreler: {filters}")
        return filters
    except Exception as e:
        print(f"❌ Filtre çıkarma hatası: {e}")
        return {}

async def hybrid_search_listings(question: str) -> List[Dict]:
    """Supabase'de HIZLI ve GÜÇLÜ hibrit arama yapar (Vektör + Filtre)."""
    if not supabase:
        print("❌ Supabase istemcisi mevcut değil. Arama yapılamıyor.")
        return []
        
    filters = await extract_filters_from_query(question)
    query_embedding = await get_embedding(question)
    if not query_embedding:
        return []
        
    try:
        print("⚡️ Supabase'de hibrit arama yapılıyor...")
        rpc_params = {
            "query_embedding": query_embedding,
            "match_threshold": MATCH_THRESHOLD,
            "match_count": MATCH_COUNT,
            "p_max_fiyat": filters.get("max_fiyat"),
            "p_min_fiyat": filters.get("min_fiyat"),
            "p_oda_sayisi": filters.get("oda_sayisi"),
            "p_ilce": filters.get("ilce"),
            "p_mahalle": filters.get("mahalle")
        }
        rpc_params = {k: v for k, v in rpc_params.items() if v is not None}

        # Supabase RPC çağrısı artık async await ile çalışmalı
        response = await asyncio.to_thread(
            supabase.rpc("search_listings_hybrid", rpc_params).execute
        )
        
        listings = response.data if hasattr(response, 'data') and response.data else []
        print(f"✅ Hibrit arama tamamlandı. {len(listings)} ilan bulundu.")
        return listings
        
    except Exception as e:
        print(f"❌ Supabase RPC ('search_listings_hybrid') hatası: {e}")
        import traceback
        print(traceback.format_exc())
        return []

def format_listings_to_html(listings: List[Dict]) -> str:
    """İlan listesini SibelGPT için şık bir HTML formatına dönüştürür."""
    if not listings:
        return "🔍 Üzgünüm, belirttiğiniz kriterlere uygun bir ilan bulamadım. Farklı kriterlerle tekrar deneyebilirsiniz."

    # Fiyat formatlama yardımcısı
    def format_price(price_val):
        try:
            # Gelen değer zaten sayısal (fiyat_numeric) veya sayısal-benzeri bir string olabilir
            price_num = float(price_val)
            # Türkçe formatlama için (1.234.567 TL)
            return f"{price_num:,.0f} ₺".replace(',', 'X').replace('.', ',').replace('X', '.')
        except (ValueError, TypeError):
            # Eğer sayıya çevrilemezse, orijinal string'i döndür
            return str(price_val or 'N/A')

    html_parts = [
        f"<h3 style='color: #4dabf7;'>İşte sizin için bulduğum {len(listings)} ilan:</h3>",
        "<ul style='list-style-type: none; padding: 0;'>"
    ]
    
    for ilan in listings:
        ilan_no = ilan.get('ilan_id', 'N/A')
        baslik = ilan.get('baslik', 'Başlık Yok')
        lokasyon = f"{ilan.get('ilce', '')}, {ilan.get('mahalle', '')}".strip(", ")
        fiyat = format_price(ilan.get('fiyat_numeric') or ilan.get('fiyat'))
        oda_sayisi = ilan.get('oda_sayisi', '')
        metrekare = f"{ilan.get('metrekare', '')} m²" if ilan.get('metrekare') else ''

        pdf_button = (
            f"<a href='https://sibelgpt-backend.onrender.com/generate-property-pdf/{ilan_no}' target='_blank' "
            f"style='display: inline-block; margin-top: 8px; padding: 6px 12px; background-color: #d32f2f; color: white; text-decoration: none; border-radius: 4px; font-size: 13px;'>"
            f"📄 PDF Görüntüle</a>"
        )
        
        html_parts.append(f"""
        <li style='background: rgba(30, 30, 30, 0.5); border-left: 4px solid #4dabf7; padding: 15px; margin-bottom: 12px; border-radius: 8px;'>
            <strong style='font-size: 16px; color: #ffffff;'>{baslik}</strong><br>
            <span style='font-size: 14px; color: #cccccc;'>
                📍 {lokasyon}  |  🏠 {oda_sayisi} ({metrekare})
            </span><br>
            <span style='font-size: 15px; font-weight: bold; color: #4CAF50;'>
                💰 {fiyat}
            </span>
            {pdf_button}
        </li>
        """)
        
    html_parts.append("</ul><p>Daha fazla detay veya farklı bir arama için lütfen belirtin.</p>")
    return "\n".join(html_parts)

async def check_if_property_listing_query(question: str) -> bool:
    """Sorunun ilan araması gerektirip gerektirmediğini tespit eder."""
    # Bu fonksiyon eski koddan kopyalandı ve aynen kullanılıyor.
    try:
        resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system","content": """Bu soruyu analiz et ve sadece "Evet" veya "Hayır" yanıtı ver. İLAN ARAMASI GEREKTİREN SORULAR (Evet): "Kadıköy'de satılık daire bul/ara/göster", "20 milyona kadar 3+1 daire arıyorum", "Beşiktaş'ta ev var mı?", "Maltepe'de villa göster/listele". İLAN ARAMASI GEREKTİRMEYEN SORULAR (Hayır): "Ev alırken nelere dikkat etmeliyim?", "Konut kredisi nasıl alınır?", "Tapu işlemleri nasıl yapılır?", "Emlak vergisi ne kadar?". Sadece "Evet" veya "Hayır" yanıtı ver."""},
                {"role": "user", "content": question}
            ],
            temperature=0.1, max_tokens=10
        )
        answer = resp.choices[0].message.content.strip().lower()
        is_listing_query = "evet" in answer
        print(f"📊 İlan araması tespiti: {answer} → {is_listing_query}")
        return is_listing_query
    except Exception as e:
        print(f"❌ İlan araması tespiti hatası: {e}")
        return False

# ==============================================================================
# ================= ANA SORGULAMA FONKSİYONU (YENİLENDİ) =======================
# ==============================================================================

async def answer_question(question: str, mode: str = "real-estate", conversation_history: List = None) -> Dict[str, Any]:
    print(f"🚀 YENİ HIZLI SORGULAMA SİSTEMİ BAŞLADI - Soru: {question[:50]}..., Mod: {mode}")
    
    response_data = {"reply": "", "is_listing_response": False}

    # Henüz bir modül yönlendirme veya genel bilgi sorusu mantığı eklemedik.
    # Şimdilik sadece gayrimenkul ilan aramasına odaklanıyoruz.
    
    if mode == 'real-estate':
        is_listing_query = await check_if_property_listing_query(question)
        
        if is_listing_query:
            print("🏠 İlan araması tespit edildi -> YENİ HIZLI HİBRİT ARAMA KULLANILIYOR!")
            response_data["is_listing_response"] = True # Avatar için sinyal gönder
            
            try:
                listings = await hybrid_search_listings(question)
                response_data["reply"] = format_listings_to_html(listings)
                return response_data
            except Exception as e:
                print(f"❌ Hızlı arama yolu hatası: {e}")
                response_data["reply"] = "İlanları ararken bir sorunla karşılaştım. Lütfen daha sonra tekrar deneyin."
                return response_data

    # EĞER İLAN ARAMASI DEĞİLSE, BURADA ESKİ GPT YOLU DEVREYE GİRECEK
    # Bu kısmı daha sonra ekleyebiliriz. Şimdilik basit bir mesaj döndürelim.
    print("📚 Genel bilgi sorusu veya farklı mod. (Henüz tam entegre edilmedi).")
    # Bu geçici bir yanıttır. İleride buraya tam GPT mantığını ekleyeceğiz.
    # Bu örnekte, test için basitçe soruyu geri döndürelim ve bir not ekleyelim.
    response_data["reply"] = f"Anladım, '{question}' hakkında genel bir soru sordunuz. Bu özellik şu an geliştirme aşamasındadır."

    return response_data
