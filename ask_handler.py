# ask_handler.py  –  30 Nisan 2025 güncel sürüm
# ────────────────────────────────────────────────────────────────────────────
# • Supabase tablo kolon adları:  ilan_no, baslik, fiyat, ozellikler,
#                                 lokasyon, detay_url, embedding  (vector)
# • OpenAI-Python v1.x (>=1.0) ile çalışır.
# • match_listings RPC çıktısı aynı adları döndürmelidir!

import os
import asyncio
from openai import AsyncOpenAI                       # OpenAI-Python ≥1.0
from typing import List, Dict, Optional

# Supabase-py async client (v2.x)
try:
    from supabase import AsyncClient, create_client
except ImportError:
    raise RuntimeError("supabase-py yüklü değil – `pip install supabase`")

# ── Ortam değişkenleri ──────────────────────────────────────────────────────
OAI_KEY = os.getenv("OPENAI_API_KEY")
SB_URL  = os.getenv("SUPABASE_URL")
SB_KEY  = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY")

if not all([OAI_KEY, SB_URL, SB_KEY]):
    raise RuntimeError(".env dosyasında OPENAI / SUPABASE anahtarları eksik")

openai_client = AsyncOpenAI(api_key=OAI_KEY)
supabase      = create_client(SB_URL, SB_KEY)

# ── Ayarlar ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL  = "text-embedding-3-small"
MATCH_THRESHOLD  = 0.45      # eşik – 0 ile 1 arası (daha düşük = daha geniş)
MATCH_COUNT      = 20        # dönecek ilan sayısı

SYSTEM_PROMPT = (
    "Sen SibelGPT'sin: Sibel Kazan Midilli tarafından geliştirilen, "
    "Türkiye emlak piyasası (özellikle Remax Sonuç portföyü), numeroloji ve "
    "finans konularında uzman, Türkçe yanıt veren yardımsever bir yapay zeka "
    "asistanısın.\n\n"
    "Kullanıcı emlak sorusu sorduğunda, sana sağlanan 'İLGİLİ İLANLAR' "
    "bölümündeki verileri kullanarak yanıt ver. O veriler yoksa dürüstçe "
    "söyle ve genel tavsiye ver.\n\n"
    "Cevapları kısa, net ve samimi tut; ilan başlığı, fiyat, lokasyon ve linki "
    "madde madde listeleyebilirsin."
)

# ── Embedding oluşturma ─────────────────────────────────────────────────────
async def get_embedding(text: str) -> Optional[List[float]]:
    """ Verilen metni OpenAI’dan gömme (embedding) vektörüne çevirir. """
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
        print("❌ OpenAI embedding hatası:", exc)
        return None

# ── Supabase’te benzer ilanları aran ────────────────────────────────────────
async def search_listings_in_supabase(
    query_embedding: List[float]
) -> List[Dict]:
    """match_listings RPC’sini çağırıp benzer ilanları döndürür."""
    if query_embedding is None:
        return []
    try:
        resp =supabase.rpc(
            "match_listings",
            {
                "query_embedding": query_embedding,
                "match_threshold": MATCH_THRESHOLD,
                "match_count":     MATCH_COUNT
            }
        ).execute()
        # supabase-py 2.x: resp is PostgrestResponse, kayıtlar resp.data’de
        return resp.data if hasattr(resp, "data") else resp
    except Exception as exc:
        print("❌ Supabase RPC hatası:", exc)
        return []

# ── İlan listesini prompt bağlamına çevir ───────────────────────────────────
import locale
from typing import List, Dict

def format_context_for_sibelgpt(listings: List[Dict]) -> str:
    """
    İlanları, yeni satır karakterleri (\n) ve Markdown kalın formatını
    destekleyen ortamlar için (SibelGPT gibi) formatlar.
    """
    if not listings:
        return "Uygun ilan bulunamadı." # Başlangıç mesajını kaldırdık, sadece ilanlar dönecek

    # Türkçe locale ayarlarını kullanarak para birimini formatlamak için (opsiyonel)
    try:
        locale.setlocale(locale.LC_ALL, 'tr_TR.UTF-8')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, 'tr_TR')
        except locale.Error:
            # Locale ayarlanamazsa uyarı vermeden devam edilebilir
            # print("Uyarı: Türkçe locale ayarlanamadı. Fiyat formatlaması basit olabilir.")
            pass # Hata durumunda sessizce devam et

    formatted_listing_parts = [] # Her bir ilanın formatlanmış metnini tutacak liste
    for i, l in enumerate(listings, start=1):
        baslik = l.get("baslik", "(başlık yok)")
        lokasyon = l.get("lokasyon", "?")
        fiyat_raw = l.get("fiyat")

        # Fiyatı formatla (sayısal ise) - Önceki mantıkla aynı
        fiyat_formatted = "?" # Varsayılan
        if fiyat_raw is not None:
            try:
                fiyat_num = float(str(fiyat_raw).replace('.', '').replace(',', '.'))
                try:
                    # Locale kullanarak formatla
                    fiyat_formatted = locale.currency(fiyat_num, symbol='₺', grouping=True)
                     # Ondalık kısmı .00 veya ,00 ise kaldır ve boşluk ekle
                    if fiyat_formatted.endswith('.00') or fiyat_formatted.endswith(',00'):
                        fiyat_formatted = fiyat_formatted[:-3].strip() + ' ₺'
                    else:
                        # Para birimi sembolünden önce boşluk olduğundan emin ol
                        fiyat_formatted = fiyat_formatted.replace('₺', '').strip() + ' ₺'

                except (NameError, locale.Error): # locale başarısız olursa veya ayarlanamadıysa
                    # Basit formatlama (binlik ayraçları ile)
                    fiyat_formatted = f"{fiyat_num:,.0f} ₺".replace(',', '#').replace('.', ',').replace('#', '.') # Türkçe formatına uygun hale getir
            except (ValueError, TypeError):
                fiyat_formatted = str(fiyat_raw) # Sayısal değilse olduğu gibi bırak
        else:
             fiyat_formatted = "?" # Fiyat yoksa


        # İstenen format: Sıra no, Kalın Başlık, yeni satır, girintili detaylar
        # Yeni satır için `\n`, girinti için 4 boşluk kullanıldı
        ilan_metni = (
            f"{i}. <strong>{baslik}</strong><br>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;• Lokasyon: {lokasyon}<br>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;• Fiyat: {fiyat_formatted}<br><br>"
        )
        formatted_listing_parts.append(ilan_metni)

    # Tüm ilanları aralarına ikişer yeni satır koyarak birleştir
    listings_str = "\n\n".join(formatted_listing_parts)

    # En sona iletişim bilgisini iki yeni satırla ekle
    final_output = f"{listings_str}<br><br>📞 Detaylı bilgi ve randevu için: 532 687 84 64"

    return final_output

# --- Örnek Kullanım (İkinci resimdeki verilerle) ---
example_listings_from_image2 = [
    {"baslik": "GÖZTEPE 60. YIL PARKI VE DENİZ MANZARALI GENÇ DAİRE", "fiyat": "15.750.000", "lokasyon": "İstanbul Anadolu / Kadıköy / Göztepe Mah."},
    {"baslik": "GÖZTEPE'DE YEŞİLLİKLER İÇİNDE SATILIK BOŞ 3+1 DAİRE", "fiyat": "11.000.000", "lokasyon": "İstanbul Anadolu / Kadıköy / Göztepe Mah."},
    {"baslik": "GÖZTEPE ÖMERPAŞA SOKAK OYUNCAK MÜZESİ YANI 100M2 2+1 SATILIK DAİRE", "fiyat": "10.750.000", "lokasyon": "İstanbul Anadolu / Kadıköy / Göztepe Mah."},
    {"baslik": "GÖZTEPE TÜTÜNCÜ MEHMET EFENDİ CADDESİNE 1. PARALELDE 5+1 KATTA TEK", "fiyat": "33.500.000", "lokasyon": "İstanbul Anadolu / Kadıköy / Göztepe Mah."}, # Lokasyon Göztepe Mah. olarak düzeltildi
    {"baslik": "KADIKÖY GÖZTEPE'DE SATILIK 3+1 DAİRE 128 M2 (ARSA PAYI 53.13 M2)", "fiyat": "9.800.000", "lokasyon": "İstanbul Anadolu / Kadıköy / Göztepe Mah."}
]

# Fonksiyonu çağırıp çıktıyı test etme
# formatted_output = format_context_for_sibelgpt(example_listings_from_image2)
# print(formatted_output)

# ── Ana Q&A işlevi ──────────────────────────────────────────────────────────
async def answer_question(question: str) -> str:
    print("↪ Soru:", question)

    query_emb = await get_embedding(question)
    listings  = await search_listings_in_supabase(query_emb)
    context = format_context_for_sibelgpt(listings)

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT + "\n\n" + context},
        {"role": "user",   "content": question}
    ]

    try:
        resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=1024
        )
        answer = resp.choices[0].message.content.strip()
        print("✓ Yanıt üretildi.")
        return answer
    except Exception as exc:
        print("❌ ChatCompletion hatası:", exc)
        return "Üzgünüm, şu anda sorunuza yanıt verirken bir hata oluştu."

# ── Demo (dosya doğrudan çalıştırılırsa) ────────────────────────────────────
if __name__ == "__main__":
    q = input("Soru: ")
    loop = asyncio.get_event_loop()
    print(loop.run_until_complete(answer_question(q)))
