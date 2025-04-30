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
    if not listings:
        return "🔍 Uygun ilan bulunamadı."

    import locale
    try:
        locale.setlocale(locale.LC_ALL, 'tr_TR.UTF-8')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, 'tr_TR')
        except locale.Error:
            pass

    formatted_parts = []
    for i, l in enumerate(listings, start=1):
        baslik = l.get("baslik", "(başlık yok)")
        lokasyon = l.get("lokasyon", "?")
        fiyat_raw = l.get("fiyat")
        fiyat = "?"

        try:
            fiyat_num = float(str(fiyat_raw).replace('.', '').replace(',', '.'))
            try:
                fiyat = locale.currency(fiyat_num, symbol='₺', grouping=True)
                if fiyat.endswith('.00') or fiyat.endswith(',00'):
                    fiyat = fiyat[:-3] + ' ₺'
                else:
                    fiyat = fiyat.replace('₺', '').strip() + ' ₺'
            except:
                fiyat = f"{fiyat_num:,.0f} ₺".replace(',', '#').replace('.', ',').replace('#', '.')
        except:
            fiyat = str(fiyat_raw) if fiyat_raw else "?"

        ilan_html = (
            f"<strong>{i}. {baslik}</strong><br>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;• Lokasyon: {lokasyon}<br>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;• Fiyat: {fiyat}<br><br>"
        )
        formatted_parts.append(ilan_html)

    final_output = "".join(formatted_parts)
    final_output += "<br>📞 Bu ilanlar hakkında daha fazla bilgi almak isterseniz: 532 687 84 64"
    final_output += "<br><br><span style='color:red;'>[Sibel Test - HTML Render]</span>"
    
    return final_output
    
# ── Ana Q&A işlevi ──────────────────────────────────────────────────────────
async def answer_question(question: str) -> str:
    print("↪ Soru:", question)

    query_emb = await get_embedding(question)
    listings  = await search_listings_in_supabase(query_emb)
    context = format_context_for_sibelgpt(listings)

  messages = [
    {
        "role": "system",
        "content": f"{SYSTEM_PROMPT}<br><br>{context}"
    },
    {
        "role": "user",
        "content": question
    }
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
