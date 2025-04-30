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
supabase      = create_client(SB_URL, SB_KEY, auth_token=None, timeout=30_000)

# ── Ayarlar ─────────────────────────────────────────────────────────────────
EMBEDDING_MODEL  = "text-embedding-3-small"
MATCH_THRESHOLD  = 0.60      # eşik – 0 ile 1 arası (daha düşük = daha geniş)
MATCH_COUNT      = 10        # dönecek ilan sayısı

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
        resp = await supabase.rpc(
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
def format_context(listings: List[Dict]) -> str:
    if not listings:
        return "İLGİLİ İLAN BULUNAMADI."
    lines = ["İLGİLİ İLANLAR:"]
    for l in listings:
        lines.append(
            f"- {l.get('baslik','(başlık yok)')} • {l.get('fiyat','?')} • "
            f"{l.get('lokasyon','?')}\n  {l.get('detay_url','')}"
        )
    return "\n".join(lines)

# ── Ana Q&A işlevi ──────────────────────────────────────────────────────────
async def answer_question(question: str) -> str:
    print("↪ Soru:", question)

    query_emb = await get_embedding(question)
    listings  = await search_listings_in_supabase(query_emb)
    context   = format_context(listings)

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
