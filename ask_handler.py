import os
import asyncio
import locale
import re
from typing import List, Dict, Optional
from openai import AsyncOpenAI

try:
    from supabase import create_client
    from supabase.client import Client
except ImportError:
    raise RuntimeError("supabase-py yüklü değil – `pip install supabase`")

# ── Ortam Değişkenleri ─────────────────────────────────────
OAI_KEY = os.getenv("OPENAI_API_KEY")
SB_URL  = os.getenv("SUPABASE_URL")
SB_KEY  = os.getenv("SUPABASE_KEY") or os.getenv("SUPABASE_ANON_KEY")

if not all([OAI_KEY, SB_URL, SB_KEY]):
    raise RuntimeError("Eksik API anahtarı veya Supabase bağlantı bilgisi.")

openai_client = AsyncOpenAI(api_key=OAI_KEY)
supabase      = create_client(SB_URL, SB_KEY)

# ── Ayarlar ────────────────────────────────────────────────
EMBEDDING_MODEL = "text-embedding-3-small"
MATCH_THRESHOLD = 0.65
MATCH_COUNT     = 20

SYSTEM_PROMPT = (
    "Sen SibelGPT'sin: Sibel Kazan Midilli tarafından geliştirilen, "
    "Türkiye emlak piyasası (özellikle Remax Sonuç portföyü), numeroloji ve "
    "finans konularında uzman, Türkçe yanıt veren yardımsever bir yapay zeka asistansın.\n\n"
    
    "Kullanıcı sana emlak sorusu sorduğunda, Supabase'den getirilen 'İLGİLİ İLANLAR' "
    "verilerini kullanarak en alakalı ilanları seçip listele. Eğer yeterli veri yoksa "
    "dürüstçe belirt ve kullanıcıya sorular sorarak ihtiyacını netleştir (örneğin: "
    "Hangi mahallede bakıyorsunuz? Kaç odalı? Bütçeniz nedir?).\n\n"
    
    "Cevaplarını kısa, net ve samimi tut; her ilanda başlık, ilan numarası, fiyat, lokasyon ve özellik bilgisi olsun.\n\n"
    
    "Yanıtlarını HTML formatında oluştur. <ul> ve <li> kullan. Satır atlamak için <br>, "
    "kalın yazı için <strong> kullan. Markdown işaretleri (*, -) kullanma.\n\n"
)

# ── Embedding Fonksiyonu ───────────────────────────────────
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
        print("❌ Embedding hatası:", exc)
        return None

# ── Supabase Sorgusu ───────────────────────────────────────
async def search_listings_in_supabase(query_embedding: List[float]) -> List[Dict]:
    if query_embedding is None:
        return []
    try:
        resp = supabase.rpc(
            "match_listings",
            {
                "query_embedding": query_embedding,
                "match_threshold": MATCH_THRESHOLD,
                "match_count":     MATCH_COUNT
            }
        ).execute()
        return resp.data if hasattr(resp, "data") else resp
    except Exception as exc:
        print("❌ Supabase RPC hatası:", exc)
        return []

# ── Formatlama Fonksiyonu ─────────────────────────────────
def format_context_for_sibelgpt(listings: List[Dict]) -> str:
    if not listings:
        return "🔍 Uygun ilan bulunamadı."

    try:
        locale.setlocale(locale.LC_ALL, 'tr_TR.UTF-8')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, 'tr_TR')
        except locale.Error:
            pass

    formatted_parts = []
    for i, l in enumerate(listings, start=1):
        ilan_no    = l.get("ilan_no", "(numara yok)")
        baslik     = re.sub(r"^\d+\.\s*", "", l.get("baslik", "(başlık yok)"))  # numara temizle
        lokasyon   = l.get("lokasyon", "?")
        fiyat_raw  = l.get("fiyat")
        ozellikler = l.get("ozellikler", "(özellik yok)")
        fiyat      = "?"

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
            f"<li><strong>{i}. {baslik}</strong><br>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;• İlan No: {ilan_no}<br>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;• Lokasyon: {lokasyon}<br>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;• Fiyat: {fiyat}<br>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;• Özellikler: {ozellikler}</li><br>"
        )
        formatted_parts.append(ilan_html)

    final_output = "<ul>" + "\n".join(formatted_parts) + "</ul>"
    final_output += "<br>📞 Bu ilanlar hakkında daha fazla bilgi almak isterseniz: 532 687 84 64"
    return final_output

# ── Ana Fonksiyon ─────────────────────────────────────────
async def answer_question(question: str) -> str:
    print("↪ Soru:", question)

    query_emb = await get_embedding(question)
    listings  = await search_listings_in_supabase(query_emb)
    context   = format_context_for_sibelgpt(listings)

    messages = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}<br><br>{context}"},
        {"role": "user",   "content": question}
    ]

    try:
        resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=1024
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        print("❌ Chat yanıt hatası:", exc)
        return "Üzgünüm, şu anda bir hata oluştu."

# ── Terminalden Test ──────────────────────────────────────
if __name__ == "__main__":
    q = input("Soru: ")
    loop = asyncio.get_event_loop()
    print(loop.run_until_complete(answer_question(q)))
