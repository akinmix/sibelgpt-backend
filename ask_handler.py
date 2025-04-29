# ask_handler.py (Supabase RAG Entegrasyonlu ve Import Düzeltmeli Sürüm)
"""
ask_handler.py
--------------
Kullanıcı sorularını yanıtlar. Emlak ile ilgili sorular için Supabase'den
güncel ilan verilerini çekerek Retrieval-Augmented Generation (RAG) yapar.

Ön-koşullar
-----------
• requirements.txt → openai>=1.0.0, supabase>=2.0.0 # Güncel paket adı: supabase
• Render ortam değişkenlerinde OPENAI_API_KEY, SUPABASE_URL, SUPABASE_KEY ayarlı.
• Supabase veritabanında 'match_listings' adında bir RPC fonksiyonu tanımlı.
"""

import os
from openai import AsyncOpenAI

# ---- Düzeltilmiş Supabase Import'ları ----
# from supabase_py_async import AsyncClient  # ESKİ YANLIŞ SATIR - SİLİNDİ
from supabase.lib.client_async import AsyncClient # YENİ DOĞRU IMPORT - AsyncClient için
# -----------------------------------------


# ── OpenAI istemcisi (Hem Chat hem Embedding için) ─────────────────────────────
openai_client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

# --- Embedding Modeli Ayarı ---
EMBEDDING_MODEL = "text-embedding-3-small"

# --- Supabase Arama Ayarları ---
MATCH_THRESHOLD = 0.78
MATCH_COUNT = 5

# --- Sistem Talimatı (Kişilik ve RAG Yönlendirmesi) ---
SYSTEM_PROMPT = (
    "Sen SibelGPT'sin: Sibel Kazan Midilli tarafından geliştirilen, "
    "Türkiye emlak piyasası (özellikle Remax Sonuç portföyü), numeroloji ve finans konularında uzman, "
    "Türkçe yanıt veren yardımsever bir yapay zeka asistanısın.\n"
    "Kullanıcının emlak ile ilgili sorularını yanıtlarken SANA SAĞLANAN GÜNCEL İLAN BİLGİLERİNİ kullan. "
    "Eğer sağlanan bilgiler soruyu yanıtlamak için yeterliyse, sadece bu bilgileri kullanarak cevap ver. "
    "Sağlanan bilgi yoksa veya yetersizse, bunu belirt.\n"
    "Cevaplarında ilanların kısa özetlerini ve URL'lerini verebilirsin. Fiyat ve konum bilgilerini de ekle.\n"
    "Yanıtlarını kısa, net, profesyonel ve samimi tut."
)


async def get_embedding(text: str) -> list[float] | None:
    """Verilen metnin OpenAI embedding'ini alır."""
    try:
        response = await openai_client.embeddings.create(
            input=[text.strip()],
            model=EMBEDDING_MODEL
        )
        if response.data and len(response.data) > 0 and response.data[0].embedding:
             return response.data[0].embedding
        else:
             print("❌ Embedding alınamadı, OpenAI yanıtı beklenmedik formatta.")
             return None
    except Exception as e:
        print(f"❌ OpenAI Embedding hatası: {e}")
        return None

# AsyncClient tipi doğru yerden import edildiği için burası çalışmalı.
async def search_listings_in_supabase(query_embedding: list[float], db_client: AsyncClient) -> list[dict]:
    """Verilen embedding ile Supabase'de benzer ilanları arar."""
    if not query_embedding:
        return []
    try:
        response = await db_client.rpc(
            'match_listings',
            {
                'query_embedding': query_embedding,
                'match_threshold': MATCH_THRESHOLD,
                'match_count': MATCH_COUNT
            }
        ).execute()

        if response.data:
            return response.data
        else:
            return []
    except Exception as e:
        print(f"❌ Supabase RPC hatası (match_listings): {e}")
        return []


def format_context(listings: list[dict]) -> str:
    """Supabase'den gelen ilan listesini okunabilir bir bağlam metnine dönüştürür."""
    if not listings:
        return "Veritabanında bu soruyla ilgili güncel ilan bulunamadı."

    context = "İlgili olabilecek güncel ilanlar:\n\n"
    for listing in listings:
        summary = listing.get('summary', 'Özet bilgisi yok')
        price = listing.get('price', 'Fiyat bilgisi yok')
        location = listing.get('location', 'Konum bilgisi yok')
        url = listing.get('url', 'URL bilgisi yok')

        context += f"- Özet: {summary}\n"
        context += f"  Fiyat: {price}\n"
        context += f"  Konum: {location}\n"
        context += f"  URL: {url}\n\n"

    return context.strip()


# Ana fonksiyon – backend'de main.py çağırıyor
# AsyncClient tipi doğru yerden import edildiği için burası da çalışmalı.
async def answer_question(question: str, db_client: AsyncClient) -> str:
    """
    Kullanıcıdan gelen soruyu OpenAI ChatCompletion ile yanıtla.
    Supabase'den ilgili ilanları çekip RAG uygular.
    """
    print(f"Gelen Soru: {question}")

    print("Soru için embedding oluşturuluyor...")
    query_embedding = await get_embedding(question)

    print("Supabase'de benzer ilanlar aranıyor...")
    listings = await search_listings_in_supabase(query_embedding, db_client)

    print("Cevap için bağlam oluşturuluyor...")
    context = format_context(listings)

    print("OpenAI ile nihai cevap üretiliyor...")
    final_system_prompt = SYSTEM_PROMPT
    if listings:
         final_system_prompt += "\n\nKullanıcının sorusuna cevap verirken KULLANILACAK GÜNCEL İLAN BİLGİLERİ:\n---\n" + context + "\n---"

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": final_system_prompt},
                {"role": "user", "content": question},
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        final_answer = response.choices[0].message.content.strip()
        print(f"Üretilen Cevap: {final_answer}")
        return final_answer

    except Exception as e:
        print(f"❌ OpenAI ChatCompletion hatası: {e}")
        return "❌ Cevap üretilirken bir hata oluştu. Lütfen tekrar deneyin."
