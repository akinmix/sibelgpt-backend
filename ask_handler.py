# ask_handler.py (RPC Çağrısı Düzeltilmiş Sürüm)
"""
ask_handler.py
--------------
Kullanıcı sorularını yanıtlar. Emlak ile ilgili sorular için Supabase'den
güncel ilan verilerini çekerek Retrieval-Augmented Generation (RAG) yapar.

Ön-koşullar
-----------
• requirements.txt → openai>=1.0.0, supabase>=2.0.0
• Render ortam değişkenlerinde OPENAI_API_KEY, SUPABASE_URL, SUPABASE_KEY ayarlı.
• Supabase veritabanında 'match_listings' adında bir RPC fonksiyonu tanımlı.
"""

import os
from openai import AsyncOpenAI

# ---- Supabase Import'u Yok ----

# ── OpenAI istemcisi ───────────────────────────────────────────────────────────
openai_client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

# --- Ayarlar ---
EMBEDDING_MODEL = "text-embedding-3-small"
MATCH_THRESHOLD = 0.78
MATCH_COUNT = 5
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
        response = await openai_client.embeddings.create(input=[text.strip()], model=EMBEDDING_MODEL)
        if response.data and response.data[0].embedding:
             return response.data[0].embedding
        else:
             print("❌ Embedding alınamadı.")
             return None
    except Exception as e:
        print(f"❌ OpenAI Embedding hatası: {e}")
        return None

async def search_listings_in_supabase(query_embedding: list[float], db_client: "AsyncClient") -> list[dict]:
    """Verilen embedding ile Supabase'de benzer ilanları arar."""
    if not query_embedding:
        return []
    try:
        print(f"Supabase RPC 'match_listings' çağrılıyor...") # Debug log
        # ---- DÜZELTME BURADA ----
        response = await db_client.rpc(
            'match_listings',
            {
                'query_embedding': query_embedding,
                'match_threshold': MATCH_THRESHOLD,
                'match_count': MATCH_COUNT
            }
        ) # <-- .execute() silindi!
        # -------------------------
        print(f"Supabase RPC yanıtı alındı. Data: {response.data if hasattr(response, 'data') else 'Yanıt formatı beklenmedik'}") # Debug log

        # Yanıtın yapısını kontrol edelim (Supabase kütüphane versiyonuna göre değişebilir)
        # Genellikle sonuçlar response objesinin 'data' attribute'unda olur.
        if hasattr(response, 'data') and response.data:
             return response.data
        else:
             print("Supabase'den beklenen formatta veri alınamadı veya veri boş.")
             return []
    except Exception as e:
        print(f"❌ Supabase RPC hatası (match_listings çağrılırken): {e}")
        # Hatanın detayını görmek için traceback'i de loglayabiliriz (isteğe bağlı)
        # import traceback
        # print(traceback.format_exc())
        return []


def format_context(listings: list[dict]) -> str:
    """Supabase'den gelen ilan listesini okunabilir bir bağlam metnine dönüştürür."""
    if not listings:
        return "Veritabanında bu soruyla ilgili güncel ilan bulunamadı."
    context = "İlgili olabilecek güncel ilanlar:\n\n"
    for listing in listings:
        summary = listing.get('ozet', 'Özet bilgisi yok') # SQL fonksiyonu 'ozet' döndürüyor
        price = listing.get('fiyat', 'Fiyat bilgisi yok') # SQL fonksiyonu 'fiyat' döndürüyor
        location = listing.get('location', 'Konum bilgisi yok')
        url = listing.get('url', 'URL bilgisi yok')
        context += f"- Özet: {summary}\n  Fiyat: {price}\n  Konum: {location}\n  URL: {url}\n\n"
    return context.strip()


async def answer_question(question: str, db_client: "AsyncClient") -> str:
    """
    Kullanıcıdan gelen soruyu OpenAI ChatCompletion ile yanıtla. RAG uygular.
    """
    print(f"Gelen Soru: {question}")
    query_embedding = await get_embedding(question)

    # search_listings_in_supabase artık düzeltilmiş olmalı
    listings = await search_listings_in_supabase(query_embedding, db_client)

    context = format_context(listings)
    final_system_prompt = SYSTEM_PROMPT
    # Bulunan bağlamı (ilanlar veya "bulunamadı" mesajı) HER ZAMAN prompt'a ekle
    final_system_prompt += "\n\nVERİTABANI ARAMA SONUCU:\n---\n" + context + "\n---"

    try:
        print("OpenAI ChatCompletion çağrılıyor...") # Debug log
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": final_system_prompt}, {"role": "user", "content": question}],
            temperature=0.7,
            max_tokens=1024,
        )
        final_answer = response.choices[0].message.content.strip()
        print(f"Üretilen Cevap: {final_answer}")
        return final_answer
    except Exception as e:
        print(f"❌ OpenAI ChatCompletion hatası: {e}")
        return "❌ Cevap üretilirken bir hata oluştu."
