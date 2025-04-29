# ask_handler.py (await ... execute() Kullanımı)
import os
from openai import AsyncOpenAI

# --- AsyncClient import denemesi ---
try:
    # Doğrudan supabase'ten import etmeyi dene
    from supabase import AsyncClient
except ImportError:
    try:
        # Eski yolu dene
        from supabase.lib.client_async import AsyncClient
    except ImportError:
        AsyncClient = None # Tip kontrolü için None ata
# ---------------------------------


# ... (OpenAI client, Ayarlar, SYSTEM_PROMPT, get_embedding) ...


async def search_listings_in_supabase(query_embedding: list[float], db_client: AsyncClient) -> list[dict]:
    """Verilen embedding ile Supabase'de benzer ilanları arar."""
    if not db_client or not AsyncClient: return []
    if not query_embedding: return []
    try:
        print(f"Supabase RPC 'match_listings' çağrılıyor (AsyncClient ile)...")
        # --- ASENKRON İSTEMCİ İÇİN DOĞRU YÖNTEM ---
        response = await db_client.rpc(
            'match_listings',
            {'query_embedding': query_embedding, 'match_threshold': MATCH_THRESHOLD, 'match_count': MATCH_COUNT}
        ).execute() # execute() GEREKLİ!
        # ------------------------------------------
        print(f"Supabase RPC yanıtı alındı. Data: {response.data if hasattr(response, 'data') else 'Yanıt formatı beklenmedik'}")
        if hasattr(response, 'data') and response.data:
             return response.data
        else: return []
    except Exception as e:
        print(f"❌ Supabase RPC hatası: {e}")
        return []

# ... format_context (ozet, fiyat kullanıldığına emin olun) ...
def format_context(listings: list[dict]) -> str:
    if not listings:
        return "Veritabanında bu soruyla ilgili güncel ilan bulunamadı."
    context = "İlgili olabilecek güncel ilanlar:\n\n"
    for listing in listings:
        summary = listing.get('ozet', 'Özet bilgisi yok') # 'ozet' kullandığınızdan emin olun
        price = listing.get('fiyat', 'Fiyat bilgisi yok') # 'fiyat' kullandığınızdan emin olun
        location = listing.get('location', 'Konum bilgisi yok')
        url = listing.get('url', 'URL bilgisi yok')
        context += f"- Özet: {summary}\n  Fiyat: {price}\n  Konum: {location}\n  URL: {url}\n\n"
    return context.strip()


async def answer_question(question: str, db_client: AsyncClient) -> str: # Gerçek tipi kullan
    if not db_client or not AsyncClient:
        return "❌ Dahili bir hata oluştu (Supabase istemcisi yok)."

    print(f"Gelen Soru: {question}")
    query_embedding = await get_embedding(question)
    listings = await search_listings_in_supabase(query_embedding, db_client)
    context = format_context(listings)
    final_system_prompt = SYSTEM_PROMPT
    final_system_prompt += "\n\nVERİTABANI ARAMA SONUCU:\n---\n" + context + "\n---"
    try:
        print("OpenAI ChatCompletion çağrılıyor...")
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system", "content": final_system_prompt}, {"role": "user", "content": question}],
            temperature=0.7, max_tokens=1024,
        )
        final_answer = response.choices[0].message.content.strip()
        print(f"Üretilen Cevap: {final_answer}")
        return final_answer
    except Exception as e:
        print(f"❌ OpenAI ChatCompletion hatası: {e}")
        return "❌ Cevap üretilirken bir hata oluştu."
