# ask_handler.py (Supabase RAG Entegrasyonlu Sürüm)
"""
ask_handler.py
--------------
Kullanıcı sorularını yanıtlar. Emlak ile ilgili sorular için Supabase'den
güncel ilan verilerini çekerek Retrieval-Augmented Generation (RAG) yapar.

Ön-koşullar
-----------
• requirements.txt → openai>=1.0.0, supabase-py-async>=2.0.0
• Render ortam değişkenlerinde OPENAI_API_KEY, SUPABASE_URL, SUPABASE_KEY ayarlı.
• Supabase veritabanında 'match_listings' adında bir RPC fonksiyonu tanımlı.
"""

import os
from openai import AsyncOpenAI
from supabase_py_async import AsyncClient  # Supabase istemcisi için eklendi

# ── OpenAI istemcisi (Hem Chat hem Embedding için) ─────────────────────────────
# Bu istemci hem sohbet tamamlama hem de embedding oluşturma için kullanılacak.
# main.py'da zaten başlatılmıştı ama burada da ayrı olması sorun yaratmaz,
# ancak API key'i tekrar okumak yerine main.py'dan almak daha verimli olabilirdi.
# Şimdilik bu haliyle bırakalım, çalışacaktır.
openai_client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")
)

# --- Embedding Modeli Ayarı ---
EMBEDDING_MODEL = "text-embedding-3-small" # OpenAI'ın güncel ve uygun maliyetli modeli

# --- Supabase Arama Ayarları ---
MATCH_THRESHOLD = 0.78 # Benzerlik eşiği (0 ile 1 arası, 1'e yakın daha benzer) - Bu değeri ayarlayabilirsiniz
MATCH_COUNT = 5        # En fazla kaç benzer ilan getirileceği

# --- Sistem Talimatı (Kişilik ve RAG Yönlendirmesi) ---
# Sistem talimatını RAG'a uygun hale getiriyoruz.
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
            input=[text.strip()],  # Girdiyi liste içinde gönder
            model=EMBEDDING_MODEL
        )
        # Yanıtın veri yapısını kontrol et
        if response.data and len(response.data) > 0 and response.data[0].embedding:
             return response.data[0].embedding
        else:
             print("❌ Embedding alınamadı, OpenAI yanıtı beklenmedik formatta.")
             return None
    except Exception as e:
        print(f"❌ OpenAI Embedding hatası: {e}")
        return None


async def search_listings_in_supabase(query_embedding: list[float], db_client: AsyncClient) -> list[dict]:
    """Verilen embedding ile Supabase'de benzer ilanları arar."""
    if not query_embedding:
        return []
    try:
        # Supabase'de tanımlı olması gereken RPC fonksiyonunu çağırıyoruz
        response = await db_client.rpc(
            'match_listings', # Supabase'de oluşturacağımız fonksiyonun adı
            {
                'query_embedding': query_embedding,    # Sorgu vektörü
                'match_threshold': MATCH_THRESHOLD,    # Benzerlik eşiği
                'match_count': MATCH_COUNT             # Maksimum sonuç sayısı
            }
        ).execute()

        # response.data içinde sonuçlar liste olarak döner
        if response.data:
            # print(f"Supabase'den {len(response.data)} ilan bulundu.") # Test için log
            return response.data
        else:
            # print("Supabase'de eşleşen ilan bulunamadı.") # Test için log
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
        # İlan verilerinde 'summary', 'price', 'location', 'url' anahtarlarının olduğunu varsayıyoruz
        # Bu anahtarlar sizin bot1_remax_supabase.py scriptinizin Supabase'e yazdığı sütun adları olmalı.
        summary = listing.get('summary', 'Özet bilgisi yok')
        price = listing.get('price', 'Fiyat bilgisi yok')
        location = listing.get('location', 'Konum bilgisi yok')
        url = listing.get('url', 'URL bilgisi yok')
        # similarity = listing.get('similarity', '') # Benzerlik skorunu da ekleyebiliriz (opsiyonel)

        context += f"- Özet: {summary}\n"
        context += f"  Fiyat: {price}\n"
        context += f"  Konum: {location}\n"
        context += f"  URL: {url}\n\n"
        # context += f"  (Benzerlik: {similarity:.4f})\n\n" # Opsiyonel

    return context.strip()


# Ana fonksiyon – backend'de main.py çağırıyor
# Artık db_client parametresini de alıyor!
async def answer_question(question: str, db_client: AsyncClient) -> str:
    """
    Kullanıcıdan gelen soruyu OpenAI ChatCompletion ile yanıtla.
    Supabase'den ilgili ilanları çekip RAG uygular.
    """
    print(f"Gelen Soru: {question}") # Gelen soruyu loglayalım

    # 1. Kullanıcı sorusunun embedding'ini al
    print("Soru için embedding oluşturuluyor...")
    query_embedding = await get_embedding(question)

    # 2. Embedding ile Supabase'de benzer ilanları ara
    print("Supabase'de benzer ilanlar aranıyor...")
    listings = await search_listings_in_supabase(query_embedding, db_client)

    # 3. Bulunan ilanlardan bir bağlam metni oluştur
    print("Cevap için bağlam oluşturuluyor...")
    context = format_context(listings)
    # print(f"Oluşturulan Bağlam:\n---\n{context}\n---") # Test için log

    # 4. Gelişmiş prompt ile OpenAI'a soruyu sor
    print("OpenAI ile nihai cevap üretiliyor...")
    final_system_prompt = SYSTEM_PROMPT # Ana sistem promptunu al
    # Eğer bağlam varsa, sistem promptuna ekle
    if listings: # Sadece gerçekten ilan bulunduysa bağlamı ekle
         final_system_prompt += "\n\nKullanıcının sorusuna cevap verirken KULLANILACAK GÜNCEL İLAN BİLGİLERİ:\n---\n" + context + "\n---"
    # Eğer ilan bulunamadıysa, bu durumu belirten mesaj zaten context içinde olacak ve prompta eklenecek.

    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": final_system_prompt}, # Güncellenmiş sistem promptu
                {"role": "user", "content": question},
            ],
            temperature=0.7, # Yaratıcılık seviyesi
            max_tokens=1024, # Cevap uzunluğunu biraz artırdım, ilan bilgileri sığsın diye
        )
        final_answer = response.choices[0].message.content.strip()
        print(f"Üretilen Cevap: {final_answer}") # Üretilen cevabı loglayalım
        return final_answer

    except Exception as e:
        print(f"❌ OpenAI ChatCompletion hatası: {e}")
        return "❌ Cevap üretilirken bir hata oluştu. Lütfen tekrar deneyin."
