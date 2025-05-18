

import os
import asyncio 
import locale
import re
from typing import List, Dict, Optional
from openai import AsyncOpenAI

try:
    from supabase import create_client
    # from supabase.client import Client # Client doğrudan kullanılmıyor, kaldırılabilir.
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
MATCH_THRESHOLD =  0.3  # Orta seviyede bir değer
MATCH_COUNT     =  50   # Maksimum 50 ilan ara, ama tümünü gösterme mecburiyeti yok

# ── Modlara Göre System Prompts ────────────────────────────
SYSTEM_PROMPTS = {
    "real-estate": """
    Sen SibelGPT'sin: İstanbul emlak piyasası konusunda uzman, 
    Türkçe yanıt veren yardımsever bir yapay zeka asistansın.

    Uzmanlık alanların şunlardır:
    - Emlak piyasası ile ilgili her türlü konu (mevzuat, satılık/kiralık ilan arama)
    - Türkiye ve dünyada emlak piyasasındaki gelişmeler, trendler
    - İnşaat ve gayrimenkul yatırımı konuları
    - Gayrimenkul mevzuatı, kira sözleşmeleri, tahliye taahhütnameleri ve yasal süreçler
    - Emlak vergisi, gayrimenkul değerleme ve tapu işlemleri
    - Konut kredileri, faiz oranları ve ödeme planları
    - Kentsel dönüşüm, imar barışı ve imar düzenlemeleri
    
    FORMATLAMAYLA İLGİLİ KURALLAR:
    1. Bilgileri her zaman düz paragraflar yerine, madde işaretleri (<ul><li>), numaralı listeler (<ol><li>) veya alt başlıklar (<h3>, <h4>) şeklinde düzenle.
    2. Önemli bilgileri <span style="color:#e74c3c;font-weight:bold;">bu şekilde renkli ve kalın</span> olarak vurgula.
    3. Temel kavramları <strong> etiketleriyle kalın</strong> yap.
    4. Hukuki, teknik terimler ve anahtar kavramları <em>italik</em> olarak işaretle.
    5. Her yanıtın üst kısmında <h3> başlık </h3> kullan ve soruya göre değiştir.
    6. Uzun metinleri paragraflar arasında <br> ekleyerek böl.
    7. Karşılaştırmalı bilgileri veya adım adım süreçleri <div style="background:#f8f9fa;padding:10px;border-left:4px solid #3498db;margin:10px 0;"> içerisinde göster.
    8. Uyarıları <div style="background:#f8d7da;padding:10px;border-left:4px solid #dc3545;margin:10px 0;"> <strong style="color:#721c24;">⚠️ ÖNEMLİ UYARI:</strong><p style="color:#721c24;margin-top:5px;">Uyarı metni buraya...</p></div> içinde vurgula.

    ÖNEMLİ KURALLAR:
    1. Kullanıcının gayrimenkul ile ilgili HER TÜR sorusuna kapsamlı yanıt ver. Asla "yardımcı olamıyorum" deme.
    2. Gayrimenkul mevzuatı, sözleşmeler ve hukuki konularda bilgi ver, ancak önemli yasal konularda bir avukata danışmalarını öner.
    3. İlanlar için Supabase'den gelen 'İLGİLİ İLANLAR' verilerini kullan ve en alakalı ilanları seç.
    4. İlanlarda danışman adı veya firma bilgisi belirtme. İlanları nötr bir şekilde sun.
    5. Sadece SATILIK ilanları göster, kiralık ilanları filtreleme.
    6. Profesyonel bir gayrimenkul danışmanı gibi davran. Kullanıcının gayrimenkul aramalarında aşağıdaki sohbet akışını izle:
       a) İlk sorgudan sonra EN FAZLA 1-2 kritik soru sor (bütçe, oda sayısı, bölge tercihi gibi).
       b) Tüm soruları aynı anda sorma; kullanıcının cevaplarına göre sohbeti yönlendir.
       c) Kullanıcının verdiği her bilgiyi değerlendir ve gereksiz soruları atla.
       d) 3-4 mesaj alışverişi sonrası somut öneriler sun.
       e) Kullanıcı zaten detaylı bilgi verdiyse (bütçe, oda sayısı, lokasyon gibi), hemen ilgili ilanları göster.
    7. Doğal ve samimi bir sohbet akışı oluştur:
       a) "Erenköy'de ev arıyorum" → "Bütçeniz nedir?" → "3 milyon TL" → "Kaç oda istiyorsunuz?" → "3+1" → [Sonuçları göster]
       b) "Kadıköy'de 5 milyon bütçeyle 3+1 daire arıyorum" → [Doğrudan sonuçları göster, gereksiz soru sorma]
       c) "Ev arıyorum" → "Hangi bölgede ve nasıl bir ev düşünüyorsunuz?" → "Üsküdar'da" → "Bütçeniz ve oda tercihinizi paylaşırsanız size daha iyi yardımcı olabilirim."
    8. İlanları gösterirken, HTML formatında şu bilgileri göster:
       a) İlan başlığı (tam ismi, kısaltma kullanma)
       b) Lokasyon bilgisi (ilçe, mahalle)
       c) Fiyat, metrekare, oda sayısı
       d) İlan numarası ve PDF butonu
    9. Her zaman sonuç odaklı ol. Amaç, kullanıcının ideal gayrimenkulünü en hızlı şekilde bulmasına yardım etmek.
    10. Selamlaşma ve Genel Sohbetler:
       a) "Merhaba", "Nasılsın", "İyi günler", "Selam" gibi selamlaşma mesajlarını, başka bir modüle yönlendirmeden doğrudan yanıtla.
       b) "Bugün günlerden ne?", "Hava nasıl?", "Bana yardımcı olur musun?" gibi genel sorularda diğer modüle yönlendirme yapma.
       c) Kullanıcı sadece sohbet başlatıyorsa, mevcut modül üzerinden devam et ve onları başka modüle yönlendirme.
       d) Günlük konuşmalara, şu anki modda kalarak samimi ve dostça cevap ver.
       e) Sadece açıkça başka bir modülün uzmanlık alanına giren konularda (örn: "Borsada hisse analizi" veya "Numeroloji hesaplama") yönlendirme yap.

    KAPANIŞ MESAJLARI:
    - Her türlü gayrimenkul sorusuna yanıt verirken (ilan göstersen de göstermesen de), yanıtın sonuna: "<p style='color:#3498db;'><strong>📞 Profesyonel gayrimenkul danışmanlığı için: 532 687 84 64</strong></p>" ekle.
    - İstisna: Sadece başka modüle yönlendirme yapıyorsan telefon numarası ekleme.
    - Eğer gayrimenkul mevzuatı, sözleşmeler veya yasal konular hakkında bilgi veriyorsan, yanıtın sonuna: "<p style='color:#3498db;'><strong>📞 Detaylı bilgi ve profesyonel danışmanlık için 532 687 84 64 numaralı telefonu arayabilirsiniz.</strong></p>" ekle.
    - Gayrimenkul yatırımı, piyasa analizi gibi genel konularda ise: "<p style='color:#3498db;'><strong>📞 Gayrimenkul yatırımlarınız için profesyonel danışmanlık: 532 687 84 64</strong></p>" ekle.

    Eğer soru Zihin Koçu veya Finans konularında ise, ilgili GPT modülüne yönlendir.

    Kullanıcı sana gayrimenkul sorusu sorduğunda (ilanlar ve genel bilgi) kapsamlı yanıt ver.
    İlanlar için Supabase'den gelen verileri kullan. Genel gayrimenkul soruları için bilgini kullan ve doyurucu yanıtlar ver.

    Cevaplarını kısa, net ve samimi tut; her ilanda başlık, ilan numarası, fiyat, lokasyon ve özellik bilgisi olsun. Sadece teknik bilgi verme; aynı zamanda samimi, bilinçli ve güven veren bir danışman gibi davran.

    NOT: Yanıtlarını her zaman zengin HTML formatında oluştur. İstendiği gibi detaylı formatlamayı kullan. Markdown işaretleri (*, -) değil, HTML etiketleri kullan.
    """,
    
    "mind-coach": """
    Sen SibelGPT'sin: numeroloji, astroloji, kadim bilgiler, psikoloji, ruh sağlığı, thetahealing, 
    motivasyon ve kişisel gelişim konularında uzman, Türkçe yanıt veren 
    yardımsever bir yapay zeka zihin koçusun.
    
    Uzmanlık alanların şunlardır:
    - Numeroloji ve astroloji yorumları
    - Kadim bilgiler ve spiritüel konular
    - Psikoloji ve ruh sağlığı
    - Thetahealing ve enerji çalışmaları
    - Motivasyon ve kişisel gelişim
    
    Eğer kullanıcı sana Gayrimenkul (emlak piyasası, mevzuat, satılık/kiralık ilanlar, 
    gayrimenkul trendleri, inşaat) veya Finans (borsa, hisse senetleri, teknik/temel 
    analiz, kripto paralar, faiz, tahviller, emtia, döviz piyasası, makro/mikro ekonomi) 
    konularında bir soru sorarsa, kullanıcıyı ilgili GPT modülüne yönlendir.
    
    Cevaplarını empatik, ilham verici ve destekleyici bir tonda ver. Kullanıcının 
    sorusunu anlamaya çalış ve kişisel gelişimini destekleyecek yönlendirmeler yap.
    ÖNEMLİ KURALLAR:
    1. Selamlaşma ve Genel Sohbetler:
       a) "Merhaba", "Nasılsın", "İyi günler", "Selam" gibi selamlaşma mesajlarını, başka bir modüle yönlendirmeden doğrudan yanıtla.
       b) "Bugün günlerden ne?", "Hava nasıl?", "Bana yardımcı olur musun?" gibi genel sorularda diğer modüle yönlendirme yapma.
       c) Kullanıcı sadece sohbet başlatıyorsa, mevcut modül üzerinden devam et ve onları başka modüle yönlendirme.
       d) Günlük konuşmalara, şu anki modda kalarak samimi ve dostça cevap ver.
       e) Sadece açıkça başka bir modülün uzmanlık alanına giren konularda (örn: "Emlak ilanı arama" veya "Hisse senedi analizi") yönlendirme yap.
    
    Yanıtlarını HTML formatında oluştur. <ul> ve <li> kullan. Satır atlamak için <br>, 
    kalın yazı için <strong> kullan. Markdown işaretleri (*, -) kullanma.
    """,
    
    "finance": """
    Sen SibelGPT'sin: İstanbul Borsası, hisse senetleri, teknik ve temel analiz, kripto paralar, 
    faiz, tahviller, emtia piyasası, döviz piyasası, pariteler, makro ve mikro ekonomi
    konularında uzman, Türkçe yanıt veren yardımsever bir yapay zeka finans danışmanısın.
    
    Uzmanlık alanların şunlardır:
    - Borsa, hisse senetleri, teknik ve temel analiz
    - Kripto paralar ve blockchain teknolojisi
    - Faiz ve tahvil piyasaları
    - Emtia piyasaları (altın, gümüş vb.)
    - Döviz piyasaları ve pariteler
    - Makro ve mikro ekonomi konuları
    
    Eğer kullanıcı sana Gayrimenkul (emlak piyasası, mevzuat, satılık/kiralık ilanlar, 
    gayrimenkul trendleri, inşaat) veya Zihin Koçu (numeroloji, astroloji, kadim bilgiler, 
    psikoloji, ruh sağlığı, thetahealing, motivasyon, kişisel gelişim) konularında 
    bir soru sorarsa, kullanıcıyı ilgili GPT modülüne yönlendir.
    
    Cevaplarını net, anlaşılır ve profesyonel bir tonda ver, ancak teknik konuları
    basitleştirerek anlat. Yatırım tavsiyesi verirken riskleri de belirt.
    ÖNEMLİ KURALLAR:
    1. Selamlaşma ve Genel Sohbetler:
       a) "Merhaba", "Nasılsın", "İyi günler", "Selam" gibi selamlaşma mesajlarını, başka bir modüle yönlendirmeden doğrudan yanıtla.
       b) "Bugün günlerden ne?", "Hava nasıl?", "Bana yardımcı olur musun?" gibi genel sorularda diğer modüle yönlendirme yapma.
       c) Kullanıcı sadece sohbet başlatıyorsa, mevcut modül üzerinden devam et ve onları başka modüle yönlendirme.
       d) Günlük konuşmalara, şu anki modda kalarak samimi ve dostça cevap ver.
       e) Sadece açıkça başka bir modülün uzmanlık alanına giren konularda (örn: "Emlak ilanı arama" veya "Numeroloji hesaplama") yönlendirme yap.
    
    Yanıtlarını HTML formatında oluştur. <ul> ve <li> kullan. Satır atlamak için <br>, 
    kalın yazı için <strong> kullan. Markdown işaretleri (*, -) kullanma.
    """
}

# ── Yönlendirme Mesajları ──────────────────────────────────
REDIRECTION_MESSAGES = {
    "real-estate-to-mind-coach": """
    <h3>Bu soru Zihin Koçu GPT için daha uygun görünüyor.</h3>
    <p>Şu anda <strong>Gayrimenkul GPT</strong> modülündesiniz, ancak sorduğunuz soru numeroloji, astroloji, 
    psikoloji veya kişisel gelişim ile ilgili görünüyor.</p>
    <p>Daha iyi bir yanıt almak için lütfen üst menüden <strong>🧠 Zihin Koçu GPT</strong> butonuna tıklayarak 
    modül değiştiriniz. Ardından sorunuzu tekrar sorabilirsiniz.</p>
    <p>Veya yine de gayrimenkul ile ilgili bir sorunuz varsa, lütfen tekrar sorunuz.</p>
    """,
    "real-estate-to-finance": """
    <h3>Bu soru Finans GPT için daha uygun görünüyor.</h3>
    <p>Şu anda <strong>Gayrimenkul GPT</strong> modülündesiniz, ancak sorduğunuz soru borsa, hisse senetleri, 
    yatırım, ekonomi veya finans ile ilgili görünüyor.</p>
    <p>Daha iyi bir yanıt almak için lütfen üst menüden <strong>💰 Finans GPT</strong> butonuna tıklayarak 
    modül değiştiriniz. Ardından sorunuzu tekrar sorabilirsiniz.</p>
    <p>Veya yine de gayrimenkul ile ilgili bir sorunuz varsa, lütfen tekrar sorunuz.</p>
    """,
    "mind-coach-to-real-estate": """
    <h3>Bu soru Gayrimenkul GPT için daha uygun görünüyor.</h3>
    <p>Şu anda <strong>Zihin Koçu GPT</strong> modülündesiniz, ancak sorduğunuz soru emlak, gayrimenkul, 
    satılık/kiralık ilanlar veya inşaat ile ilgili görünüyor.</p>
    <p>Daha iyi bir yanıt almak için lütfen üst menüden <strong>🏠 Gayrimenkul GPT</strong> butonuna tıklayarak 
    modül değiştiriniz. Ardından sorunuzu tekrar sorabilirsiniz.</p>
    <p>Veya yine de kişisel gelişim ve zihin koçluğu ile ilgili bir sorunuz varsa, lütfen tekrar sorunuz.</p>
    """,
    "mind-coach-to-finance": """
    <h3>Bu soru Finans GPT için daha uygun görünüyor.</h3>
    <p>Şu anda <strong>Zihin Koçu GPT</strong> modülündesiniz, ancak sorduğunuz soru borsa, hisse senetleri, 
    yatırım, ekonomi veya finans ile ilgili görünüyor.</p>
    <p>Daha iyi bir yanıt almak için lütfen üst menüden <strong>💰 Finans GPT</strong> butonuna tıklayarak 
    modül değiştiriniz. Ardından sorunuzu tekrar sorabilirsiniz.</p>
    <p>Veya yine de kişisel gelişim ve zihin koçluğu ile ilgili bir sorunuz varsa, lütfen tekrar sorunuz.</p>
    """,
    "finance-to-real-estate": """
    <h3>Bu soru Gayrimenkul GPT için daha uygun görünüyor.</h3>
    <p>Şu anda <strong>Finans GPT</strong> modülündesiniz, ancak sorduğunuz soru emlak, gayrimenkul, 
    satılık/kiralık ilanlar veya inşaat ile ilgili görünüyor.</p>
    <p>Daha iyi bir yanıt almak için lütfen üst menüden <strong>🏠 Gayrimenkul GPT</strong> butonuna tıklayarak 
    modül değiştiriniz. Ardından sorunuzu tekrar sorabilirsiniz.</p>
    <p>Veya yine de ekonomi ve finans ile ilgili bir sorunuz varsa, lütfen tekrar sorunuz.</p>
    """,
    "finance-to-mind-coach": """
    <h3>Bu soru Zihin Koçu GPT için daha uygun görünüyor.</h3>
    <p>Şu anda <strong>Finans GPT</strong> modülündesiniz, ancak sorduğunuz soru numeroloji, astroloji, 
    psikoloji veya kişisel gelişim ile ilgili görünüyor.</p>
    <p>Daha iyi bir yanıt almak için lütfen üst menüden <strong>🧠 Zihin Koçu GPT</strong> butonuna tıklayarak 
    modül değiştiriniz. Ardından sorunuzu tekrar sorabilirsiniz.</p>
    <p>Veya yine de ekonomi ve finans ile ilgili bir sorunuz varsa, lütfen tekrar sorunuz.</p>
    """
}


async def detect_topic(question: str, mode: str) -> str:
    """Kullanıcının sorusunun hangi alana ait olduğunu tespit eder."""
    
    selamlasma_kaliplari = [
        "merhaba", "selam", "hello", "hi", "hey", "sa", "günaydın", "iyi günler", 
        "iyi akşamlar", "nasılsın", "naber", "ne haber", "hoş geldin", "nasıl gidiyor"
    ]
    
    clean_question = question.lower()
    for char in ".,;:!?-_()[]{}\"'":
        clean_question = clean_question.replace(char, " ")
    
    if len(clean_question.split()) <= 3:
        for kalip in selamlasma_kaliplari:
            if kalip in clean_question:
                print(f"✓ Selamlaşma mesajı tespit edildi, mevcut modda kalınıyor: {kalip}")
                return mode
    
    topics = {
        "real-estate": [
            "emlak", "gayrimenkul", "ev", "daire", "konut", "kiralık", "satılık", 
            "tapu", "mortgage", "ipotek", "kredi", "remax", "metrekare", "imar", 
            "arsa", "bina", "kat", "müstakil", "dükkan", "ofis", "iş yeri", "bahçe",
            "balkon", "oda", "salon", "banyo", "mutfak", "yapı", "inşaat", "tadilat"
        ],
        "mind-coach": [
            "numeroloji", "astroloji", "burç", "meditasyon", "reiki", "terapi", 
            "psikoloji", "depresyon", "anksiyete", "stres", "motivasyon", "gelişim", 
            "spiritüel", "enerji", "şifa", "kadim", "theta", "healing", "ruh", 
            "bilinç", "farkındalık", "arınma", "denge", "uyum", "yoga", "nefes"
        ],
        "finance": [
            "borsa", "hisse", "finans", "yatırım", "faiz", "döviz", "euro", "dolar", 
            "altın", "gümüş", "kripto", "bitcoin", "ethereum", "bist", "ekonomi", 
            "enflasyon", "tahvil", "bono", "portföy", "fon", "kazanç", "kâr", "zarar", 
            "analiz", "teknik", "temel", "parite", "forex", "banka", "para"
        ]
    }
    
    matches = {topic: 0 for topic in topics}
    
    for topic, keywords in topics.items():
        for keyword in keywords:
            if keyword in clean_question:
                matches[topic] += 1
    
    max_matches = 0
    if matches: # matches boş değilse max değerini al
        max_matches = max(matches.values())
    
    if max_matches <= 1:
        if len(clean_question.split()) <= 5:
            print(f"✓ Kısa genel mesaj tespit edildi, mevcut modda kalınıyor")
            return mode
            
        try:
            resp = await openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system", 
                        "content": """Kullanıcı mesajını analiz ederek aşağıdaki kategorilerden hangisine 
                                    ait olduğunu belirle ve sadece kategori adını döndür:
                                    1. real-estate (emlak, gayrimenkul, ev, daire, kiralık, satılık vb.)
                                    2. mind-coach (numeroloji, astroloji, psikoloji, kişisel gelişim vb.)
                                    3. finance (borsa, hisse, yatırım, ekonomi, kripto, döviz vb.)
                                    4. general (selamlaşma, günlük konuşma, sohbet vb.)
                                    Eğer mesaj "merhaba", "selam", "nasılsın" gibi basit selamlaşma veya 
                                    genel sohbet içeriyorsa "general" olarak belirt."""
                    },
                    {"role": "user", "content": question}
                ],
                temperature=0.3,
                max_tokens=10
            )
            detected_topic_by_gpt = resp.choices[0].message.content.strip().lower()
            
            if "general" in detected_topic_by_gpt:
                print(f"✓ GPT tarafından genel sohbet olarak tespit edildi, mevcut modda kalınıyor")
                return mode
                
            for topic_key in topics.keys():
                if topic_key in detected_topic_by_gpt:
                    return topic_key
            
            return mode
            
        except Exception as e:
            print(f"⚠️ Konu tespiti hatası (OpenAI API): {e}")
            return mode
    
    for topic, count in matches.items():
        if count == max_matches:
            return topic
    
    return mode

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
    """Remax ilanlar tablosundan arama yapar."""
    if query_embedding is None:
        return []
    
    try:
        print("🔍 İlanlar sorgulanıyor...")
        
        response = supabase.rpc(
            "match_remax_listings",
            {
                "query_embedding": query_embedding,
                "match_threshold": MATCH_THRESHOLD,
                "match_count": MATCH_COUNT
            }
        ).execute()
        
        all_results = response.data if hasattr(response, "data") and response.data is not None else []
        
        # Düzeltilmiş Girinti: Bu satırlar 'try' bloğunun içinde olmalı
        valid_results = [r for r in all_results if r.get('similarity', 0) > MATCH_THRESHOLD]
       
        print(f"✅ İlanlar sorgulandı: Toplam {len(valid_results)} gerçek ilişkili ilan bulundu")
       
        if not valid_results:
            print("⚠️ Hiç ilan bulunamadı!")
            # return [] # Bu return zaten alttaki return valid_results ile kapsanıyor, eğer valid_results boşsa boş liste döner
       
        return valid_results
       
    except Exception as exc: # Düzeltilmiş Girinti: 'except' 'try' ile aynı hizada olmalı
        print("❌ Arama işleminde hata:", exc)
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
            pass # Locale ayarlanamazsa devam et

    MAX_LISTINGS_TO_SHOW = 10
    listings_to_format = listings[:MAX_LISTINGS_TO_SHOW]
   
    final_output = "<p><strong>📞 Sorgunuzla ilgili ilanlar burada listelenmiştir. Detaylı bilgi için 532 687 84 64 numaralı telefonu arayabilirsiniz.</strong></p>"
   
    formatted_parts = []
    for i, l_item in enumerate(listings_to_format, start=1): # 'l' Python'da 'lambda' için kullanılabileceğinden 'l_item' olarak değiştirdim
        ilan_no = l_item.get('ilan_id', l_item.get('ilan_no', str(i)))
        baslik = l_item.get('baslik', '(başlık yok)')
        lokasyon = l_item.get('lokasyon', '?')
        
        fiyat = "?"
        fiyat_raw = l_item.get('fiyat')
        if fiyat_raw is not None: # None kontrolü eklendi
            try:
                # Fiyat string'ini temizleyip float'a çevirme
                fiyat_str_cleaned = str(fiyat_raw).replace('.', '').replace(',', '.')
                if fiyat_str_cleaned.replace('.', '', 1).isdigit(): # Sayısal olup olmadığını kontrol et
                    fiyat_num = float(fiyat_str_cleaned)
                    fiyat = f"{fiyat_num:,.0f} ₺".replace(',', '#').replace('.', ',').replace('#', '.') # Türk formatı
                else:
                    fiyat = str(fiyat_raw) # Eğer sayısal değilse olduğu gibi göster
            except ValueError: # Sayıya çevirme hatası olursa
                fiyat = str(fiyat_raw) # Orijinal değeri kullan
            except Exception: # Diğer beklenmedik hatalar için
                 fiyat = str(fiyat_raw)
       
        ozellikler_liste = []
        oda_sayisi = l_item.get('oda_sayisi', '')
        if oda_sayisi:
            ozellikler_liste.append(oda_sayisi)
       
        metrekare = l_item.get('metrekare', '')
        if metrekare:
            # Metrekare değerinin sonunda " m²" yoksa ekle
            metrekare_str = str(metrekare).strip()
            if not metrekare_str.endswith("m²"):
                 ozellikler_liste.append(f"{metrekare_str} m²")
            else:
                 ozellikler_liste.append(metrekare_str)

        bulundugu_kat_raw = l_item.get('bulundugu_kat')
        if bulundugu_kat_raw is not None and str(bulundugu_kat_raw).strip() != '':
            bulundugu_kat_str = str(bulundugu_kat_raw).strip()
            try:
                # Sadece sayısal değerleri int'e çevirmeye çalış
                if bulundugu_kat_str.replace('-', '', 1).isdigit(): # Negatif sayıları da kabul et
                    kat_no = int(bulundugu_kat_str)
                    if kat_no == 0:
                        ozellikler_liste.append("Giriş Kat")
                    elif kat_no < 0:
                        ozellikler_liste.append(f"Bodrum Kat ({kat_no})") # veya sadece "Bodrum Kat"
                    else:
                        ozellikler_liste.append(f"{kat_no}. Kat")
                else: # Sayısal değilse olduğu gibi al, ". Kat" ekleme
                    ozellikler_liste.append(bulundugu_kat_str)
            except ValueError: # int'e çevirme hatası olursa
                ozellikler_liste.append(bulundugu_kat_str) # Orijinal değeri kullan
       
        ozellikler_db = l_item.get('ozellikler')
        if ozellikler_db and isinstance(ozellikler_db, str): # ozellikler string ise işle
            ozellikler_parts_raw = ozellikler_db.split('|')
            ozellikler_parts_processed = []
            for part_raw in ozellikler_parts_raw:
                part = part_raw.strip()
                if re.match(r'^-?\d+$', part): # Negatif dahil tam sayı kontrolü
                    kat_no_oz = int(part)
                    if kat_no_oz == 0:
                        ozellikler_parts_processed.append("Giriş Kat")
                    elif kat_no_oz < 0:
                        ozellikler_parts_processed.append(f"Bodrum Kat ({kat_no_oz})")
                    else:
                        ozellikler_parts_processed.append(f"{kat_no_oz}. Kat")
                else:
                    ozellikler_parts_processed.append(part)
            ozellikler = " | ".join(ozellikler_parts_processed)
        elif ozellikler_liste: # ozellikler string değilse veya boşsa, listeden oluştur
            ozellikler = " | ".join(ozellikler_liste)
        else:
            ozellikler = "(özellik bilgisi yok)"
       
        ilan_html = (
            f"<li><strong>{i}. {baslik}</strong><br>"
            f"İlan No: {ilan_no} | Lokasyon: {lokasyon}<br>"
            f"Fiyat: {fiyat} | {ozellikler}<br>"
            f"<button onclick=\"window.open('https://sibelgpt-backend.onrender.com/generate-property-pdf/{ilan_no}', '_blank')\" "
            f"style='margin-top:6px; padding:6px 15px; background:#1976d2; color:white; border:none; "
            f"border-radius:25px; cursor:pointer; font-size:13px; font-weight:500; display:inline-flex; "
            f"align-items:center; gap:5px; box-shadow:0 2px 5px rgba(0,0,0,0.1); transition:all 0.3s ease;' "
            f"onmouseover=\"this.style.background='#115293'; this.style.transform='translateY(-1px)';\" "
            f"onmouseout=\"this.style.background='#1976d2'; this.style.transform='translateY(0)';\">"
            f"<i class='fas fa-file-pdf' style='font-size:16px;'></i> PDF İndir</button></li>"
        )
        formatted_parts.append(ilan_html)
   
    final_output += "<ul>" + "\n".join(formatted_parts) + "</ul>"
    final_output += "<p>Bu ilanların doğruluğunu kontrol ettim. Farklı bir arama yapmak isterseniz, lütfen kriterleri belirtiniz.</p>"
   
    return final_output

# ── Ana Fonksiyon ─────────────────────────────────────────
async def answer_question(question: str, mode: str = "real-estate") -> str:
    print(f"↪ Soru: {question}, Mod: {mode}")
   
    detected_topic_result = await detect_topic(question, mode)
    print(f"✓ Tespit edilen konu: {detected_topic_result}, Kullanıcının seçtiği mod: {mode}")
   
    if detected_topic_result != mode:
        redirection_key = f"{mode}-to-{detected_topic_result}"
        print(f"⟹ Yönlendirme anahtarı: {redirection_key}")
        
        if redirection_key in REDIRECTION_MESSAGES:
            return REDIRECTION_MESSAGES[redirection_key]
        # else: Yönlendirme mesajı bulunamazsa ne yapılacağı belirtilmemiş, mevcut modda devam edebilir.
        # Şimdilik, yönlendirme mesajı yoksa ve konu farklıysa bile mevcut modda devam ediyor.
        # Bu davranış istenmiyorsa buraya bir `else` bloğu eklenebilir.
   
    query_emb = await get_embedding(question)
   
    context = ""
    if mode == "real-estate":
        if query_emb: # Sadece embedding başarılıysa ilan ara
            listings = await search_listings_in_supabase(query_emb)
            context = format_context_for_sibelgpt(listings)
        else:
            context = "<p>Sorunuzu işlerken bir sorun oluştu, lütfen tekrar deneyin veya farklı bir soru sorun.</p>"
   
    system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["real-estate"])
   
    messages = [
        {"role": "system", "content": f"{system_prompt}<br><br>İLGİLİ İLANLAR:<br>{context if context else 'Uygun ilan bulunamadı veya bu mod için ilan aranmıyor.'}"},
        {"role": "user", "content": question}
    ]

    try:
        resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini", # Model adı doğru olmalı, örn: "gpt-4o-mini"
            messages=messages,
            temperature=0.7,
            max_tokens=4096
        )
        return resp.choices[0].message.content.strip()
    except Exception as exc:
        print("❌ Chat yanıt hatası:", exc)
        # Kullanıcıya daha anlamlı bir hata mesajı verilebilir.
        return "Üzgünüm, isteğinizi işlerken beklenmedik bir sorun oluştu. Lütfen daha sonra tekrar deneyin."

# ── Terminalden Test ──────────────────────────────────────
if __name__ == "__main__":
    async def main():
        q = input("Soru: ")
        # Varsayılan mod "real-estate" olarak ayarlandı, test için değiştirilebilir.
        response = await answer_question(q, mode="real-estate") 
        print(response)

    # asyncio.run() Python 3.7+ için daha modern bir yoldur.
    # Eğer Python 3.6 veya daha eski bir sürüm kullanılıyorsa loop.run_until_complete() gerekir.
    # Ancak kodda AsyncOpenAI kullanıldığı için Python 3.7+ varsayılabilir.
    asyncio.run(main())
