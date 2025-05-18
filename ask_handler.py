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
1    1. Selamlaşma ve Genel Sohbetler:
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
1   1. Selamlaşma ve Genel Sohbetler:
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


async def detect_topic(question: str) -> str:
    """Kullanıcının sorusunun hangi alana ait olduğunu tespit eder."""
    
    # Alanları tanımlayan anahtar kelimeler
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
    
    # Soruyu küçük harfe çevirip noktolama işaretlerini kaldıralım
    clean_question = question.lower()
    for char in ".,;:!?-_()[]{}\"'":
        clean_question = clean_question.replace(char, " ")
    
    # Her kategori için eşleşen kelime sayısını sayalım
    matches = {topic: 0 for topic in topics}
    
    for topic, keywords in topics.items():
        for keyword in keywords:
            if keyword in clean_question:
                matches[topic] += 1
    
    # En çok eşleşen kategoriyi bulalım
    max_matches = max(matches.values())
    
    # Eğer hiç eşleşme yoksa veya çok az eşleşme varsa, içeriği belirsiz olarak işaretleyelim
    if max_matches <= 1:
        # Yeterli eşleşme yoksa, sorunun içeriğini OpenAI API ile analiz et
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
                                    Eğer mesaj belirgin bir kategoriye ait değilse, en yakın kategoriyi seç."""
                    },
                    {"role": "user", "content": question}
                ],
                temperature=0.3,
                max_tokens=10
            )
            detected_topic = resp.choices[0].message.content.strip().lower()
            
            # Eğer yanıt direkt kategori adı değilse, içindeki kategori adını çıkaralım
            for topic in topics.keys():
                if topic in detected_topic:
                    return topic
            
            # Hala belirleyemediyse, varsayılan olarak real-estate döndür
            return "real-estate"
            
        except Exception as e:
            print(f"⚠️ Konu tespiti hatası: {e}")
            # Hata durumunda varsayılan olarak real-estate döndür
            return "real-estate"
    
    # En çok eşleşme olan kategoriyi döndür
    for topic, count in matches.items():
        if count == max_matches:
            return topic
    
    # Hiçbir şey bulunamazsa varsayılan olarak real-estate döndür
    return "real-estate"

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
        # İlanları sorgula
        print("🔍 İlanlar sorgulanıyor...")
        
        response = supabase.rpc(
            "match_remax_listings",
            {
                "query_embedding": query_embedding,
                "match_threshold": MATCH_THRESHOLD,
                "match_count": MATCH_COUNT  # Maksimum sayı
            }
        ).execute()
        
        all_results = response.data if hasattr(response, "data") else []
        
        # Gerçek sonuç sayısını göster - benzerlik puanına göre filtreleme
        valid_results = [r for r in all_results if r.get('similarity', 0) > MATCH_THRESHOLD]
        
        print(f"✅ İlanlar sorgulandı: Toplam {len(valid_results)} gerçek ilişkili ilan bulundu")
        
        if not valid_results:
            print("⚠️ Hiç ilan bulunamadı!")
            return []
        
        return valid_results
        
    except Exception as exc:
        print("❌ Arama işleminde hata:", exc)
        return []

# ── Formatlama Fonksiyonu ─────────────────────────────────
def format_context_for_sibelgpt(listings: List[Dict]) -> str:
    """İlanları formatlayarak eksiksiz HTML'e dönüştürür ve PDF butonu ekler."""
    if not listings:
        return "🔍 Uygun ilan bulunamadı."

    # Locale ayarı
    try:
        locale.setlocale(locale.LC_ALL, 'tr_TR.UTF-8')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, 'tr_TR')
        except locale.Error:
            pass

    # Maksimum ilan sayısını sınırlama - SibelGPT yanıt sınırlamasına uygun
    MAX_LISTINGS_TO_SHOW = 10  # Daha fazla ilan göstermek için artırıldı
    listings_to_format = listings[:MAX_LISTINGS_TO_SHOW]
    
    # Toplam ve gösterilen ilan sayısını hesapla
    total_count = len(listings)
    shown_count = len(listings_to_format)
    
    # Açıklayıcı mesaj ve telefon numarasını birleştir
    final_output = "<p><strong>📞 Sorgunuzla ilgili ilanlar burada listelenmiştir. Detaylı bilgi için 532 687 84 64 numaralı telefonu arayabilirsiniz.</strong></p>"
    
    formatted_parts = []
    for i, l in enumerate(listings_to_format, start=1):
        # İlan numarası - ilan_id veya ilan_no alanından
        ilan_no = l.get('ilan_id', l.get('ilan_no', str(i)))
        
        # Başlık - tam başlığı göster, kısaltma yapma
        baslik = l.get('baslik', '(başlık yok)')
        
        # Lokasyon - tam haliyle göster
        lokasyon = l.get('lokasyon', '?')
        
        # Fiyat formatlaması
        fiyat = "?"
        fiyat_raw = l.get('fiyat')
        if fiyat_raw:
            try:
                fiyat_num = float(str(fiyat_raw).replace('.', '').replace(',', '.'))
                fiyat = f"{fiyat_num:,.0f} ₺".replace(',', '.') 
            except:
                fiyat = str(fiyat_raw)
        
        # Özellikler - tüm bilgileri dahil et
        ozellikler_liste = []
        
        # Oda sayısı - doğrudan al
        oda_sayisi = l.get('oda_sayisi', '')
        if oda_sayisi:
            ozellikler_liste.append(oda_sayisi)
        
        # Metrekare - doğrudan al
        metrekare = l.get('metrekare', '')
        if metrekare:
            ozellikler_liste.append(f"{metrekare} m²")
        
        # Kat bilgisi - bulundugu_kat alanından
        bulundugu_kat = l.get('bulundugu_kat')
        if bulundugu_kat is not None and bulundugu_kat != '':
            try:
                # Artık tam sayı olduğunu biliyoruz
                kat_no = int(bulundugu_kat)
                
                # Özel durumlar için kontrol
                if kat_no == 0:
                    ozellikler_liste.append("Giriş Kat")
                elif kat_no < 0:
                    ozellikler_liste.append("Bodrum Kat")
                else:
                    # Tam sayıya "Kat" kelimesini ekleyelim
                    ozellikler_liste.append(f"{kat_no}. Kat")
            except:
                # Sayı olarak çevrilemezse olduğu gibi göster ama "Kat" ifadesini ekle
                if "kat" not in str(bulundugu_kat).lower():
                    ozellikler_liste.append(f"{bulundugu_kat}. Kat")
                else:
                    ozellikler_liste.append(f"{bulundugu_kat}")
        
        # Özellikler string'i - varsa alanı kullan, yoksa liste oluştur
        if 'ozellikler' in l and l['ozellikler']:
            ozellikler = l['ozellikler']
            
            # Tek başına sayı olan alanları bul ve "X. Kat" olarak değiştir
            ozellikler_parts = ozellikler.split('|')
            for i, part in enumerate(ozellikler_parts):
                part = part.strip()
                # Eğer bu kısım sadece bir sayı ise
                if re.match(r'^\d+$', part):
                    kat_no = int(part)
                    if kat_no == 0:
                        ozellikler_parts[i] = "Giriş Kat"
                    elif kat_no < 0:
                        ozellikler_parts[i] = "Bodrum Kat"
                    else:
                        ozellikler_parts[i] = f"{kat_no}. Kat"
            
            ozellikler = " | ".join(ozellikler_parts)
        else:
            ozellikler = " | ".join(ozellikler_liste) if ozellikler_liste else "(özellik bilgisi yok)"
        
        # HTML oluştur - başlık kırpılmadan, tüm bilgiler dahil edilmiş ve PDF butonu eklenmiş
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
    
    # Liste HTML'i ekle
    final_output += "<ul>" + "\n".join(formatted_parts) + "</ul>"
    
    final_output += "<p>Bu ilanların doğruluğunu kontrol ettim. Farklı bir arama yapmak isterseniz, lütfen kriterleri belirtiniz.</p>"
    
    return final_output

# ── Ana Fonksiyon ─────────────────────────────────────────
async def answer_question(question: str, mode: str = "real-estate") -> str:
    """Kullanıcının sorusuna yanıt verir ve gerektiğinde başka modüle yönlendirir."""
    
    print(f"↪ Soru: {question}, Mod: {mode}")
    
    # Sorunun hangi alana ait olduğunu tespit et
    detected_topic = await detect_topic(question)
    
    # Tanılama için loglama ekle
    print(f"✓ Tespit edilen konu: {detected_topic}, Kullanıcının seçtiği mod: {mode}")
    
    # Eğer tespit edilen konu, seçili moddan farklıysa yönlendirme mesajı göster
    if detected_topic != mode:
        redirection_key = f"{mode}-to-{detected_topic}"
        print(f"⟹ Yönlendirme anahtarı: {redirection_key}")
        
        if redirection_key in REDIRECTION_MESSAGES:
            return REDIRECTION_MESSAGES[redirection_key]
    
    # Normal işleme devam et
    query_emb = await get_embedding(question)
    
    # Gayrimenkul modu için Supabase'den ilanları getir
    if mode == "real-estate":
        listings = await search_listings_in_supabase(query_emb)
        context = format_context_for_sibelgpt(listings)
    else:
        # Diğer modlar için boş context
        context = ""
    
    # Seçili moda göre system prompt'u al
    system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["real-estate"])
    
    messages = [
        {"role": "system", "content": f"{system_prompt}<br><br>{context}"},
        {"role": "user", "content": question}
    ]

    try:
        resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=4096
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
