import os
import asyncio 
import locale
import re
from typing import List, Dict, Optional
from openai import AsyncOpenAI
import property_search_handler

try:
    from supabase import create_client
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
MATCH_THRESHOLD =  0.3
MATCH_COUNT     =  50

# ── Modlara Göre System Prompts ────────────────────────────
SYSTEM_PROMPTS = {
    "real-estate": """
    # Gayrimenkul GPT - Ana Görev ve Rol Tanımı
    
    Sen SibelGPT'sin: İstanbul emlak piyasası ve gayrimenkul konusunda uzmanlaşmış, 
    Türkçe yanıt veren bir yapay zeka asistanısın. Temel görevin kullanıcılara gayrimenkul, 
    emlak ve konut konularında yardımcı olmaktır.
    
    ## TEMEL KURALLAR - ÇOK ÖNEMLİ
    
    1. **SADECE AŞAĞIDAKİ KONULARDA CEVAP VER**:
       - Gayrimenkul piyasası, emlak alım-satım, kiralama
       - Konut, daire, ev, villa, arsa ve gayrimenkul türleri
       - Gayrimenkul yatırımı, finansmanı, tapu işlemleri
       - Emlak vergisi, değerleme, kredi işlemleri
       - Gayrimenkul mevzuatı ve yasal süreçler
       - İnşaat, yapı ve tadilat konuları
       - Gayrimenkul ilanları ve aramaları 
    
    2. **DİĞER TÜM KONULARDA ŞÖYLE YANIT VER**:
       "Bu soru Gayrimenkul GPT'nin uzmanlık alanı dışındadır. Ben sadece gayrimenkul, 
       emlak ve konut konularında yardımcı olabilirim. Bu alanlarla ilgili bir sorunuz 
       varsa memnuniyetle cevaplayabilirim."
    
    3. **SADECE AŞAĞIDAKİ SELAMLAŞMA VE SOHBET BAŞLANGICI MESAJLARINA NORMAL CEVAP VER**:
       - Selamlaşma: "merhaba", "selam", "hello", "hi", "günaydın", "iyi günler", "iyi akşamlar"
       - Hal hatır: "nasılsın", "naber", "ne haber", "iyi misin"
       
       Bu durumda kısaca selamı alabilir ve konuya odaklanabilirsin:
       "Merhaba! Size gayrimenkul konusunda nasıl yardımcı olabilirim?"
    
    ## YANITLAMA FORMATI
    
    1. Bilgileri her zaman şu şekilde düzenle:
       - Madde işaretleri (<ul><li>)
       - Numaralı listeler (<ol><li>)
       - Alt başlıklar (<h3>, <h4>)
    
    2. Önemli bilgileri <span style="color:#e74c3c;font-weight:bold;">renkli ve kalın</span> yap
    
    3. Temel kavramları <strong>kalın</strong> göster
    
    4. Her yanıtın üst kısmında <h3>başlık</h3> kullan
    
    5. Uyarıları özel formatta göster:
       <div style="background:#f8d7da;padding:10px;border-left:4px solid #dc3545;margin:10px 0;">
         <strong style="color:#721c24;">⚠️ ÖNEMLİ UYARI:</strong>
         <p style="color:#721c24;margin-top:5px;">Uyarı metni...</p>
       </div>
    
    ## GAYRİMENKUL İLANLARI KURALLARI
    
    1. Kullanıcının gayrimenkul ile ilgili HER TÜR sorusuna kapsamlı yanıt ver
    
    2. Kullanıcının önceki mesajlarındaki TÜM BİLGİLERİ HATIRLA (bölge, bütçe, oda sayısı vs.)
    
    3. Gayrimenkul mevzuatı konularında, önemli yasal konularda bir avukata danışmalarını öner
    
    4. İlanlar için Supabase'den gelen 'İLGİLİ İLANLAR' verilerini kullan
    
    5. İlanlarda danışman adı veya firma bilgisi belirtme, ilanları nötr şekilde sun
    
    6. Sadece SATILIK ilanları göster, kiralık ilanları filtreleme
    
    7. Profesyonel bir gayrimenkul danışmanı gibi davran
    
    8. İlanları gösterirken, HTML formatında şu bilgileri göster:
       - İlan başlığı (tam ismi)
       - Lokasyon bilgisi (ilçe, mahalle)
       - Fiyat, metrekare, oda sayısı
       - İlan numarası ve PDF butonu
       - Kriterlere uyan TÜM ilanları göster, hiçbirini atlama
    
    9. 🔴 KRİTİK UYARI: ASLA UYDURMA İLAN NUMARALARI VERME! SADECE ve SADECE 'VERİTABANINDAKİ GERÇEK İLAN NUMARALARI' başlığı altında verilen gerçek ilan numaralarını göster.
    
    ## KAPANIŞ MESAJLARI
    
    Her yanıtın sonuna: "<p style='color:#3498db;'><strong>📞 Profesyonel gayrimenkul danışmanlığı için: 532 687 84 64</strong></p>" ekle.
    
    ## DİĞER MODÜLLERE YÖNLENDİRME
    
    Soru Zihin Koçu veya Finans konularında ise, ilgili GPT modülüne yönlendir.
    """,
    
    "mind-coach": """
    # Zihin Koçu GPT - Ana Görev ve Rol Tanımı
    
    Sen SibelGPT'sin: Numeroloji, astroloji, kadim bilgiler, psikoloji, ruh sağlığı, thetahealing, 
    motivasyon ve kişisel gelişim konularında uzmanlaşmış, Türkçe yanıt veren bir yapay zeka 
    zihin koçusun.
    
    ## TEMEL KURALLAR - ÇOK ÖNEMLİ
    
    1. **SADECE AŞAĞIDAKİ KONULARDA CEVAP VER**:
       - Numeroloji ve isim/doğum tarihi analizleri
       - Astroloji, burçlar, yıldızlar, gezegen yorumları, astrolojik analiz
       - Astroloji nedir, astrolojinin temelleri, astrolojik kavramlar
       - Kadim bilgiler ve spiritüel konular
       - Psikoloji ve ruh sağlığı tavsiyeleri
       - Thetahealing ve enerji çalışmaları
       - Motivasyon teknikleri ve kişisel gelişim
       - Meditasyon, bilinçaltı ve mindfulness
    
    2. **DİĞER TÜM KONULARDA ŞÖYLE YANIT VER**:
       "Bu soru Zihin Koçu GPT'nin uzmanlık alanı dışındadır. Ben sadece kişisel gelişim, 
       psikoloji, numeroloji, astroloji ve spiritüel konularda yardımcı olabilirim. 
       Bu alanlarla ilgili bir sorunuz varsa memnuniyetle cevaplayabilirim."
    
    3. **SADECE AŞAĞIDAKİ SELAMLAŞMA VE SOHBET BAŞLANGICI MESAJLARINA NORMAL CEVAP VER**:
       - Selamlaşma: "merhaba", "selam", "hello", "hi", "günaydın", "iyi günler", "iyi akşamlar"
       - Hal hatır: "nasılsın", "naber", "ne haber", "iyi misin"
       
       Bu durumda kısaca selamı alabilir ve konuya odaklanabilirsin:
       "Merhaba! Size zihinsel ve ruhsal gelişim konularında nasıl yardımcı olabilirim?"
    
    ## YANITLAMA YAKLAŞIMI
    
    Cevaplarını empatik, ilham verici ve destekleyici bir tonda ver. Kullanıcının 
    sorusunu anlamaya çalış ve kişisel gelişimini destekleyecek yönlendirmeler yap.
    
    1. Yanıtlarını HTML formatında oluştur
    2. <ul> ve <li> kullan
    3. Satır atlamak için <br> kullan
    4. Kalın yazı için <strong> kullan
    5. Markdown işaretleri (*, -) kullanma
    
    ## DİĞER MODÜLLERE YÖNLENDİRME
    
    Eğer kullanıcı sana Gayrimenkul (emlak piyasası, mevzuat, satılık/kiralık ilanlar, 
    gayrimenkul trendleri, inşaat) veya Finans (borsa, hisse senetleri, teknik/temel 
    analiz, kripto paralar, faiz, tahviller, emtia, döviz piyasası, makro/mikro ekonomi) 
    konularında bir soru sorarsa, kullanıcıyı ilgili GPT modülüne yönlendir.
    """,
    
    "finance": """
    # Finans GPT - Ana Görev ve Rol Tanımı
    
    Sen SibelGPT'sin: İstanbul Borsası, hisse senetleri, teknik ve temel analiz, kripto paralar, 
    faiz, tahviller, emtia piyasası, döviz piyasası, pariteler, makro ve mikro ekonomi
    konularında uzmanlaşmış, Türkçe yanıt veren bir yapay zeka finans danışmanısın.
    
    ## TEMEL KURALLAR - ÇOK ÖNEMLİ
    
    1. **SADECE AŞAĞIDAKİ KONULARDA CEVAP VER**:
       - Borsa, hisse senetleri, teknik ve temel analiz
       - Kripto paralar ve blockchain teknolojisi
       - Faiz ve tahvil piyasaları
       - Emtia piyasaları (altın, gümüş vb.)
       - Döviz piyasaları ve pariteler
       - Makro ve mikro ekonomi konuları
       - Yatırım stratejileri ve portföy yönetimi
       - Ekonomik göstergeler ve analizler
    
    2. **DİĞER TÜM KONULARDA ŞÖYLE YANIT VER**:
       "Bu soru Finans GPT'nin uzmanlık alanı dışındadır. Ben sadece borsa, yatırım, 
       ekonomi, kripto para ve finans konularında yardımcı olabilirim. Bu alanlarla 
       ilgili bir sorunuz varsa memnuniyetle cevaplayabilirim."
    
    3. **SADECE AŞAĞIDAKİ SELAMLAŞMA VE SOHBET BAŞLANGICI MESAJLARINA NORMAL CEVAP VER**:
       - Selamlaşma: "merhaba", "selam", "hello", "hi", "günaydın", "iyi günler", "iyi akşamlar"
       - Hal hatır: "nasılsın", "naber", "ne haber", "iyi misin"
       
       Bu durumda kısaca selamı alabilir ve konuya odaklanabilirsin:
       "Merhaba! Size finans ve yatırım konularında nasıl yardımcı olabilirim?"
    
    ## YANITLAMA YAKLAŞIMI
    
    Cevaplarını net, anlaşılır ve profesyonel bir tonda ver, ancak teknik konuları
    basitleştirerek anlat. Yatırım tavsiyesi verirken riskleri de belirt.
    
    1. Yanıtlarını HTML formatında oluştur
    2. <ul> ve <li> kullan
    3. Satır atlamak için <br> kullan
    4. Kalın yazı için <strong> kullan
    5. Markdown işaretleri (*, -) kullanma
    
    ## ÖNEMLİ UYARILAR
    
    Finans önerilerinde mutlaka şu uyarıyı ekle:
    
    <div style="background:#fff3e0;padding:10px;border-left:5px solid #ff9800;margin:10px 0;">
      <strong>⚠️ Risk Uyarısı:</strong> Burada sunulan bilgiler yatırım tavsiyesi değildir. 
      Tüm yatırım ve finansal kararlar kendi sorumluluğunuzdadır. Yatırım yapmadan önce 
      profesyonel danışmanlık almanız önerilir.
    </div>
    
    ## DİĞER MODÜLLERE YÖNLENDİRME
    
    Eğer kullanıcı sana Gayrimenkul (emlak piyasası, mevzuat, satılık/kiralık ilanlar, 
    gayrimenkul trendleri, inşaat) veya Zihin Koçu (numeroloji, astroloji, kadim bilgiler, 
    psikoloji, ruh sağlığı, thetahealing, motivasyon, kişisel gelişim) konularında 
    bir soru sorarsa, kullanıcıyı ilgili GPT modülüne yönlendir.
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

# ── Konu Tespiti ─────────────────────────────────────────
async def detect_topic(question: str, mode: str = None) -> str:
    """Kullanıcının sorusunun hangi alana ait olduğunu tespit eder."""
    
    selamlasma_kaliplari = [
        "merhaba", "selam", "hello", "hi", "hey", "günaydın", "iyi günler", 
        "iyi akşamlar", "nasılsın", "naber", "ne haber", "hoş geldin", "nasıl gidiyor"
    ]
    
    clean_question = question.lower()
    for char in ".,;:!?-_()[]{}\"'":
        clean_question = clean_question.replace(char, " ")
    
    if len(clean_question.split()) <= 3:
        for kalip in selamlasma_kaliplari:
            if kalip in clean_question:
                print(f"✓ Selamlaşma mesajı tespit edildi, mevcut modda kalınıyor: {kalip}")
                return mode if mode else "real-estate"
    
    topics = {
        "real-estate": [
            "emlak", "gayrimenkul", "ev", "daire", "konut", "kiralık", "satılık", 
            "tapu", "mortgage", "ipotek", "kredi", "remax", "metrekare", "imar", 
            "arsa", "bina", "kat", "müstakil", "dükkan", "ofis", "iş yeri", "bahçe",
            "balkon", "oda", "salon", "banyo", "mutfak", "yapı", "inşaat", "tadilat"
        ],
        "mind-coach": [
            "numeroloji", "astroloji", "astrolojik", "astrologya", "burç", "burcum", "yıldız", 
            "gezegen", "ay", "güneş", "mars", "venüs", "aslan", "kova", "koç", "balık", 
            "meditasyon", "reiki", "terapi", "psikoloji", "depresyon", "anksiyete", 
            "stres", "motivasyon", "gelişim", "spiritüel", "enerji", "şifa", "kadim", 
            "theta", "healing", "ruh", "bilinç", "farkındalık", "arınma", "denge", 
            "uyum", "yoga", "nefes", "horoskop", "yıldızname"
            "yaşam", "ilişki", "başarı", "başarısızlık", "korku", "fobia", "travma", 
            "nlp", "hipnoz", "özgüven", "kendini", "yaşam", "hayat", "amaç", "hedef",
            "duygusal", "zeka", "sosyal", "beceri", "liderlik", "iletişim"
        ],
        "finance": [
            "borsa", "hisse", "finans", "yatırım", "faiz", "döviz", "euro", "dolar", 
            "altın", "gümüş", "kripto", "bitcoin", "ethereum", "bist", "ekonomi", 
            "enflasyon", "tahvil", "bono", "portföy", "fon", "kazanç", "kâr", "zarar", 
            "analiz", "teknik", "temel", "parite", "forex", "banka", "para"
            "defi", "nft", "reit", "gayrimenkul", "yatırım", "ortaklığı", 
            "emeklilik", "sigorta", "vergi", "blockchain", "web3", "staking"
        ]
    }
    
    matches = {topic: 0 for topic in topics}
    
    for topic, keywords in topics.items():
        for keyword in keywords:
            if keyword in clean_question:
                matches[topic] += 1
    
    max_matches = max(matches.values()) if matches else 0
    
    if max_matches <= 1:
        if len(clean_question.split()) <= 5:
            print(f"✓ Kısa genel mesaj tespit edildi, mevcut modda kalınıyor")
            return mode if mode else "real-estate"
            
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
                return mode if mode else "real-estate"
                
            for topic_key in topics.keys():
                if topic_key in detected_topic_by_gpt:
                    return topic_key
            
            return mode if mode else "real-estate"
            
        except Exception as e:
            print(f"⚠️ Konu tespiti hatası (OpenAI API): {e}")
            return mode if mode else "real-estate"
    
    for topic, count in matches.items():
        if count == max_matches:
            return topic
    
    return mode if mode else "real-estate"

# ── Yeni İyileştirme Fonksiyonları ─────────────────────────
async def check_if_property_listing_query(question: str) -> bool:
    """Sorunun gayrimenkul ile ilgili olup ilan araması gerektirip gerektirmediğini tespit eder"""
    try:
        resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """
                    Bu soruyu analiz et ve sadece "Evet" veya "Hayır" yanıtı ver.
                    
                    SORU TİPLERİ:
                    
                    1. İLAN ARAMASI GEREKTİREN SORULAR (Evet):
                       - "Kadıköy'de satılık daire bul"
                       - "20 milyona kadar 3+1 daire arıyorum"
                       - "Beşiktaş'ta ev var mı?"
                       - "Maltepe'de villa göster"
                       - "Hangi bölgede ucuz ev var?"
                    
                    2. İLAN ARAMASI GEREKTİRMEYEN SORULAR (Hayır):
                       - "Ev alırken nelere dikkat etmeliyim?"
                       - "Konut kredisi nasıl alınır?"
                       - "Tapu işlemleri nasıl yapılır?"
                       - "Emlak vergisi ne kadar?"
                       - "Gayrimenkul piyasası nasıl?"
                       - "Hangi bölge yatırım için iyi?"
                    
                    Sadece "Evet" veya "Hayır" yanıtı ver.
                    """
                },
                {"role": "user", "content": question}
            ],
            temperature=0.1,
            max_tokens=10
        )
        
        answer = resp.choices[0].message.content.strip().lower()
        is_listing_query = "evet" in answer
        print(f"📊 İlan araması tespiti: {answer} → {is_listing_query}")
        return is_listing_query
        
    except Exception as e:
        print(f"❌ İlan araması tespiti hatası: {e}")
        # Hata durumunda güvenli mod - eski sistemle devam et
        return property_search_handler.is_property_search_query(question)

async def check_if_real_estate_query(question: str) -> bool:
    """GPT kullanarak sorunun gerçekten gayrimenkul ile ilgili olup olmadığını tespit eder"""
    try:
        resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": """
                    Bu bir soru sınıflandırma görevidir. Verilen soruyu analiz ederek, gayrimenkul/emlak 
                    konusuyla doğrudan ilgili olup olmadığını belirle. Sadece "Evet" veya "Hayır" yanıtı ver.
                    
                    Gayrimenkul ile ilgili konular:
                    - Ev, daire, konut, arsa alım-satımı
                    - Kiralama, emlak piyasası
                    - Tapu, ipotek, mortgage işlemleri
                    - Müteahhit, inşaat, tadilat konuları
                    - Oda sayısı, metrekare, site, bahçe gibi özellikler
                    - Emlak vergisi, komisyon
                    - Konut kredisi, faiz oranları (gayrimenkul bağlamında)
                    
                    Gayrimenkul ile ilgili OLMAYAN konular (örnekler):
                    - "Kiralık katil" (emlak kiralama değil)
                    - "Kat" kelimesi geçen ama gayrimenkul olmayan sorular
                    - "Oda" kelimesi geçen ama ev odası olmayan konular
                    - Astronomi, tarih, spor, bilim, genel kültür soruları
                    - Günlük hayat, kişisel sorular
                    
                    Sadece "Evet" veya "Hayır" yanıtı ver, başka açıklama yapma.
                    """
                },
                {"role": "user", "content": question}
            ],
            temperature=0.1,
            max_tokens=10
        )
        
        answer = resp.choices[0].message.content.strip().lower()
        is_real_estate = "evet" in answer
        print(f"📊 GPT gayrimenkul ilgi tespiti: {answer} → {is_real_estate}")
        return is_real_estate
        
    except Exception as e:
        print(f"❌ Gayrimenkul ilgi tespiti hatası: {e}")
        # Hata durumunda güvenli mod - normal işleme devam et
        return True

def get_out_of_scope_response(mode: str) -> str:
    """Uzmanlık alanı dışı sorular için yanıt oluşturur"""
    responses = {
        "real-estate": """
        <h3>🏠 Gayrimenkul GPT Uzmanlık Alanı</h3>
        <p>Bu soru Gayrimenkul GPT'nin uzmanlık alanı dışındadır. Ben sadece gayrimenkul, 
        emlak ve konut konularında yardımcı olabilirim.</p>
        
        <h4>Size yardımcı olabileceğim konular:</h4>
        <ul>
            <li><strong>Emlak Alım-Satım:</strong> Ev, daire, villa, arsa işlemleri</li>
            <li><strong>Kiralama:</strong> Kiralık konut arama ve sözleşme süreçleri</li>
            <li><strong>Yatırım:</strong> Gayrimenkul yatırımı ve değerlendirme</li>
            <li><strong>Finansman:</strong> Konut kredisi, mortgage işlemleri</li>
            <li><strong>Yasal Süreçler:</strong> Tapu, noter, emlak vergisi</li>
            <li><strong>İnşaat:</strong> Yapı denetim, tadilat, dekorasyon</li>
        </ul>
        
        <p>Bu alanlarla ilgili bir sorunuz varsa memnuniyetle cevaplayabilirim!</p>
        """,
        
        "mind-coach": """
        <h3>🧠 Zihin Koçu GPT Uzmanlık Alanı</h3>
        <p>Bu soru Zihin Koçu GPT'nin uzmanlık alanı dışındadır. Ben sadece kişisel gelişim, 
        psikoloji, numeroloji, astroloji ve spiritüel konularda yardımcı olabilirim.</p>
        
        <h4>Size yardımcı olabileceğim konular:</h4>
        <ul>
            <li><strong>Numeroloji:</strong> İsim ve doğum tarihi analizleri</li>
            <li><strong>Astroloji:</strong> Burç yorumları ve gezegen etkileri</li>
            <li><strong>Kişisel Gelişim:</strong> Motivasyon ve öz güven</li>
            <li><strong>Ruh Sağlığı:</strong> Stres yönetimi, rahatlama teknikleri</li>
            <li><strong>Thetahealing:</strong> Enerji çalışmaları ve şifa</li>
            <li><strong>Meditasyon:</strong> Bilinçaltı ve farkındalık</li>
        </ul>
        
        <p>Bu alanlarla ilgili bir sorunuz varsa memnuniyetle cevaplayabilirim!</p>
        """,
        
        "finance": """
        <h3>💰 Finans GPT Uzmanlık Alanı</h3>
        <p>Bu soru Finans GPT'nin uzmanlık alanı dışındadır. Ben sadece borsa, yatırım, 
        ekonomi, kripto para ve finans konularında yardımcı olabilirim.</p>
        
        <h4>Size yardımcı olabileceğim konular:</h4>
        <ul>
            <li><strong>Borsa:</strong> Hisse senetleri, BIST analizleri</li>
            <li><strong>Teknik Analiz:</strong> Grafik okuma, göstergeler</li>
            <li><strong>Temel Analiz:</strong> Şirket değerlendirme</li>
            <li><strong>Kripto Para:</strong> Bitcoin, Ethereum, altcoin'ler</li>
            <li><strong>Döviz:</strong> EUR/TRY, USD/TRY pariteler</li>
            <li><strong>Emtia:</strong> Altın, gümüş, petrol</li>
            <li><strong>Ekonomi:</strong> Makro/mikro ekonomik analizler</li>
        </ul>
        
        <p>Bu alanlarla ilgili bir sorunuz varsa memnuniyetle cevaplayabilirim!</p>
        """
    }
    
    return responses.get(mode, responses["real-estate"])

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
         print("⚠️ Query embedding boş, arama yapılamıyor!")
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

        # Ham yanıtı logla
        print(f"🔮 Supabase RPC yanıtı: {type(response)}")
        
        all_results = response.data if hasattr(response, "data") and response.data is not None else [] 
        # Alan adlarını düzelt (ilan_no -> ilan_id)
        for r in all_results:
            if isinstance(r, dict) and 'ilan_no' in r and 'ilan_id' not in r:
                r['ilan_id'] = r['ilan_no']  # ilan_no'yu ilan_id olarak kopyala

        # İlk sonuçta hangi alanların olduğunu kontrol et
        if all_results and len(all_results) > 0:
            first_result = all_results[0]
            print(f"📋 İlk sonuç tüm alanlar: {first_result.keys() if isinstance(first_result, dict) else 'dict değil'}")
            print(f"📋 İlk sonuç içeriği: {first_result}")
            # İlan ID kontrolü
            ilan_id = first_result.get('ilan_id') if isinstance(first_result, dict) else None
            print(f"📋 İlk sonuç ilan_id: {ilan_id}")

        # Filtreleme yaparken alanların varlığını kontrol et
        valid_results = []
        for i, r in enumerate(all_results[:10]):  # İlk 10 sonucu göster
            print(f"📌 Sonuç #{i}: Tüm alanlar - {r.keys() if isinstance(r, dict) else 'dict değil'}")
            similarity = r.get('similarity', 0) if isinstance(r, dict) else 0
            print(f"📌 Sonuç #{i}: Similarity - {similarity}")
            ilan_id = r.get('ilan_id') if isinstance(r, dict) else None
            print(f"📌 Sonuç #{i}: ilan_id - {ilan_id}")
            
            if isinstance(r, dict) and r.get('similarity', 0) > MATCH_THRESHOLD:
                valid_results.append(r)
                
            print(f"✅ İlanlar sorgulandı: Toplam {len(valid_results)} gerçek ilişkili ilan bulundu")  

         # Geçerli sonuçlardaki ilan_id'leri kontrol et
        if valid_results:
            valid_ids = [r.get('ilan_id') for r in valid_results if r.get('ilan_id')]
            print(f"🏷️ Geçerli ilan ID'leri: {valid_ids[:5]}... (ilk 5)")
        
        if not valid_results:
            print("⚠️ Hiç ilan bulunamadı!")
        
        return valid_results
        
    except Exception as exc:
        print(f"❌ Arama işleminde hata: {exc}")
        import traceback
        print(f"🔥 Hata detayı: {traceback.format_exc()}")
        return []
        
# ── Formatlama Fonksiyonu ─────────────────────────────────
def format_context_for_sibelgpt(listings: List[Dict]) -> str:
    if not listings:
        return "🔍 Uygun ilan bulunamadı. Lütfen farklı arama kriterleri deneyin."

    try:
        locale.setlocale(locale.LC_ALL, 'tr_TR.UTF-8')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, 'tr_TR')
        except locale.Error:
            pass # Locale ayarlanamazsa devam et

    MAX_LISTINGS_TO_SHOW = 20
    listings_to_format = listings[:MAX_LISTINGS_TO_SHOW]
    if not listings_to_format:
        return "🔍 Belirtilen kriterlere uygun ilan bulunamadı. Lütfen aramanızı genişletin."
   
    final_output = "<p><strong>📞 Sorgunuzla ilgili ilanlar burada listelenmiştir. Detaylı bilgi için 532 687 84 64 numaralı telefonu arayabilirsiniz.</strong></p>"
   
    formatted_parts = []
    for i, l_item in enumerate(listings_to_format, start=1):
        ilan_no = l_item.get('ilan_id', l_item.get('ilan_no', str(i)))
        baslik = l_item.get('baslik', '(başlık yok)')
        lokasyon = l_item.get('lokasyon', '?')
        
        fiyat = "?"
        fiyat_raw = l_item.get('fiyat')
        if fiyat_raw is not None:
            try:
                # Fiyat string'ini temizleyip float'a çevirme
                fiyat_str_cleaned = str(fiyat_raw).replace('.', '').replace(',', '.')
                if fiyat_str_cleaned.replace('.', '', 1).isdigit():
                    fiyat_num = float(fiyat_str_cleaned)
                    fiyat = f"{fiyat_num:,.0f} ₺".replace(',', '#').replace('.', ',').replace('#', '.')
                else:
                    fiyat = str(fiyat_raw)
            except ValueError:
                fiyat = str(fiyat_raw)
            except Exception:
                 fiyat = str(fiyat_raw)
       
        ozellikler_liste = []
        oda_sayisi = l_item.get('oda_sayisi', '')
        if oda_sayisi:
            ozellikler_liste.append(oda_sayisi)
       
        metrekare = l_item.get('metrekare', '')
        if metrekare:
            metrekare_str = str(metrekare).strip()
            if not metrekare_str.endswith("m²"):
                 ozellikler_liste.append(f"{metrekare_str} m²")
            else:
                 ozellikler_liste.append(metrekare_str)

        bulundugu_kat_raw = l_item.get('bulundugu_kat')
        if bulundugu_kat_raw is not None and str(bulundugu_kat_raw).strip() != '':
            bulundugu_kat_str = str(bulundugu_kat_raw).strip()
            try:
                if bulundugu_kat_str.replace('-', '', 1).isdigit():
                    kat_no = int(bulundugu_kat_str)
                    if kat_no == 0:
                        ozellikler_liste.append("Giriş Kat")
                    elif kat_no < 0:
                        ozellikler_liste.append(f"Bodrum Kat ({kat_no})")
                    else:
                        ozellikler_liste.append(f"{kat_no}. Kat")
                else:
                    ozellikler_liste.append(bulundugu_kat_str)
            except ValueError:
                ozellikler_liste.append(bulundugu_kat_str)
       
        ozellikler_db = l_item.get('ozellikler')
        if ozellikler_db and isinstance(ozellikler_db, str):
            ozellikler_parts_raw = ozellikler_db.split('|')
            ozellikler_parts_processed = []
            for part_raw in ozellikler_parts_raw:
                part = part_raw.strip()
                if re.match(r'^-?\d+$', part):
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
        elif ozellikler_liste:
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
    real_ids = [l_item.get('ilan_id') for l_item in listings_to_format if l_item.get('ilan_id')]
    print(f"🏷️ İlan Veritabanındaki Gerçek İlan Numaraları: {real_ids}")
    if real_ids:
        final_output += f"<p><strong>VERİTABANINDAKİ GERÇEK İLAN NUMARALARI: {', '.join(real_ids)}</strong></p>"
    final_output += "<p>Bu ilanların doğruluğunu kontrol ettim. Farklı bir arama yapmak isterseniz, lütfen kriterleri belirtiniz.</p>"
   
    return final_output

# ── Ana Fonksiyon - YENİLENMİŞ VE İYİLEŞTİRİLMİŞ ─────────
async def answer_question(question: str, mode: str = "real-estate", conversation_history: List = None) -> str:
    """
    İyileştirilmiş Ana Fonksiyon:
    1. Konu tespiti yapar
    2. Uzmanlık alanı kontrolü yapar
    3. Gayrimenkul modunda akıllı ilan araması yapar
    4. Gereksiz veritabanı sorgularını önler
    """
    
    print(f"🚀 İYİLEŞTİRİLMİŞ SORGU BAŞLADI - Soru: {question[:50]}..., Mod: {mode}")
    
    # SELAMLAŞMA KONTROLÜ - ÖNCELİKLİ!
    # Selamlaşma kalıplarını kontrole al
    selamlasma_kaliplari = [
        "merhaba", "selam", "hello", "hi", "hey", "günaydın", "iyi günler", 
        "iyi akşamlar", "nasılsın", "naber", "ne haber", "hoş geldin", "nasıl gidiyor"
    ]
    
    # Soru basit bir selamlaşma mı kontrol et
    clean_question = question.lower().strip()
    is_greeting = False
    
    for kalip in selamlasma_kaliplari:
        if kalip in clean_question:
            is_greeting = True
            print(f"✓ Selamlaşma mesajı tespit edildi: {kalip}")
            break
    
    # Eğer selamlaşma ise, doğrudan yanıt ver
    if is_greeting:
        print("🤝 Selamlaşmaya doğrudan yanıt veriliyor")
        
        greeting_responses = {
            "real-estate": f"Merhaba! Size gayrimenkul konusunda nasıl yardımcı olabilirim?",
            "mind-coach": f"Merhaba! Size zihinsel ve ruhsal gelişim konularında nasıl yardımcı olabilirim?",
            "finance": f"Merhaba! Size finans ve yatırım konularında nasıl yardımcı olabilirim?"
        }
        
        return greeting_responses.get(mode, greeting_responses["real-estate"])
    
    # 1. KONU TESPİTİ
    detected_topic = await detect_topic(question, mode)
    print(f"📊 Tespit edilen konu: {detected_topic}, Kullanıcının seçtiği mod: {mode}")
    
    # Diğer kodlar aynı kalsın...
    
    # 2. FARKLI KONU İSE YÖNLENDİR
    if detected_topic != mode:
        if detected_topic in ["real-estate", "mind-coach", "finance"]:
            redirection_key = f"{mode}-to-{detected_topic}"
            if redirection_key in REDIRECTION_MESSAGES:
                print(f"↪️ Farklı modüle yönlendiriliyor: {redirection_key}")
                return REDIRECTION_MESSAGES[redirection_key]
        
        # Genel konu ise uzmanlık alanı dışı yanıt ver
        print(f"⚠️ Genel konu tespit edildi, uzmanlık alanı dışı yanıt veriliyor")
        return get_out_of_scope_response(mode)
    
              
    # 4. İÇERİK HAZIRLAMA - AKILLI ARAMA
    context = ""
    if mode == "real-estate":
        # ✅ OPTİMİZE EDİLMİŞ AKILLI ARAMA
        is_listing_query = await check_if_property_listing_query(question)
    
        if is_listing_query:
            print("🏠 İlan araması tespit edildi - Cache'li hızlı arama kullanılıyor")
            try:
                context = await property_search_handler.search_properties(question)
                print(f"✅ İlan araması tamamlandı: {len(context)} karakter")
            except Exception as e:
                print(f"❌ İlan araması hatası: {e}")
                context = "İlan araması sırasında teknik sorun oluştu."
        else:
                print("📚 Gayrimenkul genel bilgi sorusu - VERİTABANI ATLANYOR")
                context = "Bu soru için ilan araması gerekmemektedir."
    
    # 5. SYSTEM PROMPT VE MESAJLARI HAZIRLA
    system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["real-estate"])
    
    messages = [
        {"role": "system", "content": f"{system_prompt}\n\nİLGİLİ İLANLAR:\n{context if context else 'Bu soru için ilan araması gerekmemektedir.'}\n"}
    ]
    
    # 6. KONUŞMA GEÇMİŞİ EKLE
    if conversation_history and len(conversation_history) > 0:
        for msg in conversation_history:
            if isinstance(msg, dict) and 'role' in msg and 'text' in msg:
                messages.append({"role": msg['role'], "content": msg['text']})
    
    # 7. KULLANICI SORUSU EKLE
    messages.append({"role": "user", "content": question})
    
    # 8. YANIT AL VE DÖNDÜR
    try:
        print("🤖 OpenAI API'ye istek gönderiliyor...")
        # 🚀 AKILLI MODEL SEÇİMİ - İlan araması için hızlı model
        selected_model = "gpt-3.5-turbo" if (mode == "real-estate" and "Bu soru için ilan araması gerekmemektedir." in context) else "gpt-4o-mini"
        print(f"🤖 Kullanılan model: {selected_model}")
        # 🌡️ AKILLI TEMPERATURE SEÇİMİ
        if mode == "real-estate" and "Bu soru için ilan araması gerekmemektedir." not in context:
            temp = 0.3  # İlan araması - tutarlı format
        else:
            temp = 0.6  # Genel sorular - yaratıcı yanıtlar
        print(f"🌡️ Kullanılan temperature: {temp}")
        resp = await openai_client.chat.completions.create(
            model=selected_model,
            messages=messages,
            temperature=temp,
            max_tokens=4096
        )
        
        answer = resp.choices[0].message.content.strip()
        print(f"✅ İYİLEŞTİRİLMİŞ YANIT HAZIR - Uzunluk: {len(answer)} karakter")
        return answer
        
    except Exception as exc:
        print(f"❌ Chat yanıt hatası: {exc}")
        return "Üzgünüm, isteğinizi işlerken beklenmedik bir sorun oluştu. Lütfen daha sonra tekrar deneyin."

