import os
import asyncio 
import locale
import re
from typing import List, Dict, Optional
from openai import AsyncOpenAI
import property_search_handler

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
       - Astroloji, burçlar ve gezegen yorumları
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


async def detect_topic(question: str, mode: str = None) -> str:
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
                return mode if mode else "real-estate"
    
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
    real_ids = [l_item.get('ilan_id') for l_item in listings_to_format if l_item.get('ilan_id')]
    print(f"🏷️ İlan Veritabanındaki Gerçek İlan Numaraları: {real_ids}")
    if real_ids:
        final_output += f"<p><strong>VERİTABANINDAKİ GERÇEK İLAN NUMARALARI: {', '.join(real_ids)}</strong></p>"
    final_output += "<p>Bu ilanların doğruluğunu kontrol ettim. Farklı bir arama yapmak isterseniz, lütfen kriterleri belirtiniz.</p>"
   
    return final_output

# ── Ana Fonksiyon ─────────────────────────────────────────
async def answer_question(question: str, mode: str = "real-estate", conversation_history: List = None) -> str:
    """Kullanıcının sorusuna yanıt verir ve gerektiğinde başka modüle yönlendirir."""
    
    print(f"↪ Soru: {question}, Mod: {mode}")
   
    detected_topic_result = await detect_topic(question, mode)
    print(f"✓ Tespit edilen konu: {detected_topic_result}, Kullanıcının seçtiği mod: {mode}")
   
    if detected_topic_result != mode:
        redirection_key = f"{mode}-to-{detected_topic_result}"
        print(f"⟹ Yönlendirme anahtarı: {redirection_key}")
        
        if redirection_key in REDIRECTION_MESSAGES:
            return REDIRECTION_MESSAGES[redirection_key]
   
    context = ""
    if mode == "real-estate":
        # İlan araması olup olmadığını kontrol et
        if property_search_handler.is_property_search_query(question):
            print("📢 İlan araması tespit edildi, yeni arama modülü kullanılıyor...")
            # Yeni arama modülünü kullan
            context = await property_search_handler.search_properties(question)
        else:
            # Eski yöntemi kullan
            print("📢 Normal soru tespit edildi, standart arama kullanılıyor...")
            query_emb = await get_embedding(question)
            if query_emb:
                listings = await search_listings_in_supabase(query_emb)
                context = format_context_for_sibelgpt(listings)
            else:
                context = "<p>Sorunuzu işlerken bir sorun oluştu, lütfen tekrar deneyin veya farklı bir soru sorun.</p>"
   
    system_prompt = SYSTEM_PROMPTS.get(mode, SYSTEM_PROMPTS["real-estate"])
   
    # Mesajları oluştur - sistem mesajını ekle
    messages = [
        {"role": "system", "content": f"{system_prompt}<br><br>İLGİLİ İLANLAR:<br>{context if context else 'Uygun ilan bulunamadı veya bu mod için ilan aranmıyor.'}<br><br>Bu HTML formatındaki ilanları OLDUĞU GİBİ kullanıcıya göster, HİÇBİR DEĞİŞİKLİK yapma! Sadece ekle, filtreleme, özetleme veya değiştirme YAPMA! Tüm ilanlar olduğu gibi kullanıcıya gösterilmeli!"}
    ]
    
    # Eğer sohbet geçmişi varsa ekle
    if conversation_history and len(conversation_history) > 0:
        for msg in conversation_history:
            if isinstance(msg, dict) and 'role' in msg and 'text' in msg:
                messages.append({"role": msg['role'], "content": msg['text']})
    
    # Kullanıcının yeni sorusunu ekle
    messages.append({"role": "user", "content": question})

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
        return "Üzgünüm, isteğinizi işlerken beklenmedik bir sorun oluştu. Lütfen daha sonra tekrar deneyin."

# ── Terminalden Test ──────────────────────────────────────
if __name__ == "__main__":
    async def main():
        q = input("Soru: ")
        # Varsayılan mod "real-estate" olarak ayarlandı, test için değiştirilebilir.
        response = await answer_question(q, mode="real-estate", conversation_history=[]) 
        print(response)

    # asyncio.run() Python 3.7+ için daha modern bir yoldur.
    # Eğer Python 3.6 veya daha eski bir sürüm kullanılıyorsa loop.run_until_complete() gerekir.
    # Ancak kodda AsyncOpenAI kullanıldığı için Python 3.7+ varsayılabilir.
    asyncio.run(main())
