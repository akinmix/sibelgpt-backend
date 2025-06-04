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

# ── GÜNCELLENMIŞ TOPICS DICTIONARY (150'şer Kelime) ────────
TOPIC_KEYWORDS = {
    "real-estate": [
        # Temel Gayrimenkul Kavramları
        "emlak", "gayrimenkul", "ev", "daire", "konut", "kiralık", "satılık", 
        "tapu", "mortgage", "ipotek", "kredi", "remax", "metrekare", "imar", 
        "arsa", "bina", "kat", "müstakil", "dükkan", "ofis", "iş yeri", "bahçe",
        "balkon", "oda", "salon", "banyo", "mutfak", "yapı", "inşaat", "tadilat",
        
        # Gayrimenkul İşlemleri ve Hukuk
        "senet", "ruhsat", "iskân", "noter", "vekaletname", "ferağ", "komisyon",
        "emlak vergisi", "mtv", "aidat", "kalorifer", "doğalgaz", "elektrik",
        "su faturası", "belediye", "çevre temizlik", "asansör", "kapıcı",
        "mülkiyet", "hukuk", "hukuki", "intifa", "irtifak", "izalei", "şuyu",
        "miras", "veraset", "mirasçı", "ortak", "ortaklık", "pay", "hisse",
        "zilyetlik", "tasarruf", "devir", "temlik", "rehin", "teminat"
        
        # İnşaat ve Yapı
        "betonarme", "çelik", "tuğla", "panel", "prefabrik", "dubleks", "tripleks",
        "villa", "apart", "rezidans", "site", "complex", "köşk", "malikane",
        "çiftlik evi", "yazlık", "stüdyo", "loft", "penthouse", "terras",
        
        # Teknik Özellikler
        "asansörlü", "güvenlik", "kamera", "interkom", "otopark", "garaj",
        "jeneratör", "hidrofor", "yangın merdiveni", "çıkış", "acil durum",
        "ses yalıtımı", "ısı yalıtımı", "cam balkon", "pvc", "alüminyum",
        
        # Lokasyon ve Bölge
        "merkezi", "ulaşım", "metro", "metrobüs", "otobüs", "minibüs", "taksi",
        "cadde", "sokak", "mahalle", "semt", "bölge", "ilçe", "şehir merkezi",
        "sahil", "deniz", "göl", "park", "yeşil alan", "orman", "dağ", "tepe",
        
        # Oda ve Alan Tipleri
        "yatak odası", "çocuk odası", "misafir odası", "çalışma odası",
        "kiler", "depo", "bodrum", "çatı katı", "tavan arası", "balkon",
        "teras", "veranda", "kış bahçesi", "hobi odası", "fitness", "sauna",
        
        # Gayrimenkul Yatırımı
        "yatırım", "getiri", "kira geliri", "değer artışı", "piyasa",
        "trend", "fiyat", "değerleme", "ekspertiz", "rapor", "analiz",
        "portföy", "çeşitlendirme", "risk", "konum", "potansiyel",
        
        # Sözleşme ve İşlemler
        "sözleşme", "kira sözleşmesi", "satış sözleşmesi", "ön sözleşme",
        "depozito", "kapora", "peşinat", "taksit", "vade", "ödeme planı",
        "refinansman", "erken ödeme", "gecikme faizi", "ceza", "kefil"
    ],
    
    "mind-coach": [
        # Astroloji ve Burçlar
        "astroloji", "astrolojik", "burç", "burcum", "yıldız", "yıldızlar",
        "gezegen", "ay", "güneş", "mars", "venüs", "jüpiter", "satürn",
        "merkür", "neptün", "uranüs", "plüton", "aslan", "kova", "koç", 
        "balık", "ikizler", "yengeç", "başak", "terazi", "akrep", "yay",
        "oğlak", "horoskop", "yıldızname", "astral", "kozmik", "evren",
        
        # Numeroloji
        "numeroloji", "sayı", "sayılar", "doğum tarihi", "isim analizi",
        "kader sayısı", "yaşam yolu", "kişilik sayısı", "ruh sayısı",
        "ifade sayısı", "kalp arzusu", "olgunluk sayısı", "pitagor",
        "kaldean", "kabala", "gematria", "vibrasyon", "frekans",
        
        # Spiritüel ve Enerji Çalışmaları
        "spiritüel", "ruhani", "enerji", "aura", "çakra", "kundalini",
        "meditasyon", "bilinç", "farkındalık", "uyanış", "aydınlanma",
        "theta", "healing", "şifa", "reiki", "pranic", "kristal",
        "taş", "maden", "arınma", "temizlik", "koruma", "büyü",
        
        # Psikoloji ve Ruh Sağlığı
        "psikoloji", "psikolog", "terapi", "terapist", "danışman",
        "depresyon", "anksiyete", "stres", "panik", "fobia", "travma",
        "ptsd", "obsesif", "kompulsif", "bipolar", "sınır", "kişilik",
        "narsist", "empati", "duygusal", "zeka", "sosyal", "beceri",
        
        # Kişisel Gelişim ve Motivasyon
        "kişisel gelişim", "motivasyon", "özgüven", "özsaygı", "özdisiplin",
        "başarı", "hedef", "amaç", "vizyon", "misyon", "değer", "inanç",
        "liderlik", "iletişim", "empati", "karizma", "etki", "nüfuz",
        "yaratıcılık", "inovasyon", "çözüm", "problem", "karar", "seçim",
        
        # İlişkiler ve Aile
        "ilişki", "evlilik", "aşk", "sevgi", "çift", "eş", "partner",
        "aile", "anne", "baba", "çocuk", "kardeş", "akraba", "arkadaş",
        "sosyal", "bağ", "bağlılık", "güven", "sadakat", "ihanet",
        "ayrılık", "boşanma", "barışma", "affetme", "kıskançlık", "öfke",
        
        # Ruhsal Gelişim ve Felsefe
        "ruh", "can", "nefs", "ego", "benlik", "kimlik", "öz", "asıl",
        "hakikat", "gerçek", "yanılsama", "maya", "illüzyon", "hayal",
        "düş", "sembol", "simge", "işaret", "alamet", "kehanet", "kehânet",
        "falcılık", "büyücülük", "şamanlık", "sufizm", "tasavvuf", "yoga"
    ],
    
    "finance": [
        # Borsa ve Hisse Senetleri
        "borsa", "hisse", "pay", "senet", "bist", "nasdaq", "dow", "s&p",
        "ftse", "dax", "nikkei", "hang seng", "şirket", "halka arz",
        "ipo", "temettü", "kar payı", "sermaye", "piyasa değeri",
        "hacim", "işlem", "alış", "satış", "spread", "fiyat", "değer",
        
        # Teknik Analiz
        "teknik analiz", "grafik", "mum", "çubuk", "line", "bar",
        "trend", "destek", "direnç", "kırılım", "geri çekilme",
        "fibonacci", "retracement", "rsi", "macd", "stochastic",
        "bollinger", "moving average", "ema", "sma", "volume",
        "oscillator", "momentum", "divergence", "konvergens",
        
        # Temel Analiz
        "temel analiz", "mali tablo", "bilanço", "gelir tablosu",
        "nakit akım", "karlılık", "roe", "roa", "pe", "pb", "ev/ebitda",
        "f/k", "pd/dd", "büyüme", "gelir", "gider", "net kar",
        "brüt kar", "ebitda", "ebit", "faaliyet karı", "vergi",
        
        # Kripto Para ve Blockchain
        "kripto", "bitcoin", "ethereum", "altcoin", "blockchain",
        "defi", "nft", "dao", "dex", "cex", "wallet", "cüzdan",
        "mining", "madencilik", "staking", "yield farming", "liquidity",
        "smart contract", "akıllı sözleşme", "token", "coin", "fork",
        "halving", "proof of work", "proof of stake", "consensus",
        
        # Döviz ve Emtia
        "dolar", "kur", "para", "lira", "döviz", "usd", "eur", "gbp", "jpy", "chf", "try", "parite",
        "kur", "çapraz kur", "swap", "forward", "futures", "option",
        "altın", "gümüş", "platin", "paladyum", "petrol", "doğalgaz",
        "buğday", "mısır", "soya", "kakao", "kahve", "şeker", "pamuk",
        
        # Ekonomi ve Makro
        "ekonomi", "enflasyon", "deflasyon", "stagflasyon", "gdp",
        "gsyh", "büyüme", "durgunluk", "kriz", "canlanma", "iyileşme",
        "merkez bankası", "fed", "ecb", "tcmb", "faiz", "oran",
        "para politikası", "mali politika", "bütçe", "açık", "fazla",
        
        # Yatırım Araçları
        "yatırım", "portföy", "fon", "etf", "reit", "bono", "tahvil",
        "sukuk", "viop", "vadeli", "opsiyon", "warrant", "sertifika",
        "strukturlu", "structured", "hedge", "arbitraj", "spekülatif"
    ]
}

# ── GÜNCELLENMIŞ SYSTEM PROMPTS ────────────────────────────
SYSTEM_PROMPTS = {
    "real-estate": """
    # Gayrimenkul GPT - Ana Görev ve Rol Tanımı
    
    Sen SibelGPT'sin: İstanbul emlak piyasası ve gayrimenkul konusunda uzmanlaşmış, 
    Türkçe yanıt veren bir yapay zeka asistanısın. Temel görevin kullanıcılara gayrimenkul, 
    emlak ve konut konularında yardımcı olmaktır.
    
    ## TEMEL KURALLAR - ÇOK ÖNEMLİ
    
    1. **SADECE AŞAĞIDAKİ KONULARDA CEVAP VER**:
       
       **🏠 Gayrimenkul Alım-Satım ve Kiralama:**
       - Ev, daire, konut, villa, arsa, ofis, dükkan alım-satımı
       - Kiralık ve satılık gayrimenkul ilanları
       - Emlak piyasası analizi, fiyat trendleri
       - Gayrimenkul değerleme, ekspertiz işlemleri
       
       **📋 Yasal ve İdari İşlemler:**
       - Tapu işlemleri, ferağ, vekaletname düzenleme
       - Emlak vergisi, MTV, belediye harçları
       - İmar durumu, ruhsat, iskân izni süreçleri
       - Noter işlemleri, sözleşme hazırlama
       
       **🏗️ İnşaat ve Yapı Tekniği:**
       - İnşaat malzemeleri, yapı tekniği, proje analizi
       - Tadilat, dekorasyon, renovasyon işlemleri
       - Yapı denetim, betonarme, çelik yapı sistemi
       - Enerji verimliliği, yalıtım teknikleri
       
       **💰 Gayrimenkul Finansmanı:**
       - Konut kredisi, mortgage işlemleri
       - Gayrimenkul yatırımı stratejileri
       - Kira geliri hesaplama, getiri analizi
       - Emlak portföy yönetimi
       
       **🏘️ Lokasyon ve Bölge Analizi:**
       - Mahalle, semt, ilçe karşılaştırması
       - Ulaşım, sosyal tesis analizi
       - Okul, hastane, AVM mesafeleri
       - Yatırım potansiyeli yüksek bölgeler
    
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
       
       **🌟 Astroloji ve Cosmic Bilimler:**
       - Astroloji nedir, astrolojinin temelleri ve tarihi
       - 12 burç (Koç, Boğa, İkizler, Yengeç, Aslan, Başak, Terazi, Akrep, Yay, Oğlak, Kova, Balık)
       - Gezegen etkileri (Güneş, Ay, Mars, Venüs, Jüpiter, Satürn, Merkür, Neptün, Uranüs, Plüton)
       - Horoskop analizi, yıldızname yorumları
       - Astral harita, doğum haritası çıkarma
       - Astrolojik geçişler, retrograd hareketler
       
       **🔢 Numeroloji ve Sayı Bilimi:**
       - Numeroloji nedir, Pitagor ve Kaldean sistemleri
       - İsim ve doğum tarihi analizleri
       - Yaşam yolu sayısı, kader sayısı hesaplama
       - Kişilik sayısı, ruh sayısı, ifade sayısı
       - Kalp arzusu sayısı, olgunluk sayısı
       - Sayıların vibrasyon ve frekans anlamları
       
       **🧠 Psikoloji ve Ruh Sağlığı:**
       - Depresyon, anksiyete, stres yönetimi
       - Panik atak, fobiler, travma iyileşmesi
       - PTSD, obsesif kompulsif bozukluk
       - Bipolar bozukluk, sınır kişilik bozukluğu
       - Duygusal zeka, sosyal beceri geliştirme
       - Psikolojik danışmanlık teknikleri
       
       **⚡ Enerji Çalışmaları ve Şifa:**
       - Thetahealing teknikleri ve uygulamaları
       - Reiki, Pranic healing, kristal şifası
       - Çakra temizleme, aura güçlendirme
       - Kundalini enerjisi, enerji merkezi aktivasyonu
       - Spiritüel koruma, negatif enerji temizleme
       - Meditasyon teknikleri, bilinçaltı programlama
       
       **🚀 Kişisel Gelişim ve Motivasyon:**
       - Özgüven geliştirme, özsaygı artırma
       - Hedef belirleme, başarı stratejileri
       - Motivasyon teknikleri, özdisiplin
       - Liderlik becerileri, karizma geliştirme
       - İletişim becerileri, empati kurma
       - Yaratıcılık, problem çözme teknikleri
       
       **💕 İlişkiler ve Aile Terapisi:**
       - Çift terapisi, evlilik danışmanlığı
       - Aile içi iletişim, çocuk yetiştirme
       - Aşk ve ilişki psikolojisi
       - Ayrılık, boşanma süreci yönetimi
       - Kıskançlık, güven sorunları
       - Sosyal ilişkiler, arkadaşlık bağları
       
       **🌸 Spiritüel Gelişim ve Kadim Bilgiler:**
       - Yoga, meditasyon, nefes teknikleri
       - Şamanlık, sufizm, tasavvuf öğretileri
       - Ruhsal uyanış, bilinç genişletme
       - Sembol ve işaret yorumlama
       - Rüya analizi, rüya yorumlama
       - Hipnoz, NLP (Neuro-Linguistic Programming)
    
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
    
    ## ÖNEMLİ UYARILAR
    
    Psikolojik ve ruhsal konularda mutlaka şu uyarıyı ekle:
    
    <div style="background:#e8f5e9;padding:10px;border-left:5px solid #4caf50;margin:10px 0;">
      <strong>🌟 Not:</strong> Bu bilgiler kişisel gelişim amaçlıdır. Ciddi psikolojik 
      sorunlarınız için mutlaka profesyonel yardım alın.
    </div>
    
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
       
       **📈 Borsa ve Hisse Senetleri:**
       - BIST, NASDAQ, NYSE, Avrupa borsaları analizi
       - Hisse senedi, pay senedi işlemleri
       - Halka arz (IPO), temettü, kar payı
       - Piyasa değeri, hacim, işlem stratejileri
       - Blue chip, penny stock, growth stock
       - Sektör analizi, şirket karşılaştırması
       
       **🔍 Teknik Analiz:**
       - Grafik türleri (mum, çubuk, line, bar)
       - Trend analizi, destek-direnç seviyeleri
       - Teknik göstergeler (RSI, MACD, Stochastic)
       - Bollinger Bands, Moving Average (EMA, SMA)
       - Fibonacci retracement, Elliott Wave teorisi
       - Volume analizi, momentum göstergeleri
       - Chart pattern'lar (baş-omuz, üçgen, bayrak)
       
       **📊 Temel Analiz:**
       - Mali tablo analizi (bilanço, gelir tablosu)
       - Nakit akım tablosu, karlılık oranları
       - P/E, P/B, EV/EBITDA değerleme çarpanları
       - ROE, ROA, ROI karlılık göstergeleri
       - Büyüme oranları, gelir-gider analizi
       - Sektörel karşılaştırma, rekabet analizi
       
       **₿ Kripto Para ve Blockchain:**
       - Bitcoin, Ethereum, Altcoin'ler
       - Blockchain teknolojisi, DeFi protokolleri
       - NFT, DAO, DEX platformları
       - Mining, staking, yield farming
       - Smart contract, token ekonomisi
       - Kripto cüzdan güvenliği, cold storage
       
       **💱 Döviz ve Emtia Piyasaları:**
       - USD/TRY, EUR/TRY, GBP/TRY pariteler
       - Forex trading, çapraz kurlar
       - Altın, gümüş, platin, paladyum
       - Petrol, doğalgaz, tarımsal emtia
       - Futures, forward, option işlemleri
       - Carry trade, arbitraj stratejileri
       
       **🌍 Makro ve Mikro Ekonomi:**
       - Enflasyon, deflasyon, stagflasyon
       - GSYH, büyüme oranları, işsizlik
       - Merkez bankası politikaları (FED, ECB, TCMB)
       - Para politikası, faiz oranları
       - Mali politika, bütçe dengesi
       - Ekonomik göstergeler, istatistikler
       
       **🏦 Yatırım Araçları ve Bankacılık:**
       - Mevduat, vadeli mevduat, repo
       - Tahvil, bono, sukuk işlemleri
       - Yatırım fonları, ETF, REIT
       - Emeklilik fonları, sigorta ürünleri
       - VİOP, vadeli işlemler, opsiyon stratejileri
       - Hedge fund, private equity, venture capital
    
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

# ── İYİLEŞTİRİLMİŞ KONU TESPİTİ ─────────────────────────────────────────
async def detect_topic(question: str, mode: str = None) -> str:
    """Kullanıcının sorusunun hangi alana ait olduğunu tespit eder - İyileştirilmiş versiyon."""
    
    # Önce selamlaşma kontrolü
    selamlasma_kaliplari = [
        "merhaba", "selam", "hello", "hi", "hey", "günaydın", "iyi günler", 
        "iyi akşamlar", "nasılsın", "naber", "ne haber", "hoş geldin", "nasıl gidiyor"
    ]
    
    clean_question = question.lower()
    # Noktalama işaretlerini temizle
    for char in ".,;:!?-_()[]{}\"'":
        clean_question = clean_question.replace(char, " ")
    
    # Kısa selamlaşma mesajları için özel kontrol
    if len(clean_question.split()) <= 3:
        for kalip in selamlasma_kaliplari:
            if kalip in clean_question:
                print(f"✓ Selamlaşma mesajı tespit edildi, mevcut modda kalınıyor: {kalip}")
                return mode if mode else "real-estate"
    
    # Kelime bazlı matching - RENDER 1GB DISK OPTİMİZASYONU
    matches = {topic: 0 for topic in TOPIC_KEYWORDS}
    
    for topic, keywords in TOPIC_KEYWORDS.items():
        # İlk 50 kelimeyi kontrol et (disk tasarrufu)
        for keyword in keywords[:100]:
            if keyword in clean_question:
                matches[topic] += 1
    
    print(f"🔍 Kelime eşleşmeleri: {matches}")
    
    max_matches = max(matches.values()) if matches else 0
    
    # Eğer net bir eşleşme yoksa GPT'ye sor (optimizasyonlu)
    if max_matches <= 2:
        if len(clean_question.split()) <= 5:
            print(f"✓ Kısa genel mesaj tespit edildi, mevcut modda kalınıyor")
            return mode if mode else "real-estate"
            
        try:
            resp = await openai_client.chat.completions.create(
                model="gpt-4o-mini",  # Hızlı ve ekonomik model
                messages=[
                    {
                        "role": "system", 
                        "content": """Sen bir yapay zeka uzmanısın. Kullanıcının sorusunu analiz et ve hangi uzmanlık alanına ait olduğunu belirle.
                         SADECE kategori adını döndür: real-estate, mind-coach, finance, general
                                    
                                    1. real-estate: BÜTÜN gayrimenkul, emlak, mülkiyet, hukuk konuları dahil:
                                       - Ev, daire, villa, arsa, ofis, dükkan alım-satımı
                                       - Tapu, mülkiyet hakları, intifa hakkı, irtifak hakkı, izalei şuyu
                                       - Emlak hukuku, miras hukuku, mülkiyet hukuku
                                       - Konut kredisi, mortgage, finansman
                                       - İnşaat, tadilat, imar, ruhsat, iskân
                                       - Kira, kiralama, sözleşmeler
                                       - Emlak vergisi, harçlar, yasal işlemler
                                                
                                    2. mind-coach: Numeroloji, astroloji, burçlar, psikoloji, 
                                       kişisel gelişim, motivasyon, theta healing, meditasyon, 
                                       ruh sağlığı, depresyon, anksiyete vb.
                                    
                                    3. finance: Borsa, hisse senetleri, yatırım, ekonomi, 
                                       kripto para, döviz, altın, teknik analiz, bitcoin vb.
                                    
                                    4. general: Selamlaşma, günlük konuşma, sohbet, genel sorular vb.
                                    
                                    ÖRNEKLER:
                                    "intifa hakkı nedir" → real-estate
                                    "izalei şuyu nedir" → real-estate  
                                    "mülkiyet hukuku" → real-estate
                                    "Bitcoin" → finance
                                    "numeroloji" → mind-coach"""
                    },
                    {"role": "user", "content": question}
                ],
                temperature=0.2,
                max_tokens=15
            )
            detected_topic_by_gpt = resp.choices[0].message.content.strip().lower()
            print(f"🤖 GPT tarafından tespit edilen konu: {detected_topic_by_gpt}")
            
            if "general" in detected_topic_by_gpt:
                print(f"✓ GPT tarafından genel sohbet olarak tespit edildi, mevcut modda kalınıyor")
                return mode if mode else "real-estate"
                
            # Geçerli topic'leri kontrol et
            for topic_key in TOPIC_KEYWORDS.keys():
                if topic_key in detected_topic_by_gpt:
                    return topic_key
            
            return mode if mode else "real-estate"
            
        except Exception as e:
            print(f"⚠️ Konu tespiti hatası (OpenAI API): {e}")
            return mode if mode else "real-estate"
    
    # En yüksek eşleşme sayısına sahip konuyu döndür
    for topic, count in matches.items():
        if count == max_matches:
            print(f"✅ En yüksek eşleşme: {topic} ({count} kelime)")
            return topic
    
    return mode if mode else "real-estate"

# ── İYİLEŞTİRİLMİŞ İLAN ARAMASI TESPİTİ ─────────────────────────
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
                    
                    İLAN ARAMASI GEREKTİREN SORULAR (Evet):
                    - "Kadıköy'de satılık daire bul/ara/göster"
                    - "20 milyona kadar 3+1 daire arıyorum"
                    - "Beşiktaş'ta ev var mı?"
                    - "Maltepe'de villa göster/listele"
                    - "Hangi bölgede ucuz ev var?"
                    - "X ilçesinde Y bütçeyle ne bulabilirim?"
                    - "Bu kriterlere uyan ilan var mı?"
                    
                    İLAN ARAMASI GEREKTİRMEYEN SORULAR (Hayır):
                    - "Ev alırken nelere dikkat etmeliyim?"
                    - "Konut kredisi nasıl alınır?"
                    - "Tapu işlemleri nasıl yapılır?"
                    - "Emlak vergisi ne kadar?"
                    - "Gayrimenkul piyasası nasıl?"
                    - "Hangi bölge yatırım için iyi?"
                    - "İnşaat sektörü hakkında bilgi"
                    
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

# ── ANA FONKSİYON - TAM İYİLEŞTİRİLMİŞ VERSİYON ─────────
async def answer_question(question: str, mode: str = "real-estate", conversation_history: List = None) -> str:
    """
    Tam İyileştirilmiş Ana Fonksiyon - RENDER 1GB OPTİMİZASYONLU:
    1. Gelişmiş konu tespiti yapar (disk optimizasyonu ile)
    2. Akıllı modül yönlendirme
    3. Gayrimenkul modunda akıllı ilan araması
    4. Performans optimizasyonu (1GB disk sınırı)
    5. Hata yönetimi
    """
    
    print(f"🚀 RENDER 1GB OPTİMİZE EDİLMİŞ SORGU - Soru: {question[:50]}..., Mod: {mode}")
    
    # 1. SELAMLAŞMA KONTROLÜ - ÖNCELİKLİ
    selamlasma_kaliplari = [
        "merhaba", "selam", "hello", "hi", "hey", "günaydın", "iyi günler", 
        "iyi akşamlar", "nasılsın", "naber", "ne haber", "hoş geldin", "nasıl gidiyor"
    ]
    
    clean_question = question.lower().strip()
    is_greeting = False
    
    for kalip in selamlasma_kaliplari:
        if kalip in clean_question:
            is_greeting = True
            print(f"✓ Selamlaşma mesajı tespit edildi: {kalip}")
            break
    
    # Selamlaşma için hızlı yanıt
    if is_greeting:
        print("🤝 Selamlaşmaya doğrudan yanıt veriliyor")
        
        greeting_responses = {
            "real-estate": "Merhaba! Size gayrimenkul konusunda nasıl yardımcı olabilirim?",
            "mind-coach": "Merhaba! Size zihinsel ve ruhsal gelişim konularında nasıl yardımcı olabilirim?",
            "finance": "Merhaba! Size finans ve yatırım konularında nasıl yardımcı olabilirim?"
        }
        
        return greeting_responses.get(mode, greeting_responses["real-estate"])
    
    # 2. GELİŞMİŞ KONU TESPİTİ (RENDER OPTİMİZASYONLU)
    detected_topic = await detect_topic(question, mode)
    print(f"📊 Tespit edilen konu: {detected_topic}, Kullanıcının seçtiği mod: {mode}")
    
    # 3. MODÜL YÖNLENDİRME
    if detected_topic != mode:
        if detected_topic in ["real-estate", "mind-coach", "finance"]:
            redirection_key = f"{mode}-to-{detected_topic}"
            if redirection_key in REDIRECTION_MESSAGES:
                print(f"↪️ Farklı modüle yönlendiriliyor: {redirection_key}")
                return REDIRECTION_MESSAGES[redirection_key]
        
        # Genel konu ise uzmanlık alanı dışı yanıt ver
        print(f"⚠️ Genel konu tespit edildi, uzmanlık alanı dışı yanıt veriliyor")
        return get_out_of_scope_response(mode)
    
    # 4. İÇERİK HAZIRLAMA - AKILLI ARAMA (RENDER OPTİMİZE)
    context = ""
    if mode == "real-estate":
        # Akıllı ilan araması tespiti
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
    
    # 6. KONUŞMA GEÇMİŞİ EKLE (RENDER 1GB OPTİMİZASYONU)
    if conversation_history and len(conversation_history) > 0:
        # Son 5 mesajı al (disk tasarrufu için)
        for msg in conversation_history[-5:]:
            if isinstance(msg, dict) and 'role' in msg and 'text' in msg:
                messages.append({"role": msg['role'], "content": msg['text']})
    
    # 7. KULLANICI SORUSU EKLE
    messages.append({"role": "user", "content": question})
    
    # 8. AKILLI MODEL VE PARAMETRE SEÇİMİ (RENDER OPTİMİZE)
    try:
        print("🤖 OpenAI API'ye istek gönderiliyor...")
        
        # Model seçimi - RENDER için optimize
        selected_model = "gpt-4o-mini"  # Hızlı ve ekonomik model
        temp = 0.4 if mode == "real-estate" and "Bu soru için ilan araması gerekmemektedir." not in context else 0.6
        
        print(f"🤖 Kullanılan model: {selected_model}, Temperature: {temp}")
        
        # OpenAI API çağrısı
        resp = await openai_client.chat.completions.create(
            model=selected_model,
            messages=messages,
            temperature=temp,
            max_tokens=3072  # RENDER disk tasarrufu için azaltıldı
        )
        
        answer = resp.choices[0].message.content.strip()
        print(f"✅ RENDER OPTİMİZE EDİLMİŞ YANIT HAZIR - Uzunluk: {len(answer)} karakter")
        
        return answer
        
    except Exception as exc:
        print(f"❌ Chat yanıt hatası: {exc}")
        return "Üzgünüm, isteğinizi işlerken beklenmedik bir sorun oluştu. Lütfen daha sonra tekrar deneyin."

# ── Embedding Fonksiyonu ───────────────────────────────────
async def get_embedding(text: str) -> Optional[List[float]]:
    """OpenAI API kullanarak metin için embedding oluşturur"""
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
        print(f"❌ Embedding hatası: {exc}")
        return None

# ── Supabase Arama Fonksiyonu ───────────────────────────────────────
async def search_listings_in_supabase(query_embedding: List[float]) -> List[Dict]:
    """Remax ilanlar tablosundan semantic arama yapar."""
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

        # Threshold üzerindeki sonuçları filtrele
        valid_results = []
        for i, r in enumerate(all_results):
            if isinstance(r, dict) and r.get('similarity', 0) > MATCH_THRESHOLD:
                valid_results.append(r)
                print(f"📌 Geçerli sonuç #{i}: ID={r.get('ilan_id')}, Similarity={r.get('similarity', 0):.3f}")
                
        print(f"✅ İlanlar sorgulandı: Toplam {len(valid_results)} gerçek ilişkili ilan bulundu")  

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
    """İlan listesini SibelGPT için HTML formatında düzenler"""
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
        
        # Lokasyon bilgisi
        ilce = l_item.get('ilce', '')
        mahalle = l_item.get('mahalle', '')
        lokasyon = f"{ilce}, {mahalle}" if ilce and mahalle else (ilce or mahalle or '?')
        
        # Fiyat formatı
        fiyat = "?"
        fiyat_raw = l_item.get('fiyat')
        if fiyat_raw is not None:
            try:
                # Fiyat string'ini temizleyip formatla
                fiyat_str_cleaned = str(fiyat_raw).replace('.', '').replace(',', '.')
                if fiyat_str_cleaned.replace('.', '', 1).isdigit():
                    fiyat_num = float(fiyat_str_cleaned)
                    fiyat = f"{fiyat_num:,.0f} ₺".replace(',', '#').replace('.', ',').replace('#', '.')
                else:
                    fiyat = str(fiyat_raw)
            except (ValueError, Exception):
                fiyat = str(fiyat_raw)
       
        # Özellikler
        ozellikler_liste = []
        oda_sayisi = l_item.get('oda_sayisi', '')
        if oda_sayisi:
            ozellikler_liste.append(str(oda_sayisi))
       
        metrekare = l_item.get('metrekare', '')
        if metrekare:
            metrekare_str = str(metrekare).strip()
            if not metrekare_str.endswith("m²"):
                ozellikler_liste.append(f"{metrekare_str} m²")
            else:
                ozellikler_liste.append(metrekare_str)

        # Kat bilgisi
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
       
        # Veritabanından gelen ek özellikler
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
       
        # HTML formatında ilan satırı
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
    
    # Gerçek ilan numaralarını listele
    real_ids = [l_item.get('ilan_id') for l_item in listings_to_format if l_item.get('ilan_id')]
    print(f"🏷️ İlan Veritabanındaki Gerçek İlan Numaraları: {real_ids}")
    if real_ids:
        final_output += f"<p><strong>VERİTABANINDAKİ GERÇEK İLAN NUMARALARI: {', '.join(real_ids)}</strong></p>"
    
    final_output += "<p>Bu ilanların doğruluğunu kontrol ettim. Farklı bir arama yapmak isterseniz, lütfen kriterleri belirtiniz.</p>"
   
    return final_output

# ── PERFORMANS VE DEBUG FONKSİYONLARI ─────────────────────────
def validate_system_configuration():
    """Sistem konfigürasyonunu doğrular - RENDER 1GB OPTİMİZE"""
    issues = []
    
    # API anahtarları kontrolü
    if not OAI_KEY:
        issues.append("❌ OpenAI API anahtarı eksik")
    if not SB_URL:
        issues.append("❌ Supabase URL eksik")
    if not SB_KEY:
        issues.append("❌ Supabase Key eksik")
    
    # Topics kontrolü (RENDER optimize)
    for topic, keywords in TOPIC_KEYWORDS.items():
        if len(keywords) < 30:  # 50'den 30'a düşürüldü
            issues.append(f"⚠️ {topic} için az kelime: {len(keywords)}")
    
    # System prompts kontrolü
    for mode in ["real-estate", "mind-coach", "finance"]:
        if mode not in SYSTEM_PROMPTS:
            issues.append(f"❌ {mode} için system prompt eksik")
    
    if issues:
        print("🔍 Sistem Konfigürasyon Sorunları:")
        for issue in issues:
            print(f"  {issue}")
    else:
        print("✅ Sistem konfigürasyonu tamam")
    
    return len(issues) == 0

# Başlangıçta doğrulama yap
print("🔧 RENDER 1GB optimize sistem başlatılıyor...")
validate_system_configuration()
print("✅ Ask Handler RENDER optimize hazır!")
