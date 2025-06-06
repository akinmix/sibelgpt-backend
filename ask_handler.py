import os
import asyncio 
import json
import re
from typing import List, Dict, Any, Optional

# Gerekli kütüphaneleri import et
from openai import AsyncOpenAI
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("UYARI: supabase-py kütüphanesi yüklü değil. Veritabanı işlemleri çalışmayacak.")

# ---- Ortam Değişkenleri ve Bağlantılar ----
OAI_KEY = os.getenv("OPENAI_API_KEY")
SB_URL  = os.getenv("SUPABASE_URL")
SB_ANON_KEY = os.getenv("SUPABASE_KEY") # Güvenli (anon) olduğunu teyit etmiştik.

if not all([OAI_KEY, SB_URL, SB_ANON_KEY]):
    raise RuntimeError("Eksik API anahtarı veya Supabase bilgisi (URL ve KEY).")

openai_client = AsyncOpenAI(api_key=OAI_KEY)
supabase: Optional[Client] = None
if SUPABASE_AVAILABLE:
    supabase = create_client(SB_URL, SB_ANON_KEY)

# ---- Ayarlar ----
EMBEDDING_MODEL = "text-embedding-3-small"
MATCH_THRESHOLD = 0.4
MATCH_COUNT = 25

# ==============================================================================
# ==================== TÜM PROMPTLAR, KEYWORDLER VE MESAJLAR ===================
# ==============================================================================

TOPIC_KEYWORDS = {
    "real-estate": ["emlak", "gayrimenkul", "ev", "daire", "konut", "kiralık", "satılık", "tapu", "mortgage", "ipotek", "kredi", "remax", "metrekare", "imar", "arsa", "bina", "kat", "müstakil", "dükkan", "ofis", "iş yeri", "bahçe", "balkon", "oda", "salon", "banyo", "mutfak", "yapı", "inşaat", "tadilat", "taşınmaz", "bağımsız bölüm", "mülkiyet", "hukuk", "hukuki", "intifa", "irtifak", "izalei", "şuyu", "kat mülkiyeti", "kat irtifakı", "paylı mülkiyet", "elbirliği mülkiyeti", "şufa hakkı", "ön alım", "ortaklığın giderilmesi", "üst hakkı", "geçit hakkı", "sükna hakkı", "zilyetlik", "tasarruf", "devir", "temlik", "rehin", "teminat", "fiili kullanım", "miras", "veraset", "mirasçı", "ortak", "ortaklık", "pay", "hisse", "arsa payı", "senet", "ruhsat", "iskân", "iskan", "noter", "vekaletname", "ferağ", "komisyon", "haciz", "kamulaştırma", "tapuda şerh", "ipotek fekki", "tapu sicil", "tapu müdürlüğü", "emlak rayiç", "belediye rayiç", "emlak vergisi", "mtv", "aidat", "harç", "satış bedeli", "vergi dairesi", "belediye başkanlığı", "beyan", "tebligat", "kentsel dönüşüm", "6306", "riskli yapı", "rezerv yapı", "imar barışı", "imar affı", "yapı kayıt belgesi", "iskan belgesi", "enerji kimlik belgesi", "betonarme", "çelik", "tuğla", "panel", "prefabrik", "dubleks", "tripleks", "villa", "apart", "rezidans", "site", "complex", "yazlık", "stüdyo", "loft", "penthouse", "terras", "bodrum", "çatı katı", "zemin etüdü", "yapı ruhsatı", "oturma raporu", "iskan raporu", "proje değişikliği", "yönetim planı", "asansörlü", "güvenlik", "kamera", "interkom", "otopark", "garaj", "jeneratör", "hidrofor", "yangın merdiveni", "ses yalıtımı", "ısı yalıtımı", "cam balkon", "pvc", "alüminyum", "kalorifer", "doğalgaz", "elektrik", "su faturası", "belediye", "çevre temizlik", "asansör", "kapıcı", "merkezi", "ulaşım", "metro", "metrobüs", "otobüs", "taksi", "cadde", "sokak", "mahalle", "semt", "bölge", "ilçe", "şehir merkezi", "sahil", "deniz", "park", "yeşil alan", "yatırım", "getiri", "kira geliri", "değer artışı", "piyasa", "trend", "fiyat", "değerleme", "ekspertiz", "rapor", "analiz", "portföy", "risk", "potansiyel", "konut kredisi faiz", "kira artış oranı", "TEFE", "TÜFE", "endeks", "gayrimenkul yatırım ortaklığı", "sözleşme", "kira sözleşmesi", "satış sözleşmesi", "ön sözleşme", "depozito", "kapora", "peşinat", "taksit", "ödeme planı", "kefil"],
    "mind-coach": ["astroloji", "astrolojik", "burç", "burcum", "yıldız", "yıldızlar", "gezegen", "ay", "güneş", "mars", "venüs", "jüpiter", "satürn", "merkür", "neptün", "uranüs", "plüton", "aslan", "kova", "koç", "balık", "ikizler", "yengeç", "başak", "terazi", "akrep", "yay", "oğlak", "horoskop", "yıldızname", "astral", "kozmik", "evren", "numeroloji", "sayı", "sayılar", "doğum tarihi", "isim analizi", "kader sayısı", "yaşam yolu", "kişilik sayısı", "ruh sayısı", "ifade sayısı", "kalp arzusu", "olgunluk sayısı", "pitagor", "kaldean", "kabala", "gematria", "vibrasyon", "frekans", "spiritüel", "ruhani", "enerji", "aura", "çakra", "kundalini", "meditasyon", "bilinç", "farkındalık", "uyanış", "aydınlanma", "theta", "healing", "şifa", "reiki", "pranic", "kristal", "taş", "maden", "arınma", "temizlik", "koruma", "büyü", "psikoloji", "psikolog", "terapi", "terapist", "danışman", "depresyon", "anksiyete", "stres", "panik", "fobia", "travma", "ptsd", "obsesif", "kompulsif", "bipolar", "sınır", "kişilik", "narsist", "empati", "duygusal", "zeka", "sosyal", "beceri", "kişisel gelişim", "motivasyon", "özgüven", "özsaygı", "özdisiplin", "başarı", "hedef", "amaç", "vizyon", "misyon", "değer", "inanç", "liderlik", "iletişim", "karizma", "etki", "nüfuz", "yaratıcılık", "inovasyon", "çözüm", "problem", "karar", "seçim", "ilişki", "evlilik", "aşk", "sevgi", "çift", "eş", "partner", "aile", "anne", "baba", "çocuk", "kardeş", "akraba", "arkadaş", "bağ", "bağlılık", "güven", "sadakat", "ihanet", "ayrılık", "boşanma", "barışma", "affetme", "kıskançlık", "öfke", "ruh", "can", "nefs", "ego", "benlik", "kimlik", "öz", "asıl", "hakikat", "gerçek", "yanılsama", "maya", "illüzyon", "hayal", "düş", "sembol", "simge", "işaret", "alamet", "kehanet", "kehânet", "falcılık", "büyücülük", "şamanlık", "sufizm", "tasavvuf", "yoga"],
    "finance": ["borsa", "hisse", "pay", "senet", "bist", "nasdaq", "dow", "s&p", "ftse", "dax", "nikkei", "hang seng", "şirket", "halka arz", "ipo", "temettü", "kar payı", "sermaye", "piyasa değeri", "hacim", "işlem", "alış", "satış", "spread", "fiyat", "değer", "teknik analiz", "grafik", "mum", "çubuk", "line", "bar", "trend", "destek", "direnç", "kırılım", "geri çekilme", "fibonacci", "retracement", "rsi", "macd", "stochastic", "bollinger", "moving average", "ema", "sma", "volume", "oscillator", "momentum", "divergence", "konvergens", "temel analiz", "mali tablo", "bilanço", "gelir tablosu", "nakit akım", "karlılık", "roe", "roa", "pe", "pb", "ev/ebitda", "f/k", "pd/dd", "büyüme", "gelir", "gider", "net kar", "brüt kar", "ebitda", "ebit", "faaliyet karı", "vergi", "kripto", "bitcoin", "ethereum", "altcoin", "blockchain", "defi", "nft", "dao", "dex", "cex", "wallet", "cüzdan", "mining", "madencilik", "staking", "yield farming", "liquidity", "smart contract", "akıllı sözleşme", "token", "coin", "fork", "halving", "proof of work", "proof of stake", "consensus", "dolar", "kur", "para", "lira", "döviz", "usd", "eur", "gbp", "jpy", "chf", "try", "parite", "çapraz kur", "swap", "forward", "futures", "option", "altın", "gümüş", "platin", "paladyum", "petrol", "doğalgaz", "buğday", "mısır", "soya", "kakao", "kahve", "şeker", "pamuk", "ekonomi", "enflasyon", "deflasyon", "stagflasyon", "gdp", "gsyh", "durgunluk", "kriz", "canlanma", "iyileşme", "merkez bankası", "fed", "ecb", "tcmb", "faiz", "oran", "para politikası", "mali politika", "bütçe", "açık", "fazla", "yatırım", "portföy", "fon", "etf", "reit", "bono", "tahvil", "sukuk", "viop", "vadeli", "opsiyon", "warrant", "sertifika", "strukturlu", "structured", "hedge", "arbitraj", "spekülatif"]
}

SYSTEM_PROMPTS = {
    "real-estate": """# Gayrimenkul GPT - Ana Görev ve Rol Tanımı\n\nSen SibelGPT'sin: İstanbul emlak piyasası ve gayrimenkul konusunda uzmanlaşmış, \nTürkçe yanıt veren bir yapay zeka asistanısın. Temel görevin kullanıcılara gayrimenkul, \nemlak ve konut konularında yardımcı olmaktır.\n\n## TEMEL KURALLAR - ÇOK ÖNEMLİ\n\n1. **SADECE AŞAĞIDAKİ KONULARDA CEVAP VER**:\n   \n   **🏠 Gayrimenkul Alım-Satım ve Kiralama:**\n   - Ev, daire, konut, villa, arsa, ofis, dükkan alım-satımı\n   - Kiralık ve satılık gayrimenkul ilanları\n   - Emlak piyasası analizi, fiyat trendleri\n   - Gayrimenkul değerleme, ekspertiz işlemleri\n   \n   ** Gayrimenkul mevzuatı ve hukuki kavramlar**:\n  - Mülkiyet hukuku, tapu işlemleri ve tapu türleri (kat mülkiyeti, kat irtifakı, paylı mülkiyet, elbirliği mülkiyeti)\n  - İntifa hakkı, irtifak hakkı, şufa hakkı (ön alım), izale-i şuyu (ortaklığın giderilmesi)\n  - Zilyetlik, devir, ferağ, temlik, ipotek, rehin, tasarruf yetkisi\n  - Miras hukuku ve veraset işlemleri\n  - Gayrimenkul alım-satım ve kira sözleşmeleri, noter işlemleri\n  - İmar durumu, ruhsat, iskan, belediye işlemleri ve resmi harçlar\n  - Emlak vergisi ve tapu harcı ile ilgili mevzuat\n  - Gayrimenkul hukukuyla ilgili kavramlar ve tanımlar\n  - Temel bilgiler, süreçler ve mevzuat açıklamaları\n  - Karmaşık veya şahsi hukuki uyuşmazlıklar ve dava durumlarında kullanıcıyı "Bu konuda kesin ve kişiye özel hukuki değerlendirme için bir avukata danışmalısınız." diyerek uyar.\n   \n   **🏗️ İnşaat ve Yapı Tekniği:**\n   - İnşaat malzemeleri, yapı tekniği, proje analizi\n   - Tadilat, dekorasyon, renovasyon işlemleri\n   - Yapı denetim, betonarme, çelik yapı sistemi\n   - Enerji verimliliği, yalıtım teknikleri\n   \n   **💰 Gayrimenkul Finansmanı:**\n   - Konut kredisi, mortgage işlemleri\n   - Gayrimenkul yatırımı stratejileri\n   - Kira geliri hesaplama, getiri analizi\n   - Emlak portföy yönetimi\n   \n   **🏘️ Lokasyon ve Bölge Analizi:**\n   - Mahalle, semt, ilçe karşılaştırması\n   - Ulaşım, sosyal tesis analizi\n   - Okul, hastane, AVM mesafeleri\n   - Yatırım potansiyeli yüksek bölgeler\n\n2. **DİĞER TÜM KONULARDA ŞÖYLE YANIT VER**:\n   "Bu soru Gayrimenkul GPT'nin uzmanlık alanı dışındadır. Ben sadece gayrimenkul, \nemlak ve konut konularında yardımcı olabilirim. Bu alanlarla ilgili bir sorunuz \nvarsa memnuniyetle cevaplayabilirim."\n\n3. **SADECE AŞAĞIDAKİ SELAMLAŞMA VE SOHBET BAŞLANGICI MESAJLARINA NORMAL CEVAP VER**:\n   - Selamlaşma: "merhaba", "selam", "hello", "hi", "günaydın", "iyi günler", "iyi akşamlar"\n   - Hal hatır: "nasılsın", "naber", "ne haber", "iyi misin"\n   \n   Bu durumda kısaca selamı alabilir ve konuya odaklanabilirsin:\n   "Merhaba! Size gayrimenkul konusunda nasıl yardımcı olabilirim?"\n\n## YANITLAMA FORMATI\n\n1. Bilgileri her zaman şu şekilde düzenle:\n   - Madde işaretleri (<ul><li>)\n   - Numaralı listeler (<ol><li>)\n   - Alt başlıklar (<h3>, <h4>)\n\n2. Önemli bilgileri <span style="color:#e74c3c;font-weight:bold;">renkli ve kalın</span> yap\n\n3. Temel kavramları <strong>kalın</strong> göster\n\n4. Her yanıtın üst kısmında <h3>başlık</h3> kullan\n\n5. Uyarıları özel formatta göster:\n   <div style="background:#f8d7da;padding:10px;border-left:4px solid #dc3545;margin:10px 0;">\n     <strong style="color:#721c24;">⚠️ ÖNEMLİ UYARI:</strong>\n     <p style="color:#721c24;margin-top:5px;">Uyarı metni...</p>\n   </div>\n\n## GAYRİMENKUL İLANLARI KURALLARI\n\nİlan araması sorguları için bu system prompt geçerli DEĞİLDİR. Arama sonuçları doğrudan veritabanından, formatlanmış HTML olarak sunulur. Bu kurallar sadece genel bilgi soruları içindir.\n\n## KAPANIŞ MESAJLARI\n\nHer yanıtın sonuna: "<p style='color:#3498db;'><strong>📞 Profesyonel gayrimenkul danışmanlığı için: 532 687 84 64</strong></p>" ekle.\n""",
    "mind-coach": """# Zihin Koçu GPT - Ana Görev ve Rol Tanımı\n\nSen SibelGPT'sin: Numeroloji, astroloji, kadim bilgiler, psikoloji, ruh sağlığı, thetahealing, \nmotivasyon ve kişisel gelişim konularında uzmanlaşmış, Türkçe yanıt veren bir yapay zeka \nzihin koçusun.\n\n## TEMEL KURALLAR - ÇOK ÖNEMLİ\n\n1. **SADECE AŞAĞIDAKİ KONULARDA CEVAP VER**:\n   \n   **🌟 Astroloji ve Cosmic Bilimler:**\n   - Astroloji, 12 burç, gezegen etkileri, horoskop analizi, doğum haritası\n   \n   **🔢 Numeroloji ve Sayı Bilimi:**\n   - Numeroloji, isim ve doğum tarihi analizleri, yaşam yolu sayısı\n   \n   **🧠 Psikoloji ve Ruh Sağlığı:**\n   - Depresyon, anksiyete, stres yönetimi, panik atak, fobiler, travma\n   \n   **⚡ Enerji Çalışmaları ve Şifa:**\n   - Thetahealing, Reiki, Pranic healing, kristal şifası, çakra, aura\n   \n   **🚀 Kişisel Gelişim ve Motivasyon:**\n   - Özgüven, hedef belirleme, başarı stratejileri, motivasyon, liderlik\n   \n   **💕 İlişkiler ve Aile Terapisi:**\n   - Çift terapisi, evlilik danışmanlığı, aile içi iletişim, aşk psikolojisi\n   \n   **🌸 Spiritüel Gelişim ve Kadim Bilgiler:**\n   - Yoga, meditasyon, sufizm, tasavvuf, rüya analizi, NLP\n\n2. **DİĞER TÜM KONULARDA ŞÖYLE YANIT VER**:\n   "Bu soru Zihin Koçu GPT'nin uzmanlık alanı dışındadır. Ben sadece kişisel gelişim, \npsikoloji, numeroloji, astroloji ve spiritüel konularda yardımcı olabilirim. \nBu alanlarla ilgili bir sorunuz varsa memnuniyetle cevaplayabilirim."\n\n3. **SADECE AŞAĞIDAKİ SELAMLAŞMA VE SOHBET BAŞLANGICI MESAJLARINA NORMAL CEVAP VER**:\n   - Selamlaşma: "merhaba", "selam", "hello", "hi", "günaydın", "iyi günler", "iyi akşamlar"\n   - Hal hatır: "nasılsın", "naber", "ne haber", "iyi misin"\n   \n   Bu durumda kısaca selamı alabilir ve konuya odaklanabilirsin:\n   "Merhaba! Size zihinsel ve ruhsal gelişim konularında nasıl yardımcı olabilirim?"\n\n## YANITLAMA YAKLAŞIMI\n\nCevaplarını empatik, ilham verici ve destekleyici bir tonda ver.\n1. Yanıtlarını HTML formatında oluştur\n2. <ul>, <li>, <strong>, <br> kullan\n\n## ÖNEMLİ UYARILAR\n\nPsikolojik ve ruhsal konularda mutlaka şu uyarıyı ekle:\n\n<div style="background:#e8f5e9;padding:10px;border-left:5px solid #4caf50;margin:10px 0;">\n  <strong>🌟 Not:</strong> Bu bilgiler kişisel gelişim amaçlıdır. Ciddi psikolojik \nsorunlarınız için mutlaka profesyonel yardım alın.\n</div>\n""",
    "finance": """# Finans GPT - Ana Görev ve Rol Tanımı\n\nSen SibelGPT'sin: Borsa, hisse senetleri, teknik/temel analiz, kripto paralar, \nekonomi ve yatırım konularında uzmanlaşmış, Türkçe yanıt veren bir yapay zeka finans danışmanısın.\n\n## TEMEL KURALLAR - ÇOK ÖNEMLİ\n\n1. **SADECE AŞAĞIDAKİ KONULARDA CEVAP VER**:\n   \n   **📈 Borsa ve Hisse Senetleri:**\n   - BIST, NASDAQ, hisse senedi işlemleri, halka arz (IPO), temettü\n   \n   **🔍 Teknik Analiz:**\n   - Grafik türleri, trend analizi, RSI, MACD, Bollinger Bands, Moving Average\n   \n   **📊 Temel Analiz:**\n   - Mali tablo analizi, P/E, P/B, EV/EBITDA, ROE, ROA\n   \n   **₿ Kripto Para ve Blockchain:**\n   - Bitcoin, Ethereum, Altcoin'ler, Blockchain, DeFi, NFT\n   \n   **💱 Döviz ve Emtia Piyasaları:**\n   - USD/TRY, EUR/TRY, Forex, Altın, gümüş, petrol\n   \n   **🌍 Makro ve Mikro Ekonomi:**\n   - Enflasyon, GSYH, büyüme, Merkez bankası politikaları, faiz oranları\n   \n   **🏦 Yatırım Araçları ve Bankacılık:**\n   - Mevduat, tahvil, bono, yatırım fonları, ETF, VİOP\n\n2. **DİĞER TÜM KONULARDA ŞÖYLE YANIT VER**:\n   "Bu soru Finans GPT'nin uzmanlık alanı dışındadır. Ben sadece borsa, yatırım, \nekonomi, kripto para ve finans konularında yardımcı olabilirim. Bu alanlarla \nilgili bir sorunuz varsa memnuniyetle cevaplayabilirim."\n\n3. **SADECE AŞAĞIDAKİ SELAMLAŞMA VE SOHBET BAŞLANGICI MESAJLARINA NORMAL CEVAP VER**:\n   - Selamlaşma: "merhaba", "selam", "hello", "hi", "günaydın", "iyi günler", "iyi akşamlar"\n   - Hal hatır: "nasılsın", "naber", "ne haber", "iyi misin"\n   \n   Bu durumda kısaca selamı alabilir ve konuya odaklanabilirsin:\n   "Merhaba! Size finans ve yatırım konularında nasıl yardımcı olabilirim?"\n\n## YANITLAMA YAKLAŞIMI\n\nCevaplarını net, anlaşılır ve profesyonel bir tonda ver.\n1. Yanıtlarını HTML formatında oluştur\n2. <ul>, <li>, <strong>, <br> kullan\n\n## ÖNEMLİ UYARILAR\n\nFinans önerilerinde mutlaka şu uyarıyı ekle:\n\n<div style="background:#fff3e0;padding:10px;border-left:5px solid #ff9800;margin:10px 0;">\n  <strong>⚠️ Risk Uyarısı:</strong> Burada sunulan bilgiler yatırım tavsiyesi değildir. \nTüm yatırım ve finansal kararlar kendi sorumluluğunuzdadır. Yatırım yapmadan önce \nprofesyonel danışmanlık almanız önerilir.\n</div>\n"""
}

REDIRECTION_MESSAGES = {
    "real-estate-to-mind-coach": "<h3>Bu soru Zihin Koçu GPT için daha uygun görünüyor.</h3><p>Şu anda <strong>Gayrimenkul GPT</strong> modülündesiniz, ancak sorduğunuz soru numeroloji, astroloji, psikoloji veya kişisel gelişim ile ilgili görünüyor.</p><p>Daha iyi bir yanıt almak için lütfen üst menüden <strong>🧠 Zihin Koçu GPT</strong> butonuna tıklayarak modül değiştiriniz. Ardından sorunuzu tekrar sorabilirsiniz.</p>",
    "real-estate-to-finance": "<h3>Bu soru Finans GPT için daha uygun görünüyor.</h3><p>Şu anda <strong>Gayrimenkul GPT</strong> modülündesiniz, ancak sorduğunuz soru borsa, hisse senetleri, yatırım, ekonomi veya finans ile ilgili görünüyor.</p><p>Daha iyi bir yanıt almak için lütfen üst menüden <strong>💰 Finans GPT</strong> butonuna tıklayarak modül değiştiriniz. Ardından sorunuzu tekrar sorabilirsiniz.</p>",
    "mind-coach-to-real-estate": "<h3>Bu soru Gayrimenkul GPT için daha uygun görünüyor.</h3><p>Şu anda <strong>Zihin Koçu GPT</strong> modülündesiniz, ancak sorduğunuz soru emlak, gayrimenkul, satılık/kiralık ilanlar veya inşaat ile ilgili görünüyor.</p><p>Daha iyi bir yanıt almak için lütfen üst menüden <strong>🏠 Gayrimenkul GPT</strong> butonuna tıklayarak modül değiştiriniz. Ardından sorunuzu tekrar sorabilirsiniz.</p>",
    "mind-coach-to-finance": "<h3>Bu soru Finans GPT için daha uygun görünüyor.</h3><p>Şu anda <strong>Zihin Koçu GPT</strong> modülündesiniz, ancak sorduğunuz soru borsa, hisse senetleri, yatırım, ekonomi veya finans ile ilgili görünüyor.</p><p>Daha iyi bir yanıt almak için lütfen üst menüden <strong>💰 Finans GPT</strong> butonuna tıklayarak modül değiştiriniz. Ardından sorunuzu tekrar sorabilirsiniz.</p>",
    "finance-to-real-estate": "<h3>Bu soru Gayrimenkul GPT için daha uygun görünüyor.</h3><p>Şu anda <strong>Finans GPT</strong> modülündesiniz, ancak sorduğunuz soru emlak, gayrimenkul, satılık/kiralık ilanlar veya inşaat ile ilgili görünüyor.</p><p>Daha iyi bir yanıt almak için lütfen üst menüden <strong>🏠 Gayrimenkul GPT</strong> butonuna tıklayarak modül değiştiriniz. Ardından sorunuzu tekrar sorabilirsiniz.</p>",
    "finance-to-mind-coach": "<h3>Bu soru Zihin Koçu GPT için daha uygun görünüyor.</h3><p>Şu anda <strong>Finans GPT</strong> modülündesiniz, ancak sorduğunuz soru numeroloji, astroloji, psikoloji veya kişisel gelişim ile ilgili görünüyor.</p><p>Daha iyi bir yanıt almak için lütfen üst menüden <strong>🧠 Zihin Koçu GPT</strong> butonuna tıklayarak modül değiştiriniz. Ardından sorunuzu tekrar sorabilirsiniz.</p>"
}

# ==============================================================================
# ==================== TÜM YARDIMCI FONKSİYONLAR ===============================
# ==============================================================================

async def get_embedding(text: str) -> Optional[List[float]]:
    """Metin için OpenAI embedding'i oluşturur."""
    try:
        resp = await openai_client.embeddings.create(model=EMBEDDING_MODEL, input=[text.strip()])
        return resp.data[0].embedding
    except Exception as e:
        print(f"❌ Embedding hatası: {e}")
        return None

async def detect_topic(question: str) -> str:
    """Kullanıcının sorusunun ana konusunu (topic) tespit eder."""
    print(f"🔎 Konu tespiti başlatıldı: {question[:50]}...")
    try:
        resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """Kullanıcının sorusunu analiz et ve SADECE şu üç kategoriden birini döndür: real-estate, mind-coach, finance. Eğer hiçbiriyle ilgili değilse veya bir selamlama ise 'general' de. Örnek: "İntifa hakkı nedir?" -> real-estate. "Bitcoin ne olur?" -> finance. "Merhaba" -> general."""},
                {"role": "user", "content": question}
            ],
            temperature=0.0, max_tokens=15
        )
        topic = resp.choices[0].message.content.strip().lower()
        print(f"🤖 GPT konu tespiti: {topic}")
        return topic if topic in ["real-estate", "mind-coach", "finance", "general"] else "general"
    except Exception as e:
        print(f"❌ Konu tespiti hatası: {e}")
        return "general" # Hata durumunda genel kabul et

async def extract_filters_from_query(question: str) -> Dict:
    """Sorgudan SADECE yapısal filtreleri çıkarır (Hızlı ve Akıllı Versiyon)."""
    print(f"🔍 Akıllı filtre çıkarma işlemi başlatıldı: {question}")
    try:
        resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": """Sen bir emlak arama asistanısın. Kullanıcının sorgusundan SADECE şu filtreleri JSON olarak çıkar: "min_fiyat", "max_fiyat", "oda_sayisi", ve "lokasyon" (TÜM ilçe/mahalle adları). 'ilce'/'mahalle' diye ayırma, sadece 'lokasyon' kullan. Örnek: "kadıköyde 5 milyona kadar 2+1 daire" -> {"max_fiyat": 5000000, "oda_sayisi": "2+1", "lokasyon": "Kadıköy"}. Sadece bulabildiklerini ekle."""},
                {"role": "user", "content": question}
            ],
            response_format={"type": "json_object"}, temperature=0.0, max_tokens=200
        )
        filters = json.loads(resp.choices[0].message.content)
        print(f"✅ Çıkarılan akıllı filtreler: {filters}")
        return filters
    except Exception as e:
        print(f"❌ Filtre çıkarma hatası: {e}")
        return {}

async def hybrid_search_listings(question: str) -> List[Dict]:
    """Supabase'de HIZLI hibrit arama yapar (v2 - Akıllı Lokasyon)."""
    if not supabase: return []
    
    filters = await extract_filters_from_query(question)
    query_embedding = await get_embedding(question)
    if not query_embedding: return []
        
    try:
        print("⚡️ Supabase'de v2 hibrit arama yapılıyor...")
        rpc_params = {
            "query_embedding": query_embedding,
            "match_threshold": MATCH_THRESHOLD,
            "match_count": MATCH_COUNT,
            "p_max_fiyat": filters.get("max_fiyat"),
            "p_min_fiyat": filters.get("min_fiyat"),
            "p_oda_sayisi": filters.get("oda_sayisi"),
            "p_lokasyon": filters.get("lokasyon")
        }
        rpc_params = {k: v for k, v in rpc_params.items() if v is not None}
        response = await asyncio.to_thread(supabase.rpc("search_listings_hybrid", rpc_params).execute)
        listings = response.data if hasattr(response, 'data') and response.data else []
        print(f"✅ v2 Hibrit arama tamamlandı. {len(listings)} ilan bulundu.")
        return listings
    except Exception as e:
        print(f"❌ Supabase RPC ('search_listings_hybrid') hatası: {e}\n{traceback.format_exc()}")
        return []

def format_listings_to_html(listings: List[Dict]) -> str:
    """İlan listesini şık bir HTML'e dönüştürür."""
    if not listings:
        return "<p>🔍 Üzgünüm, belirttiğiniz kriterlere uygun bir ilan bulamadım. Lütfen arama kriterlerinizi değiştirerek tekrar deneyin.</p>"

    def format_price(val):
        try:
            num = float(val)
            return f"{num:,.0f} ₺".replace(',', 'X').replace('.', ',').replace('X', '.')
        except (ValueError, TypeError):
            return str(val or 'N/A')

    html_parts = [
        f"<h3 style='color: #4dabf7;'>İşte sorgunuza en uygun {len(listings)} ilan:</h3>",
        "<ul style='list-style-type: none; padding: 0;'>"
    ]
    for ilan in listings:
        ilan_no = ilan.get('ilan_id', 'N/A')
        baslik = ilan.get('baslik', 'Başlık Yok')
        lokasyon = f"{ilan.get('ilce', '')}, {ilan.get('mahalle', '')}".strip(", ")
        fiyat = format_price(ilan.get('fiyat_numeric') or ilan.get('fiyat'))
        oda_sayisi = ilan.get('oda_sayisi', '')
        metrekare = f"{ilan.get('metrekare')} m²" if ilan.get('metrekare') else ''
        pdf_button = f"<a href='https://sibelgpt-backend.onrender.com/generate-property-pdf/{ilan_no}' target='_blank' style='display: inline-block; margin-top: 8px; padding: 6px 12px; background-color: #e53935; color: white; text-decoration: none; border-radius: 4px; font-size: 13px;'>📄 PDF Görüntüle</a>"
        html_parts.append(f"""<li style='background: rgba(40, 40, 40, 0.6); border-left: 4px solid #4dabf7; padding: 15px; margin-bottom: 12px; border-radius: 8px;'><strong style='font-size: 16px; color: #ffffff;'>{baslik}</strong><br><span style='font-size: 14px; color: #cccccc;'>📍 {lokasyon}  |  🏠 {oda_sayisi} ({metrekare})</span><br><span style='font-size: 15px; font-weight: bold; color: #81c784;'>💰 {fiyat}</span>{pdf_button}</li>""")
    html_parts.append("</ul><p>Daha fazla detay veya farklı bir arama için lütfen belirtin.</p>")
    return "\n".join(html_parts)

async def check_if_property_listing_query(question: str) -> bool:
    """Sorunun ilan araması gerektirip gerektirmediğini tespit eder."""
    try:
        resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "system","content": """Kullanıcının sorusunu analiz et ve sadece "Evet" veya "Hayır" yanıtı ver. İLAN ARAMASI GEREKTİREN SORULAR (Evet): "Kadıköy'de satılık daire bul/ara/göster", "20 milyona kadar 3+1 daire arıyorum", "Beşiktaş'ta ev var mı?", "Maltepe'de villa göster/listele". İLAN ARAMASI GEREKTİRMEYEN SORULAR (Hayır): "Ev alırken nelere dikkat etmeliyim?", "Konut kredisi nasıl alınır?". Sadece "Evet" veya "Hayır" yanıtı ver."""},
                      {"role": "user", "content": question}],
            temperature=0.0, max_tokens=10
        )
        is_listing_query = "evet" in resp.choices[0].message.content.strip().lower()
        print(f"📊 İlan araması tespiti: {is_listing_query}")
        return is_listing_query
    except Exception as e:
        print(f"❌ İlan araması tespiti hatası: {e}")
        return False

# ==============================================================================
# ================= ANA SORGULAMA FONKSİYONU (NİHAİ VE TAM HAL) ================
# ==============================================================================

async def answer_question(question: str, mode: str = "real-estate", conversation_history: List = None) -> Dict[str, Any]:
    print(f"🚀 NİHAİ SORGULAMA SİSTEMİ BAŞLADI - Soru: {question[:50]}..., Mod: {mode}")
    response_data = {"reply": "", "is_listing_response": False}

    # 1. Adım: Hızlı Selamlama Kontrolü
    selamlasma_kaliplari = ["merhaba", "selam", "hello", "hi", "günaydın", "iyi günler", "iyi akşamlar", "nasılsın", "naber"]
    if any(kalip in question.lower() for kalip in selamlasma_kaliplari) and len(question.split()) < 4:
        greeting_responses = {
            "real-estate": "Merhaba! Size gayrimenkul konusunda nasıl yardımcı olabilirim?",
            "mind-coach": "Merhaba! Size zihinsel ve ruhsal gelişim konularında nasıl yardımcı olabilirim?",
            "finance": "Merhaba! Size finans ve yatırım konularında nasıl yardımcı olabilirim?"
        }
        response_data["reply"] = greeting_responses.get(mode, "Merhaba, size nasıl yardımcı olabilirim?")
        return response_data

    # 2. Adım: İlan Araması Kontrolü (Sadece Gayrimenkul Modunda)
    if mode == 'real-estate':
        if await check_if_property_listing_query(question):
            print("🏠 İlan araması tespit edildi -> HIZLI HİBRİT ARAMA")
            response_data["is_listing_response"] = True
            listings = await hybrid_search_listings(question)
            response_data["reply"] = format_listings_to_html(listings)
            return response_data

    # 3. Adım: Konu Tespiti ve Yönlendirme
    detected_topic = await detect_topic(question)
    if detected_topic != "general" and detected_topic != mode:
        redirection_key = f"{mode}-to-{detected_topic}"
        if redirection_key in REDIRECTION_MESSAGES:
            print(f"↪️ Yönlendirme yapılıyor: {mode} -> {detected_topic}")
            response_data["reply"] = REDIRECTION_MESSAGES[redirection_key]
            return response_data

    # 4. Adım: Uzman GPT Yanıtı (Genel Bilgi Soruları)
    print(f"📚 Uzman GPT yanıtı oluşturuluyor. Mod: {mode}")
    try:
        system_prompt = SYSTEM_PROMPTS.get(mode, "Sen genel bir yardımcı asistansın.")
        messages = [{"role": "system", "content": system_prompt}]
        if conversation_history:
            # Sadece text ve role alanlarını alarak temiz bir geçmiş oluştur
            clean_history = [{"role": msg.get("role"), "content": msg.get("text")} for msg in conversation_history[-5:] if msg.get("role") and msg.get("text")]
            messages.extend(clean_history)
        
        messages.append({"role": "user", "content": question})

        resp = await openai_client.chat.completions.create(
            model="gpt-4o",
            messages=messages,
            temperature=0.5,
            max_tokens=2048
        )
        response_data["reply"] = resp.choices[0].message.content.strip()

    except Exception as e:
        print(f"❌ Genel GPT yanıt hatası: {e}")
        import traceback
        traceback.print_exc()
        response_data["reply"] = "Üzgünüm, bu soruya cevap verirken bir sorun oluştu."

    return response_data
