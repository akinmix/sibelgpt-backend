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
MATCH_THRESHOLD = 0.65
MATCH_COUNT     = 20

# ── Modlara Göre System Prompts ────────────────────────────
SYSTEM_PROMPTS = {
    "real-estate": """
    Sen SibelGPT'sin: Türkiye emlak piyasası konusunda uzman, 
    Türkçe yanıt veren yardımsever bir yapay zeka asistansın.
    
    Uzmanlık alanların şunlardır:
    - Emlak piyasası ile ilgili her türlü konu (mevzuat, satılık/kiralık ilan arama)
    - Türkiye ve dünyada emlak piyasasındaki gelişmeler, trendler
    - İnşaat ve gayrimenkul yatırımı konuları
    - Kullanıcının bir gayrimenkulü varsa, satış danışmanlığı yap: konum, oda sayısı, kat durumu, yapı yılı, m², iskan durumu gibi bilgileri sorarak pazarlama tavsiyesi ver.
    
    ÖNEMLİ KURALLAR:
    1. İlanlarda ASLA danışman adı veya firma bilgisi belirtme. İlanları nötr bir şekilde sun.
    2. Sadece SATILIK ilanları göster, kiralık ilanları filtreleme.
    3. Yanıtlarının sonuna her zaman "📞 Bu ilanlar hakkında daha fazla bilgi almak isterseniz: 532 687 84 64" ekle.
    4. İlanları sıralarken en uygun olanlarını üste koy, site ismini eklemeyi unutma.
    5. Benzer ilanlardaki tekrarlardan kaçın, çeşitliliği korumaya çalış.
    6. Her ilana bir numara ver ve açıkça formatla.
    7. İlan bilgilerinin doğruluğunu kontrol ettiğini belirt.
    
    Eğer kullanıcı sana Zihin Koçu (numeroloji, astroloji, kadim bilgiler, psikoloji, ruh sağlığı, 
    thetahealing, motivasyon, kişisel gelişim) veya Finans (borsa, hisse senetleri, teknik/temel 
    analiz, kripto paralar, faiz, tahviller, emtia, döviz piyasası, makro/mikro ekonomi) konularında 
    bir soru sorarsa, kullanıcıyı ilgili GPT modülüne yönlendir.
    
    Kullanıcı sana emlak sorusu sorduğunda, Supabase'den getirilen 'İLGİLİ İLANLAR' 
    verilerini kullanarak en alakalı ilanları seçip listele. Eğer yeterli veri yoksa 
    dürüstçe belirt ve kullanıcıya sorular sorarak ihtiyacını netleştir.
    
    Cevaplarını kısa, net ve samimi tut; her ilanda başlık, ilan numarası, fiyat, lokasyon ve özellik bilgisi olsun. Sadece teknik bilgi verme; aynı zamanda samimi, bilinçli ve güven veren bir danışman gibi davran.
    
    Yanıtlarını HTML formatında oluştur. <ul> ve <li> kullan. Satır atlamak için <br>, 
    kalın yazı için <strong> kullan. Markdown işaretleri (*, -) kullanma.
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
    """Her iki tablodan da (ilanlar ve remax_ilanlar) ilanları arar ve birleştirir."""
    if query_embedding is None:
        return []
    
    try:
        # Önce orijinal ilanlar tablosunu sorgula
        office_resp = supabase.rpc(
            "match_listings",
            {
                "query_embedding": query_embedding,
                "match_threshold": MATCH_THRESHOLD,
                "match_count": MATCH_COUNT // 2  # Toplam sonuç sayısının yarısı
            }
        ).execute()
        
        # Sonra remax_ilanlar tablosunu sorgula
        remax_resp = supabase.rpc(
            "match_remax_listings",  # Bu fonksiyonu Supabase'de oluşturmalısınız
            {
                "query_embedding": query_embedding,
                "match_threshold": MATCH_THRESHOLD,
                "match_count": MATCH_COUNT // 2  # Toplam sonuç sayısının yarısı
            }
        ).execute()
        
        # Sonuçları birleştir
        office_data = office_resp.data if hasattr(office_resp, "data") else office_resp
        remax_data = remax_resp.data if hasattr(remax_resp, "data") else remax_resp
        
        # Tüm sonuçları birleştir ve benzerlik puanına göre sırala
        all_results = []
        all_results.extend(office_data)
        all_results.extend(remax_data)
        
        # Benzerlik puanına göre sırala (en yüksek benzerlik önce)
        sorted_results = sorted(all_results, key=lambda x: x.get('similarity', 0), reverse=True)
        
        # En yüksek benzerliğe sahip MATCH_COUNT kadar sonucu döndür
        return sorted_results[:MATCH_COUNT]
        
    except Exception as exc:
        print("❌ Supabase RPC hatası:", exc)
        print(f"Hata detayı: {str(exc)}")
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
            pass

    formatted_parts = []
    for i, l in enumerate(listings, start=1):
        # İlan numarası belirleme (farklı tablolarda farklı alan adları olabilir)
        ilan_no = l.get("ilan_no", l.get("ilan_id", "(numara yok)"))
        
        # Başlık temizleme
        baslik = re.sub(r"^\d+\.\s*", "", l.get("baslik", "(başlık yok)"))
        
        lokasyon = l.get("lokasyon", "?")
        fiyat_raw = l.get("fiyat")
        ozellikler = l.get("ozellikler", "(özellik yok)")
        fiyat = "?"

        try:
            # Fiyat formatlaması
            fiyat_num = float(str(fiyat_raw).replace('.', '').replace(',', '.'))
            try:
                fiyat = locale.currency(fiyat_num, symbol='₺', grouping=True)
                if fiyat.endswith('.00') or fiyat.endswith(',00'):
                    fiyat = fiyat[:-3] + ' ₺'
                else:
                    fiyat = fiyat.replace('₺', '').strip() + ' ₺'
            except:
                fiyat = f"{fiyat_num:,.0f} ₺".replace(',', '#').replace('.', ',').replace('#', '.')
        except:
            fiyat = str(fiyat_raw) if fiyat_raw else "?"

        # İlan kaynak bilgisi ekleme
        source_text = ""
        if "remax" in str(ilan_no).lower() or any("remax" in str(field).lower() for field in l.values()):
            source_text = "<strong>REMAX İlanı</strong><br>"

        ilan_html = (
            f"<li>{source_text}<strong>{i}. {baslik}</strong><br>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;• İlan No: {ilan_no}<br>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;• Lokasyon: {lokasyon}<br>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;• Fiyat: {fiyat}<br>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;• Özellikler: {ozellikler}</li><br>"
        )
        formatted_parts.append(ilan_html)

    final_output = "<ul>" + "\n".join(formatted_parts) + "</ul>"
    final_output += "<br>📞 Bu ilanlar hakkında daha fazla bilgi almak isterseniz: 532 687 84 64"
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
            max_tokens=1024
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
