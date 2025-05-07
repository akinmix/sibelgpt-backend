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
    """İlanları formatlayarak eksiksiz HTML'e dönüştürür."""
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
    MAX_LISTINGS_TO_SHOW =  10 # Daha fazla ilan göstermek için artırıldı
    listings_to_format = listings[:MAX_LISTINGS_TO_SHOW]
    
    # Önemli: Telefon numarasını en başta göster
    final_output = "<p><strong>📞 Bu ilanlar hakkında bilgi almak için: 532 687 84 64</strong></p>"
    
    # Toplam ilan sayısı bilgisi ekle
    total_count = len(listings)
    shown_count = len(listings_to_format)
    
    if total_count > shown_count:
        final_output += f"<p>Toplam {total_count} ilan bulundu, en alakalı {shown_count} tanesi gösteriliyor:</p>"
    
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
                # Float olarak gelirse tamsayıya çevir (3.0 -> 3)
                kat_no = float(bulundugu_kat)
                if kat_no.is_integer():
                    kat_no = int(kat_no)
                
                # Özel durumlar için kontrol
                if kat_no == 0:
                    ozellikler_liste.append("Giriş Kat")
                elif kat_no < 0:
                    ozellikler_liste.append("Bodrum")
                else:
                    ozellikler_liste.append(f"{kat_no}. Kat")
            except:
                # Sayı olarak çevrilemezse olduğu gibi göster
                ozellikler_liste.append(f"{bulundugu_kat}. Kat")
        
        # Özellikler string'i - varsa alanı kullan, yoksa liste oluştur
        if 'ozellikler' in l and l['ozellikler']:
            ozellikler = l['ozellikler']
        else:
            ozellikler = " | ".join(ozellikler_liste) if ozellikler_liste else "(özellik bilgisi yok)"
        
        # HTML oluştur - başlık kırpılmadan, tüm bilgiler dahil edilmiş
        ilan_html = (
            f"<li><strong>{i}. {baslik}</strong><br>"
            f"İlan No: {ilan_no} | Lokasyon: {lokasyon}<br>"
            f"Fiyat: {fiyat} | {ozellikler}</li>"
        )
        formatted_parts.append(ilan_html)
    
    # Liste HTML'i ekle
    final_output += "<ul>" + "\n".join(formatted_parts) + "</ul>"
    
    final_output += "<p>Bu ilanların doğruluğunu kontrol ettim. Eğer daha fazla bilgi almak isterseniz, lütfen bir kriterle arama yapmak istediğinizi belirtin.</p>"
    
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
