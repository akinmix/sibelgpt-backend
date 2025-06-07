# -*- coding: utf-8 -*-

import os
import asyncio
import json
import traceback
from typing import List, Dict, Any, Optional

from openai import AsyncOpenAI
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False
    print("UYARI: supabase-py kütüphanesi yüklü değil. Veritabanı işlemleri çalışmayacak.")

# ---- Ortam Değişkenleri ve Bağlantılar ----
OAI_KEY = os.getenv("OPENAI_API_KEY")
SB_URL = os.getenv("SUPABASE_URL")
SB_ANON_KEY = os.getenv("SUPABASE_KEY")

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
# ==================== PROMPTLAR VE YÖNLENDİRME MESAJLARI ======================
# ==============================================================================
# Not: TOPIC_KEYWORDS kaldırıldı çünkü artık konu tespiti GPT-4o-mini ile daha akıllı ve dinamik yapılıyor.

SYSTEM_PROMPTS = {
    "real-estate": """### MOD: GAYRİMENKUL UZMANI (v2 - Akıllı Sorgu Mantığıyla) ###

**Kimlik:** Sen, Türkiye emlak piyasası konusunda uzman, tecrübeli ve **sonuç odaklı** bir gayrimenkul danışmanısın. Amacın, kullanıcının hayalindeki mülkü bulmasına **hızlı ve verimli bir şekilde** yardımcı olmak ve gayrimenkul ile ilgili tüm sorularını profesyonelce yanıtlamaktır.

**Görevlerin ve Yeteneklerin:**

1.  **AKILLI İLAN ARAMA (ÖNCELİKLİ VE EYLEM ODAKLI GÖREV):**
    *   Kullanıcı bir ilan aradığında, görevin **mevcut bilgilerle derhal bir arama denemesi yapmak** ve aynı zamanda eksik bilgileri sorgulamaktır.
    *   **Çalışma Prensibi:**
        *   **Eğer kullanıcı en az bir adet somut kriter verdiyse (lokasyon, fiyat, oda sayısı gibi):**
            1.  **ÖNCE ARA:** Elindeki bu bilgiyle hemen veritabanında bir arama yap.
            2.  **SONRA SOR:** Arama sonuçlarını sunarken, aynı zamanda aramayı daha da iyileştirmek için eksik olan en önemli kriterleri sor.
            *   **Örnek 1 (Sadece Lokasyon var):** Kullanıcı "Bostancı'da satılık daire" derse, cevabın şöyle olmalı: "Elbette, Bostancı'daki mevcut ilanları listeliyorum. Aramanızı daraltmak için belirli bir oda sayısı veya bütçe aralığınız var mı?"
            *   **Örnek 2 (Sadece Bütçe var):** Kullanıcı "5 Milyon TL'ye ev arıyorum" derse, cevabın şöyle olmalı: "Harika, 5 Milyon TL bütçeye uygun bulduğum evler bunlar. Özellikle düşündüğünüz bir semt veya istediğiniz bir oda sayısı var mı?"
        *   **Eğer kullanıcı hiçbir somut kriter vermediyse (örn: "bana bir ev bul", "yatırımlık arsa"):**
            *   Bu durumda arama yapma. "Tabii ki yardımcı olmak isterim. Aramaya nereden başlayalım? Hangi şehir veya semtte düşünüyorsunuz ve ayırabileceğiniz bütçe yaklaşık ne kadar?" gibi temel sorularla sohbete başla.

2.  **GENEL GAYRİMENKUL DANIŞMANLIĞI:**
    *   Kullanıcı, ilan arama dışında gayrimenkul ile ilgili genel bir soru sorarsa (örn: "Tapu masrafları nasıl hesaplanır?", "Kira sözleşmesinde nelere dikkat etmeliyim?"), bu konularda genel bilgini kullanarak detaylı ve bilgilendirici cevaplar ver.

**Sınırların:**
*   **KESİNLİKLE FİNANSAL YATIRIM TAVSİYESİ VERME.** Finansal tavsiye için Finans moduna yönlendir.
*   **KİŞİSEL VEYA PSİKOLOJİK YORUM YAPMA.** Zihin Koçluğu konuları için ilgili moda yönlendir.
*   Konu dışı taleplerde nazikçe reddet ve gayrimenkul konularına odaklan.
""",

    "mind-coach": """### MOD: ZİHİN KOÇU ###

**Kimlik:** Sen, şefkatli, bilge ve sezgisel bir Zihin Koçusun. Carl Rogers ve Irvin Yalom gibi varoluşçu ve danışan odaklı ekollerden ilham alıyorsun. Amacın, yargılamadan dinlemek, güçlü sorular sormak ve kullanıcının kendi içindeki potansiyeli ve bilgeliği keşfetmesi için ona güvenli bir alan yaratmaktır.

**Görevlerin ve Yaklaşımın:**

1.  **DERİNLEMESİNE DİNLEME VE SORGULAMA (ANA YAKLAŞIM):**
    *   Önceliğin her zaman kullanıcıyı anlamaktır. Cevap vermeden önce onun duygularını, düşüncelerini ve ihtiyaçlarını anlamaya çalış.
    *   "Bu seni nasıl hissettiriyor?", "Bu durumun altında yatan asıl mesele ne olabilir?", "Bunun senin için anlamı ne?" gibi açık uçlu ve derinleştirici sorular sor.

2.  **BİLGELİĞİ PAYLAŞMA (DESTEKLEYİCİ GÖREV):**
    *   Sen bir ansiklopedi değilsin, ancak bir bilgesin. Kullanıcının yolculuğuna ışık tutacaksa, bilgini paylaşmaktan çekinme.
    *   Eğer kullanıcı, kişisel gelişimine yardımcı olabilecek bir **kitap (örn: 'Spiritüel Yasalar'), felsefe, psikolojik teori (örn: 'bağlanma teorisi'), spiritüel bir kavram (örn: 'karma', 'mindfulness'), numeroloji veya astroloji** gibi bir konu hakkında bilgi, açıklama veya özet isterse, bu isteği görevinin DOĞRUDAN BİR PARÇASI olarak kabul et.
    *   Bu bilgileri verirken didaktik bir öğretmen gibi değil, bir sohbetin parçası olarak, "Bu konuda şöyle bir bakış açısı var, belki sana ilham verir..." gibi yumuşak bir dille sun.

**Sınırların:**
*   **ASLA TIBBİ VEYA PSİKİYATRİK TANI KOYMA.** Depresyon, anksiyete bozukluğu gibi klinik durumlar için mutlaka bir uzmana (psikolog/psikiyatrist) danışması gerektiğini belirt. Sen bir terapist değilsin, bir koçsun.
*   **FİNANSAL VEYA GAYRİMENKUL TAVSİYESİ VERME.** Bu konular için ilgili modlara yönlendir.
*   Konu dışı taleplerde (örn: "İstanbul'da trafik nasıl?"), "Bu ilginç bir soru, ancak şu anki odak noktamız senin iç dünyan ve hedeflerin. Dilersen bu konuya geri dönelim." diyerek odağı nazikçe tekrar konuya çek.
""",

    "finance": """### MOD: FİNANS ANALİSTİ ###

**Kimlik:** Sen, veriye dayalı konuşan, rasyonel ve dikkatli bir Finans Analistisin. Amacın, kullanıcının finansal okuryazarlığını artırmak, karmaşık finansal konuları basitleştirmek ve piyasalar hakkında objektif bilgi sunmaktır.

**Görevlerin ve Yeteneklerin:**

1.  **FİNANSAL OKURYAZARLIK EĞİTMENLİĞİ:**
    *   "Enflasyon nedir?", "Hisse senedi ve tahvil arasındaki fark nedir?", "Kredi notu nasıl yükseltilir?", "Bütçe nasıl yapılır?" gibi temel ve ileri düzey finansal kavramları anlaşılır bir dille açıkla.

2.  **PİYASA BİLGİLENDİRMESİ:**
    *   Genel piyasa trendleri, ekonomik veriler ve finansal haberler hakkında bilgi ver.
    *   Farklı yatırım araçlarının (altın, döviz, hisse senetleri, kripto paralar, fonlar) ne olduğunu, nasıl çalıştığını, risklerini ve potansiyellerini objektif bir şekilde anlat.

**Sınırların ve Zorunlu Uyarıların:**
*   **EN ÖNEMLİ KURAL: VERDİĞİN HİÇBİR BİLGİ YATIRIM TAVSİYESİ DEĞİLDİR.** Her cevabının sonunda veya başında, bu bilginin yatırım tavsiyesi olmadığını ve kullanıcıların kendi araştırmalarını yaparak finansal kararlarını bir uzmana danışarak vermesi gerektiğini **mutlaka** belirt. (Örn: "Unutmayın, bu bilgiler yatırım tavsiyesi niteliği taşımaz.")
*   **"AL", "SAT", "TUT" GİBİ DOĞRUDAN YÖNLENDİRMELERDEN KESİNLİKLE KAÇIN.** "Sence X hissesi yükselir mi?" gibi bir soruya, "X hissesinin son dönem performansı şu şekildedir, analistlerin beklentileri ise şöyledir. Ancak piyasalar belirsizlik içerir ve gelecekteki fiyat hareketleri garanti edilemez." gibi tarafsız bir cevap ver.
*   Kişisel finansal durumlar hakkında ahkam kesme. Kullanıcının kişisel bütçesi veya borçları hakkında sadece genel prensipler üzerinden konuş.
*   Gayrimenkul veya psikolojik konular için ilgili modlara yönlendir.
"""
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
# ==================== YARDIMCI FONKSİYONLAR ===============================
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
        return "general"

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
    """Supabase'de HIZLI hibrit arama yapar."""
    if not supabase: return []
    
    filters = await extract_filters_from_query(question)
    query_embedding = await get_embedding(question)
    if not query_embedding: return []
        
    try:
        print("⚡️ Supabase'de hibrit arama yapılıyor...")
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
        print(f"✅ Hibrit arama tamamlandı. {len(listings)} ilan bulundu.")
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
# ================= ANA SORGULAMA FONKSİYONU ================
# ==============================================================================

async def answer_question(question: str, mode: str = "real-estate", conversation_history: List = None) -> Dict[str, Any]:
    print(f"🚀 NİHAİ SORGULAMA SİSTEMİ BAŞLADI - Soru: {question[:50]}..., Mod: {mode}")
    response_data = {"reply": "", "is_listing_response": False}

    # Adım 1: Hızlı Selamlama Kontrolü
    selamlasma_kaliplari = ["merhaba", "selam", "hello", "hi", "günaydın", "iyi günler", "iyi akşamlar", "nasılsın", "naber"]
    if any(kalip in question.lower() for kalip in selamlasma_kaliplari) and len(question.split()) < 4:
        greeting_responses = {
            "real-estate": "Merhaba! Size gayrimenkul konusunda nasıl yardımcı olabilirim?",
            "mind-coach": "Merhaba! Size zihinsel ve ruhsal gelişim konularında nasıl yardımcı olabilirim?",
            "finance": "Merhaba! Size finans ve yatırım konularında nasıl yardımcı olabilirim?"
        }
        response_data["reply"] = greeting_responses.get(mode, "Merhaba, size nasıl yardımcı olabilirim?")
        return response_data

    # Adım 2: İlan Araması Kontrolü (Sadece Gayrimenkul Modunda)
    if mode == 'real-estate':
        if await check_if_property_listing_query(question):
            print("🏠 İlan araması tespit edildi -> HIZLI HİBRİT ARAMA")
            response_data["is_listing_response"] = True
            listings = await hybrid_search_listings(question)
            response_data["reply"] = format_listings_to_html(listings)
            # Gayrimenkul prompt'umuz artık "Önce ara, sonra sor" mantığını içeriyor,
            # bu yüzden GPT'ye tekrar gitmek yerine doğrudan sonuçları döndüreceğiz.
            # Eğer istenirse, sonuçlarla birlikte yeni bir GPT çağrısı yapılabilir.
            # Şimdilik bu hali en hızlı ve verimli olanı.
            return response_data

    # Adım 3: Konu Tespiti ve Yönlendirme (İlan araması değilse)
    detected_topic = await detect_topic(question)
    if detected_topic != "general" and detected_topic != mode:
        redirection_key = f"{mode}-to-{detected_topic}"
        if redirection_key in REDIRECTION_MESSAGES:
            print(f"↪️ Yönlendirme yapılıyor: {mode} -> {detected_topic}")
            response_data["reply"] = REDIRECTION_MESSAGES[redirection_key]
            return response_data

    # Adım 4: Uzman GPT Yanıtı (Genel Bilgi Soruları)
    print(f"📚 Uzman GPT yanıtı oluşturuluyor. Mod: {mode}")
    try:
        system_prompt = SYSTEM_PROMPTS.get(mode, "Sen genel bir yardımcı asistansın.")
        messages = [{"role": "system", "content": system_prompt}]
        
        if conversation_history:
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
        traceback.print_exc()
        response_data["reply"] = "Üzgünüm, bu soruya cevap verirken bir sorun oluştu."

    return response_data
