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
1.  **DERİNLEMESİNE DİNLEME VE SORGULAMA (ANA YAKLAŞIM):** Önceliğin her zaman kullanıcıyı anlamaktır. "Bu seni nasıl hissettiriyor?", "Bu durumun altında yatan asıl mesele ne olabilir?" gibi açık uçlu ve derinleştirici sorular sor.
2.  **BİLGELİĞİ PAYLAŞMA (DESTEKLEYİCİ GÖREV):** Eğer kullanıcı, kişisel gelişimine yardımcı olabilecek bir **kitap, felsefe, psikolojik teori, spiritüel kavram, numeroloji veya astroloji** hakkında bilgi veya özet isterse, bu isteği görevinin DOĞRUDAN BİR PARÇASI olarak kabul et. Bu bilgileri bir sohbetin parçası olarak yumuşak bir dille sun.
**Sınırların:**
*   **ASLA TIBBİ VEYA PSİKİYATRİK TANI KOYMA.** Klinik durumlar için mutlaka bir uzmana danışması gerektiğini belirt.
*   **FİNANSAL VEYA GAYRİMENKUL TAVSİYESİ VERME.** Bu konular için ilgili modlara yönlendir.
""",
    "finance": """### MOD: FİNANS ANALİSTİ ###
**Kimlik:** Sen, veriye dayalı konuşan, rasyonel ve dikkatli bir Finans Analistisin. Amacın, kullanıcının finansal okuryazarlığını artırmak, karmaşık finansal konuları basitleştirmek ve piyasalar hakkında objektif bilgi sunmaktır.
**Görevlerin ve Yeteneklerin:**
1.  **FİNANSAL OKURYAZARLIK EĞİTMENLİĞİ:** "Enflasyon nedir?", "Hisse senedi ve tahvil arasındaki fark nedir?" gibi temel finansal kavramları anlaşılır bir dille açıkla.
2.  **PİYASA BİLGİLENDİRMESİ:** Genel piyasa trendleri ve farklı yatırım araçlarının ne olduğunu, risklerini ve potansiyellerini objektif bir şekilde anlat.
**Sınırların ve Zorunlu Uyarıların:**
*   **EN ÖNEMLİ KURAL: VERDİĞİN HİÇBİR BİLGİ YATIRIM TAVSİYESİ DEĞİLDİR.** Her cevabında bu bilginin yatırım tavsiyesi olmadığını ve profesyonel danışmanlık almanın önemini belirt.
*   **"AL", "SAT", "TUT" GİBİ DOĞRUDAN YÖNLENDİRMELERDEN KESİNLİKLE KAÇIN.** "Sence X hissesi yükselir mi?" gibi bir soruya tarafsız bir analizle cevap ver.
"""
}

REDIRECTION_MESSAGES = {
    "real-estate-to-mind-coach": "<h3>Bu soru Zihin Koçu GPT için daha uygun görünüyor.</h3><p>Şu anda <strong>Gayrimenkul GPT</strong> modülündesiniz, ancak sorduğunuz soru psikoloji veya kişisel gelişim ile ilgili görünüyor.</p><p>Daha iyi bir yanıt almak için lütfen üst menüden <strong>🧠 Zihin Koçu GPT</strong> butonuna tıklayarak modül değiştiriniz.</p>",
    "real-estate-to-finance": "<h3>Bu soru Finans GPT için daha uygun görünüyor.</h3><p>Şu anda <strong>Gayrimenkul GPT</strong> modülündesiniz, ancak sorduğunuz soru borsa veya yatırım ile ilgili görünüyor.</p><p>Daha iyi bir yanıt almak için lütfen üst menüden <strong>💰 Finans GPT</strong> butonuna tıklayarak modül değiştiriniz.</p>",
    "mind-coach-to-real-estate": "<h3>Bu soru Gayrimenkul GPT için daha uygun görünüyor.</h3><p>Şu anda <strong>Zihin Koçu GPT</strong> modülündesiniz, ancak sorduğunuz soru emlak veya gayrimenkul ile ilgili görünüyor.</p><p>Daha iyi bir yanıt almak için lütfen üst menüden <strong>🏠 Gayrimenkul GPT</strong> butonuna tıklayarak modül değiştiriniz.</p>",
    "mind-coach-to-finance": "<h3>Bu soru Finans GPT için daha uygun görünüyor.</h3><p>Şu anda <strong>Zihin Koçu GPT</strong> modülündesiniz, ancak sorduğunuz soru borsa veya yatırım ile ilgili görünüyor.</p><p>Daha iyi bir yanıt almak için lütfen üst menüden <strong>💰 Finans GPT</strong> butonuna tıklayarak modül değiştiriniz.</p>",
    "finance-to-real-estate": "<h3>Bu soru Gayrimenkul GPT için daha uygun görünüyor.</h3><p>Şu anda <strong>Finans GPT</strong> modülündesiniz, ancak sorduğunuz soru emlak veya gayrimenkul ile ilgili görünüyor.</p><p>Daha iyi bir yanıt almak için lütfen üst menüden <strong>🏠 Gayrimenkul GPT</strong> butonuna tıklayarak modül değiştiriniz.</p>",
    "finance-to-mind-coach": "<h3>Bu soru Zihin Koçu GPT için daha uygun görünüyor.</h3><p>Şu anda <strong>Finans GPT</strong> modülündesiniz, ancak sorduğunuz soru psikoloji veya kişisel gelişim ile ilgili görünüyor.</p><p>Daha iyi bir yanıt almak için lütfen üst menüden <strong>🧠 Zihin Koçu GPT</strong> butonuna tıklayarak modül değiştiriniz.</p>"
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
                {"role": "system", "content": "Kullanıcının sorusunu analiz et ve SADECE şu üç kategoriden birini döndür: real-estate, mind-coach, finance. Eğer hiçbiriyle ilgili değilse veya bir selamlama ise 'general' de."},
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
    """Sorgudan yapısal filtreleri çıkarır (v3 - Daha Akıllı)."""
    print(f"🔍 Akıllı filtre çıkarma işlemi başlatıldı: {question}")
    system_content = """Sen bir emlak arama asistanısın. Kullanıcının sorgusundan SADECE şu filtreleri JSON olarak çıkar: "min_fiyat", "max_fiyat", "oda_sayisi", ve "lokasyon".
ÖNEMLİ: Türkçe'deki yer bildiren ekleri (-de, -da, -'te, -'ta, -'deki, -'daki) yok sayarak lokasyonun kök/yalın halini çıkar.
Örnek 1: "kadıköy'de 5 milyona kadar 2+1 daire" -> {"max_fiyat": 5000000, "oda_sayisi": "2+1", "lokasyon": "Kadıköy"}
Örnek 2: "beşiktaş'taki 3+1 evler" -> {"oda_sayisi": "3+1", "lokasyon": "Beşiktaş"}
Örnek 3: "Bostancı" -> {"lokasyon": "Bostancı"}
Sadece bulabildiklerini JSON'a ekle. Eğer hiçbir şey bulamazsan boş bir JSON: {} döndür."""
    try:
        resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system_content},
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

def _format_listings_for_gpt(listings: List[Dict]) -> str:
    """İlan listesini GPT'nin anlayacağı basit bir metne dönüştürür. (İç kullanım için)"""
    if not listings:
        return "İlan bulunamadı."
    
    text_parts = ["Bulunan İlanlar Özetleri:\n"]
    for i, ilan in enumerate(listings[:5], 1): # GPT'yi yormamak için ilk 5 ilanı alalım
        baslik = ilan.get('baslik', 'N/A')
        lokasyon = f"{ilan.get('ilce', '')}, {ilan.get('mahalle', '')}".strip(", ")
        fiyat = ilan.get('fiyat_numeric') or ilan.get('fiyat', 'N/A')
        oda_sayisi = ilan.get('oda_sayisi', 'N/A')
        metrekare = ilan.get('metrekare', 'N/A')
        text_parts.append(f"{i}. Başlık: {baslik} | Lokasyon: {lokasyon} | Fiyat: {fiyat} | Oda: {oda_sayisi} | m2: {metrekare}")
    
    if len(listings) > 5:
        text_parts.append(f"\n... ve toplamda {len(listings)} adet ilan bulundu.")
        
    return "\n".join(text_parts)


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
# ================= ANA SORGULAMA FONKSİYONU (v13 - AKILLI) ====================
# ==============================================================================

async def answer_question(question: str, mode: str = "real-estate", conversation_history: List = None) -> Dict[str, Any]:
    print(f"🚀 AKILLI SORGULAMA SİSTEMİ (v13) BAŞLADI - Soru: {question[:50]}..., Mod: {mode}")
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

    # Adım 2: Akıllı İlan Arama Mantığı (Sadece Gayrimenkul Modunda)
    if mode == 'real-estate' and await check_if_property_listing_query(question):
        print("🏠 İlan araması tespit edildi. Akıllı yanıtlama süreci başlatılıyor...")
        response_data["is_listing_response"] = True
        
        listings = await hybrid_search_listings(question)
        listings_summary = _format_listings_for_gpt(listings)
        
        system_prompt = SYSTEM_PROMPTS["real-estate"]
        
        # GPT-4o'ya hem arama sonuçlarını hem de nasıl davranacağını anlatan yeni bir görev veriyoruz.
        new_user_content = f"""Kullanıcının orijinal sorusu: "{question}"

Veritabanı aramam sonucunda şu ilan özetlerini buldum (eğer 'İlan bulunamadı' yazıyorsa, sonuç boştur):
---
{listings_summary}
---
Şimdi, bir Gayrimenkul Uzmanı olarak, bu bilgileri kullanarak kullanıcıya nihai bir cevap oluştur.
- Eğer ilan bulunduysa, "Önce Ara, Sonra Sor" kuralına uy: İlanları şık bir HTML listesi (<ul style='...'><li>...</li></ul>) formatında sun ve aynı zamanda aramayı daraltmak için akıllıca bir soru sor (örneğin eksik kriteri iste: bütçe, oda sayısı vb.). PDF butonu EKLEME. Final HTML cevabı formatlanmış ilan listesi ve senin eklediğin konuşma metnini içermelidir.
- Eğer ilan bulunamadıysa, kullanıcıya kibarca durumu bildir ve arama kriterlerini değiştirmesi için nazikçe önerilerde bulun.
Cevabın tamamı akıcı bir metin ve/veya HTML formatında olmalı."""

        try:
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": new_user_content}
            ]
            resp = await openai_client.chat.completions.create(
                model="gpt-4o",
                messages=messages,
                temperature=0.5,
                max_tokens=2048
            )
            response_data["reply"] = resp.choices[0].message.content.strip()
            return response_data
        except Exception as e:
            print(f"❌ İlan yanıtlama GPT hatası: {e}")
            response_data["reply"] = "<p>🔍 Üzgünüm, belirttiğiniz kriterlere uygun bir ilan bulamadım. Lütfen arama kriterlerinizi değiştirerek tekrar deneyin.</p>"
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
