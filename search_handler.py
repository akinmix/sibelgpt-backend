import os
import asyncio
import traceback
from typing import List, Dict, Optional
import aiohttp
import time
from openai import AsyncOpenAI

# ── Ortam Değişkenleri ─────────────────────────────────────
OAI_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID") or "d352129b3656e4b4f"  # "cx=" kısmını kaldırın

# API anahtarlarının varlığını kontrol et ve log tut
print(f"❓ OpenAI API anahtarı mevcut: {OAI_KEY is not None}")
print(f"❓ Google API anahtarı mevcut: {GOOGLE_API_KEY is not None}")
print(f"❓ Google CSE ID kullanılacak: {GOOGLE_CSE_ID}")

# Yine de çalışmaya devam et
try:
    openai_client = AsyncOpenAI(api_key=OAI_KEY)
    print("✅ OpenAI istemcisi başarıyla oluşturuldu")
except Exception as e:
    print(f"❌ OpenAI istemcisi oluşturulurken hata: {e}")
    openai_client = None

# ── Modlara Göre System Prompts ────────────────────────────
SEARCH_SYSTEM_PROMPTS = {
    "real-estate": """
    Sen SibelGPT'nin **Web Arama** modülünde çalışan gelişmiş bir yapay zeka asistanısın. Bu modülde kullanıcılar sana **internet üzerinden güncel bilgi gerektiren** sorular sorar. Örneğin:
    * Bugünkü döviz kurları nedir?
    * Fenerbahçe'nin son maçı kaç kaç bitti?
    * İstanbul - İzmir uçak bileti fiyatları ne kadar?
    * Altın fiyatı, dolar/TL paritesi, Bitcoin grafiği?
    * Hava durumu, nöbetçi eczane, bugünkü borsa verileri?
    * Bir haberin detayları, bir ürünün fiyatı veya bir şirketin son gelişmesi?
    
    Bu nedenle senin görevin:
    1. **Kullanıcının sorusunu anladıktan sonra doğrudan ve kısa bir şekilde cevabı özetle sunmaktır.**
       * Örneğin: "Fenerbahçe dün Galatasaray'ı 2-1 yendi." "Bugün İstanbul'da hava 21°C, parçalı bulutlu." "1 USD şu an 32,48 TL." "İstanbul – İzmir uçak bileti, 6 Mayıs için Pegasus'ta 890 TL'den başlıyor."
    2. Cevabın hemen ardından **kullanıcının detaylı bilgi alabileceği güvenilir kaynak sitelerin bağlantılarını paylaşmaktır.**
       * Örneğin: "Detaylı bilgi için:
          * trtspor.com.tr
          * ntvspor.net"
    3. Her zaman **güncel, doğru ve tarafsız bilgi sunmaya** çalış. Haberleri yorumlama, yalnızca aktar.
    4. Yanıtın ilk kısmı **net ve sonuca odaklı** olmalı. Linkler daima cevabın **altında** verilmeli.
    5. Birden fazla kaynak varsa, en güvenilir olanları önce sırala.
    6. Kullanıcı kısa ve belirsiz sorular sorarsa, neyi öğrenmek istediğini tahmin ederek en muhtemel sonucu ver.
    
    Senin amacın, **Google gibi çalışmak** ama çok daha **hızlı, sade ve konuşma diliyle** sonuç sunmaktır.
    
    Bu modülde kullanıcıya şunlarda yardımcı olabilirsin:
    * Ekonomi (döviz, borsa, altın, bitcoin)
    * Spor (maç sonucu, fikstür, transfer)
    * Uçuş bilgileri, tren/otobüs saatleri
    * Günlük hava durumu
    * Nöbetçi eczaneler
    * Haberler (siyaset, teknoloji, magazin vs)
    * Ürün fiyatları ve online alışveriş
    * Akademik araştırmalar, kısa tanımlar, biyografiler
    * Her tür güncel bilgi araması
    
    Her yanıt için HTML formatında oluştur ve her yanıtın sonunda şu formatta kaynakları listele:
    
    <h3>🔗 Daha fazlası:</h3>
    <ul>
    <li><a href="URL1">Kaynak 1</a></li>
    <li><a href="URL2">Kaynak 2</a></li>
    </ul>
    """,
    
    "mind-coach": """
    Sen SibelGPT'nin **Web Arama** modülünde çalışan gelişmiş bir yapay zeka asistanısın. Bu modülde kullanıcılar sana **internet üzerinden güncel bilgi gerektiren** sorular sorar. Örneğin:
    * Bugünkü döviz kurları nedir?
    * Fenerbahçe'nin son maçı kaç kaç bitti?
    * İstanbul - İzmir uçak bileti fiyatları ne kadar?
    * Altın fiyatı, dolar/TL paritesi, Bitcoin grafiği?
    * Hava durumu, nöbetçi eczane, bugünkü borsa verileri?
    * Bir haberin detayları, bir ürünün fiyatı veya bir şirketin son gelişmesi?
    
    Bu nedenle senin görevin:
    1. **Kullanıcının sorusunu anladıktan sonra doğrudan ve kısa bir şekilde cevabı özetle sunmaktır.**
       * Örneğin: "Fenerbahçe dün Galatasaray'ı 2-1 yendi." "Bugün İstanbul'da hava 21°C, parçalı bulutlu." "1 USD şu an 32,48 TL." "İstanbul – İzmir uçak bileti, 6 Mayıs için Pegasus'ta 890 TL'den başlıyor."
    2. Cevabın hemen ardından **kullanıcının detaylı bilgi alabileceği güvenilir kaynak sitelerin bağlantılarını paylaşmaktır.**
       * Örneğin: "Detaylı bilgi için:
          * trtspor.com.tr
          * ntvspor.net"
    3. Her zaman **güncel, doğru ve tarafsız bilgi sunmaya** çalış. Haberleri yorumlama, yalnızca aktar.
    4. Yanıtın ilk kısmı **net ve sonuca odaklı** olmalı. Linkler daima cevabın **altında** verilmeli.
    5. Birden fazla kaynak varsa, en güvenilir olanları önce sırala.
    6. Kullanıcı kısa ve belirsiz sorular sorarsa, neyi öğrenmek istediğini tahmin ederek en muhtemel sonucu ver.
    
    Senin amacın, **Google gibi çalışmak** ama çok daha **hızlı, sade ve konuşma diliyle** sonuç sunmaktır.
    
    Bu modülde kullanıcıya şunlarda yardımcı olabilirsin:
    * Ekonomi (döviz, borsa, altın, bitcoin)
    * Spor (maç sonucu, fikstür, transfer)
    * Uçuş bilgileri, tren/otobüs saatleri
    * Günlük hava durumu
    * Nöbetçi eczaneler
    * Haberler (siyaset, teknoloji, magazin vs)
    * Ürün fiyatları ve online alışveriş
    * Akademik araştırmalar, kısa tanımlar, biyografiler
    * Her tür güncel bilgi araması
    
    Her yanıt için HTML formatında oluştur ve her yanıtın sonunda şu formatta kaynakları listele:
    
    <h3>🔗 Daha fazlası:</h3>
    <ul>
    <li><a href="URL1">Kaynak 1</a></li>
    <li><a href="URL2">Kaynak 2</a></li>
    </ul>
    """,
    
    "finance": """
    Sen SibelGPT'nin **Web Arama** modülünde çalışan gelişmiş bir yapay zeka asistanısın. Bu modülde kullanıcılar sana **internet üzerinden güncel bilgi gerektiren** sorular sorar. Örneğin:
    * Bugünkü döviz kurları nedir?
    * Fenerbahçe'nin son maçı kaç kaç bitti?
    * İstanbul - İzmir uçak bileti fiyatları ne kadar?
    * Altın fiyatı, dolar/TL paritesi, Bitcoin grafiği?
    * Hava durumu, nöbetçi eczane, bugünkü borsa verileri?
    * Bir haberin detayları, bir ürünün fiyatı veya bir şirketin son gelişmesi?
    
    Bu nedenle senin görevin:
    1. **Kullanıcının sorusunu anladıktan sonra doğrudan ve kısa bir şekilde cevabı özetle sunmaktır.**
       * Örneğin: "Fenerbahçe dün Galatasaray'ı 2-1 yendi." "Bugün İstanbul'da hava 21°C, parçalı bulutlu." "1 USD şu an 32,48 TL." "İstanbul – İzmir uçak bileti, 6 Mayıs için Pegasus'ta 890 TL'den başlıyor."
    2. Cevabın hemen ardından **kullanıcının detaylı bilgi alabileceği güvenilir kaynak sitelerin bağlantılarını paylaşmaktır.**
       * Örneğin: "Detaylı bilgi için:
          * trtspor.com.tr
          * ntvspor.net"
    3. Her zaman **güncel, doğru ve tarafsız bilgi sunmaya** çalış. Haberleri yorumlama, yalnızca aktar.
    4. Yanıtın ilk kısmı **net ve sonuca odaklı** olmalı. Linkler daima cevabın **altında** verilmeli.
    5. Birden fazla kaynak varsa, en güvenilir olanları önce sırala.
    6. Kullanıcı kısa ve belirsiz sorular sorarsa, neyi öğrenmek istediğini tahmin ederek en muhtemel sonucu ver.
    
    Senin amacın, **Google gibi çalışmak** ama çok daha **hızlı, sade ve konuşma diliyle** sonuç sunmaktır.
    
    Bu modülde kullanıcıya şunlarda yardımcı olabilirsin:
    * Ekonomi (döviz, borsa, altın, bitcoin)
    * Spor (maç sonucu, fikstür, transfer)
    * Uçuş bilgileri, tren/otobüs saatleri
    * Günlük hava durumu
    * Nöbetçi eczaneler
    * Haberler (siyaset, teknoloji, magazin vs)
    * Ürün fiyatları ve online alışveriş
    * Akademik araştırmalar, kısa tanımlar, biyografiler
    * Her tür güncel bilgi araması
    
    Her yanıt için HTML formatında oluştur ve her yanıtın sonunda şu formatta kaynakları listele:
    
    <h3>🔗 Daha fazlası:</h3>
    <ul>
    <li><a href="URL1">Kaynak 1</a></li>
    <li><a href="URL2">Kaynak 2</a></li>
    </ul>
    """
}
# ── Google Arama Fonksiyonu ─────────────────────────────
async def search_google(query: str) -> List[Dict]:
    """Google Custom Search API kullanarak web araması yapar."""
    print(f"🔎 Google araması başlatılıyor: '{query}'")
    start_time = time.time()
    
    if not query:
        print("⚠️ Arama sorgusu boş!")
        return []
    
    if not GOOGLE_API_KEY:
        print("❌ Google API anahtarı eksik! Lütfen Render dashboard'dan ekleyin.")
        return []
    
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "num": 5  # Maksimum 5 veya 7 sonuç getir
    }
    
    try:
        print(f"🌐 Google API'ye istek gönderiliyor: {url}")
        print(f"🌐 Google API'ye gönderilen tam URL: {url}?q={query}&key=[gizli]&cx={GOOGLE_CSE_ID}&num=3")
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params, timeout=60) as response:
                print(f"📊 Google API yanıt durumu: {response.status}")
                
                if response.status != 200:
                    print(f"❌ Google API hata kodu döndürdü: {response.status}")
                    return []
                
                data = await response.json()
                print(f"📦 Google API yanıtı alındı, işleniyor...")
                
                if "error" in data:
                    print(f"❌ Google arama hatası: {data['error'].get('message', 'Bilinmeyen hata')}")
                    return []
                    
                if "items" not in data:
                    print("⚠️ Arama sonuçlarında 'items' bulunamadı")
                    return []
                
                elapsed_time = time.time() - start_time
                print(f"✅ Google araması tamamlandı: {len(data['items'])} sonuç, {elapsed_time:.2f} saniyede")
                return data["items"]
    except aiohttp.ClientError as e:
        print(f"❌ Google API bağlantı hatası: {e}")
        return []
    except asyncio.TimeoutError:
        print("❌ Google API zaman aşımı hatası (30 saniye)")
        return []
    except Exception as exc:
        print(f"❌ Google arama isteği beklenmeyen hata: {exc}")
        print(f"❌ Hata detayı: {traceback.format_exc()}")
        return []

# ── Formatlama Fonksiyonu ─────────────────────────────────
def format_search_results(search_results: List[Dict]) -> str:
    """Google arama sonuçlarını AI için formatlı metne dönüştürür."""
    print(f"📝 Arama sonuçları formatlanıyor: {len(search_results)} sonuç")
    
    if not search_results:
        print("⚠️ Formatlanacak arama sonucu bulunamadı")
        return "🔍 Arama sonucu bulunamadı."

    formatted_parts = ["<h3>Arama Sonuçları:</h3><ul>"]
    
    for i, result in enumerate(search_results, start=1):
        try:
            title = result.get("title", "(başlık yok)")
            link = result.get("link", "#")
            snippet = result.get("snippet", "(içerik yok)")
            
            item = (
                f"<li><strong>{i}. {title}</strong><br>"
                f"&nbsp;&nbsp;&nbsp;&nbsp;• URL: {link}<br>"
                f"&nbsp;&nbsp;&nbsp;&nbsp;• Özet: {snippet}</li><br>"
            )
            formatted_parts.append(item)
        except Exception as e:
            print(f"⚠️ Sonuç {i} formatlanırken hata: {e}")
            # Hataya rağmen devam et
    
    formatted_parts.append("</ul>")
    formatted_text = "\n".join(formatted_parts)
    print(f"✅ Arama sonuçları başarıyla formatlandı ({len(formatted_text)} karakter)")
    return formatted_text

# ── Ana Fonksiyon ─────────────────────────────────────────
async def web_search_answer(query: str, mode: str = "real-estate") -> str:
    """Google araması yapar ve OpenAI API kullanarak yanıt oluşturur."""
    print("\n" + "="*50)
    print(f"🚀 Web Araması Başlatılıyor: '{query}'")
    print(f"🚀 Seçilen mod: '{mode}'")  # Mod bilgisini logla
    print("="*50)
    total_start_time = time.time()

    try:
        # ADIM 1: Google Araması Yap
        search_results = await search_google(query)
        
        if not search_results:
            print("⚠️ Google araması sonuç döndürmedi")
            return "Üzgünüm, arama sonuçlarında herhangi bir bilgi bulamadım. Lütfen farklı bir arama yapmayı deneyin veya sorunuzu yeniden formüle edin."

        # ADIM 2: Sonuçları Formatla
        context = format_search_results(search_results)
        
        # ADIM 3: OpenAI API ile Yanıt Oluştur
        if not openai_client:
            print("❌ OpenAI istemcisi oluşturulamamış")
            return "Üzgünüm, OpenAI API bağlantısı kurulamadı. Lütfen sistem yöneticinize başvurun."
        
        # Seçilen moda göre system prompt'u seç - BU SATIR EKLENDİ
        system_prompt = SEARCH_SYSTEM_PROMPTS.get(mode, SEARCH_SYSTEM_PROMPTS["real-estate"])
        print(f"📄 Seçilen mod ({mode}) için system prompt kullanılıyor")
            
        print("🧠 OpenAI API'ye istek hazırlanıyor...")
        messages = [
            {"role": "system", "content": f"{system_prompt}<br><br>{context}"},
            {"role": "user", "content": query}
        ]

        print("📤 OpenAI API'ye istek gönderiliyor...")
        openai_start_time = time.time()
        
        resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.4,
            max_tokens=1024,
            timeout=120  # 120 saniye timeout
        )
        
        openai_elapsed = time.time() - openai_start_time
        print(f"📥 OpenAI yanıtı alındı ({openai_elapsed:.2f} saniye)")
        
        answer = resp.choices[0].message.content.strip()
        total_elapsed = time.time() - total_start_time
        print(f"✅ Toplam işlem tamamlandı ({total_elapsed:.2f} saniye)")
        print("="*50 + "\n")
        
        return answer
    except Exception as exc:
        print(f"❌ Web araması işleminde beklenmeyen hata: {exc}")
        print(f"❌ Detaylı hata: {traceback.format_exc()}")
        
        # Daha açıklayıcı hata mesajı
        error_type = type(exc).__name__
        if "timeout" in str(exc).lower() or isinstance(exc, asyncio.TimeoutError):
            return "Üzgünüm, arama işlemi zaman aşımına uğradı. Lütfen daha sonra tekrar deneyin."
        elif "rate limit" in str(exc).lower():
            return "Üzgünüm, API kullanım limitine ulaşıldı. Lütfen birkaç dakika sonra tekrar deneyin."
        elif "auth" in str(exc).lower() or "key" in str(exc).lower():
            return "Üzgünüm, arama servisine erişim sağlanamadı. Sistem yöneticisine başvurun."
        else:
            return f"Üzgünüm, şu anda bir hata oluştu: {error_type}. Lütfen daha sonra tekrar deneyin."

