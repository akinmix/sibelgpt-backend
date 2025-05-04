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

# ── Ayarlar ────────────────────────────────────────────────
SYSTEM_PROMPT = (
    "Sen SibelGPT'sin: Sibel Kazan Midilli tarafından geliştirilen, "
    "Türkiye emlak piyasası (özellikle Remax Sonuç portföyü), numeroloji ve "
    "finans konularında uzman, Türkçe yanıt veren yardımsever bir yapay zeka asistansın.\n\n"
    
    "Aşağıdaki Google arama sonuçlarını kullanarak kullanıcının sorusuna kapsamlı yanıt ver. "
    "Cevabında güncel bilgilere dayanarak en iyi yanıtı oluştur. Kaynakları doğrula ve "
    "bilgilerin doğruluğundan emin ol.\n\n"
    
    "Cevaplarını kısa, net ve samimi tut. İlgili sonuçların özeti şeklinde yanıtla.\n\n"
    
    "Yanıtlarını HTML formatında oluştur. Başlıklar için <h3>, listeler için <ul> ve <li> kullan. "
    "Satır atlamak için <br>, kalın yazı için <strong> kullan. Markdown işaretleri (*, -) kullanma.\n\n"
)

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
        "num": 5  # Maksimum 5 sonuç getir
    }
    
    try:
        print(f"🌐 Google API'ye istek gönderiliyor: {url}")
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
async def web_search_answer(query: str) -> str:
    """Google araması yapar ve OpenAI API kullanarak yanıt oluşturur."""
    print("\n" + "="*50)
    print(f"🚀 Web Araması Başlatılıyor: '{query}'")
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
            
        print("🧠 OpenAI API'ye istek hazırlanıyor...")
        messages = [
            {"role": "system", "content": f"{SYSTEM_PROMPT}<br><br>{context}"},
            {"role": "user", "content": query}
        ]

        print("📤 OpenAI API'ye istek gönderiliyor...")
        openai_start_time = time.time()
        
        resp = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0.7,
            max_tokens=1024,
            timeout=60  # 60 saniye timeout
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

# ── Terminalden Test ──────────────────────────────────────
if __name__ == "__main__":
    try:
        q = input("Arama yapmak istediğiniz sorguyu girin: ")
        print("\n⏳ İşlem sürüyor, lütfen bekleyin...\n")
        
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(web_search_answer(q))
        
        print("\n" + "="*30 + " SONUÇ " + "="*30)
        print(result)
        print("="*67 + "\n")
    except KeyboardInterrupt:
        print("\n\n❌ İşlem kullanıcı tarafından iptal edildi.")
    except Exception as e:
        print(f"\n\n❌ Test sırasında beklenmeyen hata: {e}")
        print(traceback.format_exc())
