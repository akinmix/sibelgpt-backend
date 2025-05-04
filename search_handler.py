import os
import asyncio
from typing import List, Dict, Optional
import aiohttp
from openai import AsyncOpenAI

# ── Ortam Değişkenleri ─────────────────────────────────────
OAI_KEY = os.getenv("OPENAI_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID") or "cx=d352129b3656e4b4f"

# API anahtarlarının varlığını kontrol et ama hata verme
if not OAI_KEY:
    print("⚠️ OPENAI_API_KEY ortam değişkeni eksik")
if not GOOGLE_API_KEY:
    print("⚠️ GOOGLE_API_KEY ortam değişkeni eksik")

# Yine de çalışmaya devam et
openai_client = AsyncOpenAI(api_key=OAI_KEY)

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
    if not query:
        return []
    
    if not GOOGLE_API_KEY:
        print("❌ Google API anahtarı eksik!")
        return []
    
    url = "https://www.googleapis.com/customsearch/v1"
    params = {
        "q": query,
        "key": GOOGLE_API_KEY,
        "cx": GOOGLE_CSE_ID,
        "num": 5  # Maksimum 5 sonuç getir
    }
    
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                data = await response.json()
                
                if "error" in data:
                    print(f"❌ Google arama hatası: {data['error']['message']}")
                    return []
                    
                if "items" not in data:
                    return []
                    
                return data["items"]
    except Exception as exc:
        print(f"❌ Google arama isteği hatası: {exc}")
        return []

# ── Formatlama Fonksiyonu ─────────────────────────────────
def format_search_results(search_results: List[Dict]) -> str:
    """Google arama sonuçlarını AI için formatlı metne dönüştürür."""
    if not search_results:
        return "🔍 Arama sonucu bulunamadı."

    formatted_parts = ["<h3>Arama Sonuçları:</h3><ul>"]
    
    for i, result in enumerate(search_results, start=1):
        title = result.get("title", "(başlık yok)")
        link = result.get("link", "#")
        snippet = result.get("snippet", "(içerik yok)")
        
        item = (
            f"<li><strong>{i}. {title}</strong><br>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;• URL: {link}<br>"
            f"&nbsp;&nbsp;&nbsp;&nbsp;• Özet: {snippet}</li><br>"
        )
        formatted_parts.append(item)
    
    formatted_parts.append("</ul>")
    return "\n".join(formatted_parts)

# ── Ana Fonksiyon ─────────────────────────────────────────
async def web_search_answer(query: str) -> str:
    """Google araması yapar ve OpenAI API kullanarak yanıt oluşturur."""
    print("↪ Web Araması:", query)

    search_results = await search_google(query)
    context = format_search_results(search_results)

    if not search_results:
        return "Üzgünüm, arama sonuçlarında herhangi bir bilgi bulamadım. Lütfen farklı bir arama yapmayı deneyin veya sorunuzu yeniden formüle edin."

    messages = [
        {"role": "system", "content": f"{SYSTEM_PROMPT}<br><br>{context}"},
        {"role": "user", "content": query}
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
        return "Üzgünüm, şu anda bir hata oluştu. Lütfen daha sonra tekrar deneyin."

# ── Terminalden Test ──────────────────────────────────────
if __name__ == "__main__":
    q = input("Arama: ")
    loop = asyncio.get_event_loop()
    print(loop.run_until_complete(web_search_answer(q)))
