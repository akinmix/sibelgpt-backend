import os
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# Supabase import kontrolü
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

# Ortam değişkenlerini yükle
load_dotenv()

# Dahili modüller
# NOT: Artık property_search_handler'a ihtiyacımız kalmayacak ama şimdilik dursun.
from image_handler import router as image_router
from pdf_handler import router as pdf_router
from elevenlabs_handler import router as elevenlabs_router
import ask_handler
import search_handler

# ---- Pydantic Modelleri ----
class ChatRequest(BaseModel):
    question: str
    mode: str = "real-estate"
    conversation_history: List[Dict] = []

class WebSearchRequest(BaseModel):
    question: str
    mode: str = "real-estate"

# ---- FastAPI Uygulaması ----
app = FastAPI(
    title="SibelGPT Backend",
    version="8.0.0", # Versiyonu güncelleyelim
    description="SibelGPT AI Assistant Backend API - Hız ve RAG Optimizasyonu"
)

# ---- CORS Middleware ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Production için daha spesifik bir liste kullanmak daha güvenlidir.
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Rate Limiting ----
# Not: Bu yapı tek bir worker'da çalışır. Ölçeklenmiş ortamlar için Redis gibi
# merkezi bir çözüm daha uygun olacaktır.
request_counts = {}

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if "/chat" in request.url.path:
        client_ip = request.client.host if request.client else "unknown"
        current_minute = int(time.time()) // 60
        key = f"{client_ip}:{current_minute}"
        
        request_counts[key] = request_counts.get(key, 0) + 1
        
        if request_counts[key] > 45: # Limiti biraz artıralım
            return JSONResponse(
                status_code=429,
                content={"error": "Çok fazla istek. Lütfen bir dakika bekleyin."}
            )
            
        # Eski kayıtları temizle (2 dakika öncesini sil)
        old_minute = current_minute - 2
        for k in list(request_counts.keys()):
            try:
                if int(k.split(':')[1]) < old_minute:
                    del request_counts[k]
            except (ValueError, IndexError):
                del request_counts[k] # Hatalı formatlı anahtarı sil
    
    response = await call_next(request)
    return response

# ---- Static Files ----
if os.path.exists("public"):
    app.mount("/static", StaticFiles(directory="public"), name="static")

# ---- Uygulama Başlangıç Olayı ----
@app.on_event("startup")
async def startup_event():
    print("\n=== SibelGPT Backend v8.0.0 Başlatılıyor ===")
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY") # Bu bizim anon key'imizdi.
    
    if SUPABASE_AVAILABLE and supabase_url and supabase_key:
        try:
            # state, FastAPI uygulaması boyunca paylaşılacak nesneleri tutar.
            app.state.supabase_client = create_client(supabase_url, supabase_key)
            print("✅ Supabase istemcisi oluşturuldu.")
        except Exception as e:
            print(f"❌ Supabase istemcisi oluşturulamadı: {e}")
            app.state.supabase_client = None
    else:
        print("⚠️ Supabase bilgileri eksik veya kütüphane yüklü değil.")
        app.state.supabase_client = None
    
    print("=== Başlatma Tamamlandı ===\n")

# ---- Router Kaydı ----
app.include_router(image_router, prefix="", tags=["Image Generation"])
app.include_router(pdf_router, prefix="", tags=["PDF Generation"])
app.include_router(elevenlabs_router, prefix="", tags=["Text-to-Speech"])

# ---- Ana ve Sağlık Kontrolü Endpoint'leri ----
@app.get("/", tags=["Meta"])
async def root():
    return {"service": "SibelGPT Backend", "version": "8.0.0", "status": "ok"}

@app.get("/health", tags=["Meta"])
async def health_check():
    return {"status": "healthy", "supabase_available": SUPABASE_AVAILABLE}

# ---- Güvenli Frontend Konfigürasyon Endpoint'i ----
@app.get("/api/config", tags=["Configuration"])
async def get_public_config():
    """Frontend için güvenli konfigürasyon bilgilerini sağlar."""
    return {
        "supabaseUrl": os.getenv("SUPABASE_URL"),
        "supabaseAnonKey": os.getenv("SUPABASE_KEY"),  # Bunun anon key olduğunu teyit ettik.
        "backendUrl": os.getenv("BACKEND_URL", "https://sibelgpt-backend.onrender.com")
    }

# ==========================================================
# ================= CHAT ENDPOINT (GÜNCELLENDİ) ============
# ==========================================================
@app.post("/chat", tags=["Core AI"])
async def chat(payload: ChatRequest):
    """Ana sohbet ve RAG ilan arama endpoint'i."""
    try:
        # ask_handler artık bir sözlük döndürüyor: {"reply": "...", "is_listing_response": ...}
        response_data = await ask_handler.answer_question(
            question=payload.question, 
            mode=payload.mode, 
            conversation_history=payload.conversation_history
        )
        return response_data
    except Exception as e:
        print(f"❌ Chat Endpoint Hatası: {e}")
        import traceback
        traceback.print_exc()
        # Hata durumunda da frontend'in beklediği formatta bir yanıt verelim
        return JSONResponse(
            status_code=500, 
            content={
                "reply": f"Üzgünüm, sunucuda beklenmedik bir hata oluştu. Lütfen daha sonra tekrar deneyin.",
                "is_listing_response": False
            }
        )

# ---- Web Araması Endpoint'i ----
@app.post("/web-search", tags=["Core AI"])
async def web_search(payload: WebSearchRequest):
    """Google üzerinden genel web araması yapar."""
    try:
        answer = await search_handler.web_search_answer(payload.question, payload.mode)
        # Web araması her zaman standart formatta döner
        return {"reply": answer, "is_listing_response": False}
    except Exception as e:
        print(f"❌ Web Search Hatası: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "reply": "Web araması sırasında bir hata oluştu.",
                "is_listing_response": False
            }
        )

# ---- Dashboard ve İstatistikler ----
@app.get("/statistics/simple", tags=["Dashboard"])
async def get_simple_statistics():
    """Dashboard için önceden hesaplanmış (sabit) istatistikleri döndürür."""
    # Bu veriler, ayrı bir script ile periyodik olarak güncellenip
    # bir dosyaya veya veritabanına yazılabilir. Şimdilik sabit.
    return {
        "status": "success",
        "statistics": {
            "genel_ozet": {"toplam_ilan": 5047, "ortalama_fiyat": 13051170.53, "en_cok_ilan_ilce": "Kadıköy"},
            "ilce_dagilimi": [
                {"ilce": "Kadıköy", "ilan_sayisi": 405, "ortalama_fiyat": 19890138.27},
                {"ilce": "Beylikdüzü", "ilan_sayisi": 304, "ortalama_fiyat": 8759901.32},
                {"ilce": "Kartal", "ilan_sayisi": 290, "ortalama_fiyat": 8382693.10},
                {"ilce": "Pendik", "ilan_sayisi": 273, "ortalama_fiyat": 7970626.37},
                {"ilce": "Maltepe", "ilan_sayisi": 257, "ortalama_fiyat": 8779984.43}
            ],
            "fiyat_dagilimi": [
                {"aralik": "0-5M ₺", "ilan_sayisi": 2327, "yuzde": 46.11},
                {"aralik": "5-10M ₺", "ilan_sayisi": 1582, "yuzde": 31.34},
                {"aralik": "10-20M ₺", "ilan_sayisi": 1024, "yuzde": 20.29},
                {"aralik": "20M+ ₺", "ilan_sayisi": 114, "yuzde": 2.26}
            ]
        }
    }

@app.get("/dashboard", include_in_schema=False)
async def serve_dashboard():
    """Dashboard HTML sayfasını sunar."""
    dashboard_path = Path("public/dashboard.html")
    if dashboard_path.exists():
        return FileResponse(dashboard_path)
    return JSONResponse(status_code=404, content={"error": "Dashboard HTML bulunamadı"})

# ---- Program Başlatma ----
if __name__ == "__main__":
    import uvicorn
    # Render PORT ortam değişkenini kullanır. Lokal test için 10000.
    port = int(os.getenv("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
