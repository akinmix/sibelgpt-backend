# main.py (Düzeltilmiş ve Güncellenmiş Versiyon)
import os
from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv
from pathlib import Path

# Import dosya hatalarını azaltmak için
try:
    from supabase import create_client
    from supabase.client import Client
    SUPABASE_AVAILABLE = True 
    print("DEBUG: supabase paketi başarıyla import edildi.")
except ImportError:
    SUPABASE_AVAILABLE = False
    print("DEBUG: supabase paketi import edilemedi.")

load_dotenv()

# ---- Dahili modüller ----
from image_handler import router as image_router
import ask_handler
import search_handler  # Web arama modülü importu

# ---- Modeller (Pydantic) ----
class ChatRequest(BaseModel):
    question: str
    mode: str = "real-estate"  # Varsayılan mod

class WebSearchRequest(BaseModel):
    question: str
    mode: str = "real-estate"

# ---- FastAPI Uygulaması ----
app = FastAPI(
    title="SibelGPT Backend",
    version="1.8.0", # Dashboard eklendi
)

# ---- CORS Middleware ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://www.sibelgpt.com", 
        "https://sibelgpt.com", 
        "http://localhost:3000",
        "http://localhost:8000",
        "http://localhost:10000",  # Dashboard için
        "*"  # Development için - production'da kaldırın
    ],
    allow_methods=["GET", "POST", "OPTIONS", "*"],
    allow_headers=["*"],
)

# ---- Lifespan Event ----
@app.on_event("startup")
async def startup_event():
    print("DEBUG: Startup event başlıyor.")
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    # Basitleştirilmiş Supabase istemci yönetimi
    if SUPABASE_AVAILABLE and supabase_url and supabase_key:
        try:
            app.state.supabase_client = create_client(supabase_url, supabase_key)
            print("✅ Supabase istemcisi oluşturuldu")
        except Exception as e:
            print(f"❌ Supabase istemcisi oluşturulurken hata: {e}")
            app.state.supabase_client = None
    else:
        app.state.supabase_client = None
        print("⚠️ Supabase istemci oluşturulamadı: Paket veya ortam değişkenleri eksik")
    
    # Google API anahtarı kontrolü
    if not os.environ.get("GOOGLE_API_KEY"):
        print("⚠️ GOOGLE_API_KEY ortam değişkeni eksik - Web araması çalışmayabilir")

# ---- Supabase İstemcisini Sağlama ----
async def get_supabase_client(request: Request):
    """Supabase istemcisini döndürür"""
    if hasattr(request.app.state, 'supabase_client'):
        return request.app.state.supabase_client
    return None

# ---- ROUTE KAYDI ----
app.include_router(image_router, prefix="", tags=["image"])

# ---- Ana Endpoint ----
@app.get("/", tags=["meta"])
async def root():
    return {"status": "ok", "version": "1.8.0", "service": "SibelGPT Backend"}

# ---- Chat Endpoint ----
@app.post("/chat", tags=["chat"])
async def chat(
    payload: ChatRequest,
    db_client = Depends(get_supabase_client)
):
    print(f"DEBUG: /chat endpoint'ine istek alındı. Soru: {payload.question}")
    print(f"DEBUG: İstek modu: {payload.mode}")
    # Mode parametresini geçirerek düzeltildi
    answer = await ask_handler.answer_question(payload.question, payload.mode)
    return {"reply": answer}

# ---- Web Araması Endpoint ----
@app.post("/web-search", tags=["search"])
async def web_search(payload: WebSearchRequest):
    print(f"DEBUG: /web-search endpoint'ine istek alındı. Soru: {payload.question}")
    print(f"DEBUG: İstek modu: {payload.mode}")
    answer = await search_handler.web_search_answer(payload.question, payload.mode)
    return {"reply": answer}

# ---- Dashboard İstatistik Endpoint ----
@app.get("/statistics/dashboard", tags=["statistics"])
async def get_dashboard_statistics(
    db_client = Depends(get_supabase_client)
):
    """Dashboard istatistiklerini döndürür"""
    print("DEBUG: Dashboard istatistikleri istendi")
    
    if not db_client:
        return JSONResponse(
            status_code=500,
            content={"error": "Veritabanı bağlantısı yok"}
        )
    
    try:
        # Supabase RPC fonksiyonunu çağır
        result = db_client.rpc('get_dashboard_statistics').execute()
        
        if result.data:
            print("✅ Dashboard istatistikleri başarıyla alındı")
            return {
                "status": "success",
                "statistics": result.data
            }
        else:
            print("❌ Dashboard istatistikleri boş döndü")
            return JSONResponse(
                status_code=404,
                content={"error": "Veri alınamadı"}
            )
            
    except Exception as e:
        print(f"❌ Dashboard istatistikleri alınırken hata: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# ---- Dashboard HTML Sayfası ----
@app.get("/dashboard", tags=["frontend"])
async def serve_dashboard():
    """Dashboard HTML sayfasını döndürür"""
    dashboard_path = Path("public/dashboard.html")
    
    print(f"DEBUG: Dashboard dosya yolu: {dashboard_path}")
    print(f"DEBUG: Dosya var mı: {dashboard_path.exists()}")
    
    if dashboard_path.exists():
        return FileResponse(
            dashboard_path,
            media_type="text/html",
            headers={
                "Cache-Control": "no-cache, no-store, must-revalidate",
                "Pragma": "no-cache",
                "Expires": "0"
            }
        )
    else:
        return JSONResponse(
            status_code=404,
            content={
                "error": "Dashboard sayfası bulunamadı", 
                "path": str(dashboard_path.absolute())
            }
        )

# ---- Health Check Endpoint ----
@app.get("/health", tags=["meta"])
async def health_check(db_client = Depends(get_supabase_client)):
    """Servis sağlık kontrolü"""
    health_status = {
        "status": "healthy",
        "version": "1.8.0",
        "services": {
            "supabase": db_client is not None,
            "openai": os.environ.get("OPENAI_API_KEY") is not None,
            "google": os.environ.get("GOOGLE_API_KEY") is not None
        }
    }
    return health_status

# ---- Error Handlers ----
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint bulunamadı", "path": str(request.url.path)}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={"error": "Sunucu hatası", "detail": str(exc)}
    )

# ---- Main Çalıştırma ----
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=10000, reload=True)
