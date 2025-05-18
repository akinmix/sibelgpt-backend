# main.py - SibelGPT Backend - v7.0.0 (SABİT VERİLER)
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
from elevenlabs_handler import router as elevenlabs_router

# Supabase import kontrolü
try:
    from supabase import create_client
    from supabase.client import Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

# Ortam değişkenlerini yükle
load_dotenv()

# Dahili modüller
from image_handler import router as image_router
from pdf_handler import router as pdf_router
import ask_handler
import search_handler

# ---- Modeller ----
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
    version="7.0.0",
    description="SibelGPT AI Assistant Backend API - Sabit Veriler"
)

# ---- CORS Middleware ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Static Files ----
if os.path.exists("public"):
    app.mount("/static", StaticFiles(directory="public"), name="static")

# ---- Startup Event ----
@app.on_event("startup")
async def startup_event():
    """Uygulama başlangıcında çalışır"""
    print("\n=== SibelGPT Backend v7.0.0 Başlatılıyor ===")
    
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if SUPABASE_AVAILABLE and supabase_url and supabase_key:
        try:
            app.state.supabase_client = create_client(supabase_url, supabase_key)
            print("✅ Supabase istemcisi oluşturuldu")
        except Exception as e:
            print(f"❌ Supabase hatası: {e}")
            app.state.supabase_client = None
    else:
        app.state.supabase_client = None
    
    print("=== Başlatma Tamamlandı ===\n")

# ---- Dependency ----
async def get_supabase_client(request: Request) -> Optional[Client]:
    return getattr(request.app.state, 'supabase_client', None)

# ---- Router Kaydı ----
app.include_router(image_router, prefix="", tags=["image"])
app.include_router(pdf_router, prefix="", tags=["pdf"])
app.include_router(elevenlabs_router, prefix="", tags=["speech"])

# ---- Ana Endpoint ----
@app.get("/", tags=["meta"])
async def root():
    return {
        "status": "ok",
        "service": "SibelGPT Backend",
        "version": "7.0.0"
    }

# ---- Health Check ----
@app.get("/health", tags=["meta"])
async def health_check(db_client = Depends(get_supabase_client)):
    return {
        "status": "healthy",
        "version": "7.0.0",
        "supabase": db_client is not None
    }

# ---- Chat Endpoint ----
@app.post("/chat", tags=["chat"])
async def chat(payload: ChatRequest, db_client = Depends(get_supabase_client)):
    try:
        answer = await ask_handler.answer_question(payload.question, payload.mode)
        return {"reply": answer}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---- Web Araması Endpoint ----
@app.post("/web-search", tags=["search"])
async def web_search(payload: WebSearchRequest):
    try:
        answer = await search_handler.web_search_answer(payload.question, payload.mode)
        return {"reply": answer}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

# ---- SABİT İSTATİSTİKLER (SQL'den alınan gerçek veriler) ----
@app.get("/statistics/simple", tags=["statistics"])
async def get_simple_statistics():
    """Dashboard istatistikleri - SQL'den alınan sabit veriler"""
    
    return {
        "status": "success",
        "statistics": {
            "genel_ozet": {
                "toplam_ilan": 5047,
                "ortalama_fiyat": 13051170.53,  # Tahminî değer
                "en_cok_ilan_ilce": "Kadıköy"
            },
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

# ---- Dashboard HTML ----
@app.get("/dashboard", tags=["frontend"])
async def serve_dashboard():
    dashboard_path = Path("public") / "dashboard.html"
    
    if dashboard_path.exists():
        return FileResponse(dashboard_path, media_type="text/html")
    
    return JSONResponse(status_code=404, content={"error": "Dashboard bulunamadı"})

# ---- Error Handlers ----
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(status_code=404, content={"error": "Sayfa bulunamadı"})

@app.exception_handler(500)
async def server_error_handler(request, exc):
    return JSONResponse(status_code=500, content={"error": str(exc)})

# ---- Ana Program ----
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
