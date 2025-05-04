# main.py (Düzeltilmiş Versiyon)
import os
from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from dotenv import load_dotenv

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

class ChatRequest(BaseModel):
    question: str
    mode: str = "real-estate"  # Varsayılan mod

# Web araması isteği modeli
class WebSearchRequest(BaseModel):
    question: str
    mode: str = "real-estate"

app = FastAPI(
    title="SibelGPT Backend",
    version="1.7.0", # Web araması entegrasyonu versiyonu
)

# ---- CORS Middleware ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.sibelgpt.com", "https://sibelgpt.com", "http://localhost:3000"],
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
    # Basitleştirilmiş, sadece None olabilir
    if hasattr(request.app.state, 'supabase_client'):
        return request.app.state.supabase_client
    return None

# ---- ROUTE KAYDI ----
app.include_router(image_router, prefix="", tags=["image"])

@app.post("/chat", tags=["chat"])
async def chat(
    payload: ChatRequest,
    db_client = Depends(get_supabase_client)
):
    print(f"DEBUG: /chat endpoint'ine istek alındı. Soru: {payload.question}")
    answer = await ask_handler.answer_question(payload.question)
    return {"reply": answer}

# Web Araması Endpoint'i
@app.post("/web-search", tags=["search"])
async def web_search(payload: WebSearchRequest):
    print(f"DEBUG: /web-search endpoint'ine istek alındı. Soru: {payload.question}")
    answer = await search_handler.web_search_answer(payload.question)
    return {"reply": answer}

@app.get("/", tags=["meta"])
async def root():
    return {"status": "ok", "version": "1.7.0"}
