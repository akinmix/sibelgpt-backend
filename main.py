# main.py (Async Client Init in Startup Event - Google araması entegrasyonu eklendi)
import os
from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# --- create_async_client ve AsyncClient import denemesi ---
try:
    from supabase import create_async_client, AsyncClient
    print("DEBUG: create_async_client, AsyncClient imported from supabase.")
    SUPABASE_ASYNC_AVAILABLE = True
except ImportError:
    print("DEBUG: Failed to import async components from supabase.")
    AsyncClient = None # Tip kontrolü için None ata
    SUPABASE_ASYNC_AVAILABLE = False

load_dotenv()

# ---- Dahili modüller ----
from image_handler import router as image_router
import ask_handler
import search_handler  # YENİ: Web araması modülü importu

class ChatRequest(BaseModel):
    question: str
    mode: str = "real-estate"  # YENİ: Varsayılan mod eklendi

# YENİ: Web araması isteği modeli
class WebSearchRequest(BaseModel):
    question: str
    mode: str = "real-estate"

app = FastAPI(
    title="SibelGPT Backend",
    version="1.7.0", # Versiyon (Google Arama entegrasyonu eklendi)
)

# ---- CORS Middleware ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.sibelgpt.com", "https://sibelgpt.com", "http://localhost:3000"],
    allow_methods=["GET", "POST", "OPTIONS", "*"],
    allow_headers=["*"],
)

# ---- Lifespan Event for Async Initialization ----
@app.on_event("startup")
async def startup_event():
    print("DEBUG: Startup event started.")
    supabase_url = os.environ.get("SUPABASE_URL")
    supabase_key = os.environ.get("SUPABASE_KEY")
    # app.state üzerine bir attribute tanımlayarak istemciyi sakla
    app.state.supabase_client = None

    if supabase_url and supabase_key and SUPABASE_ASYNC_AVAILABLE:
        print("DEBUG: Async Supabase client oluşturulmaya çalışılıyor...")
        try:
            # --- ASIL DÜZELTME: await ile çağır ---
            client: AsyncClient | None = await create_async_client(supabase_url, supabase_key)
            # --------------------------------------
            app.state.supabase_client = client # Oluşturulan istemciyi app.state'e ata
            if app.state.supabase_client:
                 print(f"✅ Supabase async client oluşturuldu ve app.state'e atandı. Tipi: {type(app.state.supabase_client)}")
            else:
                 print("❌ Supabase async client oluşturuldu ama None döndü?") # Beklenmedik durum
        except Exception as e:
            print(f"❌ Supabase async client oluşturulurken startup'ta hata: {e}")
            app.state.supabase_client = None # Hata durumunda None yap
    elif not SUPABASE_ASYNC_AVAILABLE:
         print("❌ Hata: Supabase async components import edilemediği için istemci oluşturulamıyor.")
    else:
        print("⚠️ Uyarı: SUPABASE_URL veya SUPABASE_KEY ortam değişkenleri bulunamadı...")

# ---- Supabase İstemcisini Sağlama (Retrieve from app.state) ----
async def get_supabase_client(request: Request) -> AsyncClient | None:
    # request objesi üzerinden app.state'e erişip istemciyi al
    # Eğer startup'ta None olarak kaldıysa None dönecektir.
    if hasattr(request.app.state, 'supabase_client'):
        return request.app.state.supabase_client
    return None

# ---- ROUTE KAYDI ----
app.include_router(image_router, prefix="", tags=["image"])

@app.post("/chat", tags=["chat"])
async def chat(
    payload: ChatRequest,
    # get_supabase_client şimdi request.app.state'ten alacak
    db_client: AsyncClient | None = Depends(get_supabase_client)
):
    print(f"DEBUG: /chat endpoint'ine istek alındı. Soru: {payload.question}, Mod: {payload.mode}")
    if db_client is None:
         print("Supabase istemcisi (app.state üzerinden) alınamadı veya None.")
         return {"reply": "❌ Veritabanı bağlantısı kurulamadığı için cevap verilemiyor."}
    # Artık db_client GERÇEK AsyncClient objesi olmalı
    answer = await ask_handler.answer_question(payload.question)
    return {"reply": answer}

# YENİ: Web Araması Endpoint'i
@app.post("/web-search", tags=["search"])
async def web_search(payload: WebSearchRequest):
    print(f"DEBUG: /web-search endpoint'ine istek alındı. Soru: {payload.question}, Mod: {payload.mode}")
    answer = await search_handler.web_search_answer(payload.question)
    return {"reply": answer}

@app.get("/", tags=["meta"])
async def root():
    return {"status": "ok", "version": "1.7.0"}
