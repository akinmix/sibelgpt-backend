# main.py (create_async_client Kullanımı)
import os
# from typing import Union # Kaldırıldı
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# --- create_async_client ve AsyncClient import denemesi ---
try:
    # Doğrudan supabase'ten import etmeyi dene
    from supabase import create_async_client, AsyncClient
    print("DEBUG: create_async_client ve AsyncClient başarıyla supabase'ten import edildi.")
    SUPABASE_ASYNC_AVAILABLE = True
except ImportError:
    try:
        # Eski yolu dene (düşük ihtimal ama kontrol edelim)
        from supabase.lib.client_async import AsyncClient
        # create_async_client'ın yeri farklı olabilir, şimdilik olmadığını varsayalım
        # Eğer sadece AsyncClient importu çalışırsa, bu da bir ilerlemedir.
        # Ama create_async_client olmadan nasıl oluşturulur bilmemiz gerekir.
        # Şimdilik eğer create_async_client üst seviyede yoksa, kullanılamıyor kabul edelim.
        print("DEBUG: AsyncClient supabase.lib.client_async'ten import edildi, ancak create_async_client bulunamadı.")
        SUPABASE_ASYNC_AVAILABLE = False # create_async_client yoksa kullanılamaz
    except ImportError:
        print("DEBUG: Ne create_async_client ne de AsyncClient bilinen yollardan import edilemedi.")
        AsyncClient = None # Tip kontrolü için None ata
        SUPABASE_ASYNC_AVAILABLE = False

load_dotenv()

# Gerçek tip hintini kullan (eğer import başarılıysa)
supabase_client: AsyncClient | None = None

supabase_url_from_env = os.environ.get("SUPABASE_URL")
supabase_key_from_env = os.environ.get("SUPABASE_KEY")
print(f"DEBUG: Ortam değişkenleri okundu...")

if supabase_url_from_env and supabase_key_from_env and SUPABASE_ASYNC_AVAILABLE:
    try:
        # --- create_async_client KULLAN ---
        supabase_client = create_async_client(supabase_url_from_env, supabase_key_from_env)
        # ----------------------------------
        print(f"DEBUG: Supabase async client oluşturuldu. Tipi: {type(supabase_client)}")
    except Exception as e:
        print(f"❌ Supabase async client oluşturulurken hata: {e}")
        supabase_client = None
elif not SUPABASE_ASYNC_AVAILABLE:
     print("❌ Hata: Supabase async client oluşturma fonksiyonu/tipi import edilemedi.")
     supabase_client = None
else:
    print("⚠️ Uyarı: SUPABASE_URL veya SUPABASE_KEY ortam değişkenleri bulunamadı...")
    supabase_client = None

# ---- Dahili modüller ----
from image_handler import router as image_router
import ask_handler

class ChatRequest(BaseModel):
    question: str

app = FastAPI(
    title="SibelGPT Backend",
    version="1.5.0", # Versiyon (create_async_client denemesi)
)

# ---- CORS AYARI ---- (OPTIONS dahil haliyle kalsın)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.sibelgpt.com", "https://sibelgpt.com", "http://localhost:3000"],
    allow_methods=["GET", "POST", "OPTIONS", "*"],
    allow_headers=["*"],
)

# ---- Supabase İstemcisini Sağlama ----
async def get_supabase_client() -> AsyncClient | None: # Gerçek tipi kullan
    return supabase_client

# ---- ROUTE KAYDI ----
app.include_router(image_router, prefix="", tags=["image"])

@app.post("/chat", tags=["chat"])
async def chat(
    payload: ChatRequest,
    db_client: AsyncClient | None = Depends(get_supabase_client) # Gerçek tipi kullan
):
    print(f"DEBUG: /chat endpoint'ine istek alındı. Soru: {payload.question}") # Bu logu koruyalım
    if db_client is None:
         print("Supabase istemcisi yok veya başlatılamadı...")
         return {"reply": "❌ Veritabanı bağlantısı kurulamadığı için cevap verilemiyor."}
    answer = await ask_handler.answer_question(payload.question, db_client)
    return {"reply": answer}

@app.get("/", tags=["meta"])
async def root():
    return {"status": "ok"}

@app.on_event("startup")
async def startup_event():
    if supabase_client: print(f"✅ Supabase async client startup'ta mevcut. Tipi: {type(supabase_client)}")
    else: print("❌ Supabase async client startup'ta mevcut değil veya başlatılamadı.")
