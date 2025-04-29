# main.py (String Literal Type Hint Kullanımı)

import os
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# ---- Sadece create_client import ediliyor ----
from supabase import create_client
# from supabase.lib.client_async import AsyncClient # BU SATIR TAMAMEN KALDIRILDI
# -------------------------------------------

# ---- .env dosyasını yükle ----
load_dotenv()

# ---- Ortam Değişkenlerini Al ----
SUPABASE_URL: str | None = os.environ.get("SUPABASE_URL")
SUPABASE_KEY: str | None = os.environ.get("SUPABASE_KEY")

# ---- Supabase İstemcisini Oluştur ----
# AsyncClient importu kaldırıldığı için tip hint'i string olarak yazılacak
# VEYA hiç tip hint kullanılmayacak. create_client'ın doğru tipi döndürdüğünü varsayıyoruz.
supabase_client: "AsyncClient" | None = None # String literal tip hint'i
if SUPABASE_URL and SUPABASE_KEY:
    try:
      # create_client'ın asenkron istemci döndürmesini umuyoruz.
      supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
      print(f"❌ Supabase istemcisi oluşturulurken hata: {e}")
      supabase_client = None
else:
    print("⚠️ Uyarı: SUPABASE_URL ve SUPABASE_KEY ortam değişkenleri bulunamadı. Supabase bağlantısı kurulamayacak.")


# ---- Dahili modüller ----
from image_handler import router as image_router
import ask_handler

class ChatRequest(BaseModel):
    question: str


app = FastAPI(
    title="SibelGPT Backend",
    version="1.3.0", # Versiyonu güncelledim (string literal hint)
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.sibelgpt.com", "https://sibelgpt.com", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Supabase İstemcisini Sağlama ----
# Tip hint'i string literal olarak güncellendi
async def get_supabase_client() -> "AsyncClient" | None:
    """
    Supabase istemcisini endpoint fonksiyonlarına enjekte eder.
    """
    return supabase_client

# ---- ROUTE KAYDI ----
app.include_router(image_router, prefix="", tags=["image"])


@app.post("/chat", tags=["chat"])
async def chat(
    payload: ChatRequest,
    # Tip hint'i string literal olarak güncellendi
    db_client: "AsyncClient" | None = Depends(get_supabase_client)
):
    """
    Frontend'den gelen soruyu cevaplar. Supabase bağlantısını kullanır.
    """
    if db_client is None:
         print("Supabase istemcisi yok veya başlatılamadı.")
         return {"reply": "❌ Veritabanı bağlantısı kurulamadığı için cevap verilemiyor."}

    answer = await ask_handler.answer_question(payload.question, db_client)
    return {"reply": answer}

@app.get("/", tags=["meta"])
async def root():
    """ Health check """
    return {"status": "ok"}

# Uygulama başlatılırken kontrol mesajı
@app.on_event("startup")
async def startup_event():
    if supabase_client:
        print("✅ Supabase istemcisi başarıyla başlatıldı (veya başlatılmaya çalışıldı).")
    else:
        print("❌ Supabase istemcisi başlatılamadı.")
