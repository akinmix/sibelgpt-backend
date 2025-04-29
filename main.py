# main.py (typing.Union Kullanımı)

import os
from typing import Union # Union import edildi
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client
# AsyncClient import'u hala yok, string literal kullanacağız

load_dotenv()

SUPABASE_URL: str | None = os.environ.get("SUPABASE_URL")
SUPABASE_KEY: str | None = os.environ.get("SUPABASE_KEY")

# ---- Supabase İstemcisini Oluştur ----
# typing.Union kullanarak tip hint'i
supabase_client: Union["AsyncClient", None] = None # Union["...", None] kullanıldı
if SUPABASE_URL and SUPABASE_KEY:
    try:
      supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception as e:
      print(f"❌ Supabase istemcisi oluşturulurken hata: {e}")
      supabase_client = None
else:
    print("⚠️ Uyarı: SUPABASE_URL ve SUPABASE_KEY ortam değişkenleri bulunamadı...")


# ---- Dahili modüller ----
from image_handler import router as image_router
import ask_handler

class ChatRequest(BaseModel):
    question: str


app = FastAPI(
    title="SibelGPT Backend",
    version="1.4.0", # Versiyonu güncelledim (typing.Union)
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.sibelgpt.com", "https://sibelgpt.com", "http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Supabase İstemcisini Sağlama ----
# typing.Union kullanarak tip hint'i
async def get_supabase_client() -> Union["AsyncClient", None]: # Union["...", None] kullanıldı
    """ Supabase istemcisini endpoint fonksiyonlarına enjekte eder. """
    return supabase_client

# ---- ROUTE KAYDI ----
app.include_router(image_router, prefix="", tags=["image"])


@app.post("/chat", tags=["chat"])
async def chat(
    payload: ChatRequest,
    # typing.Union kullanarak tip hint'i
    db_client: Union["AsyncClient", None] = Depends(get_supabase_client) # Union["...", None] kullanıldı
):
    """ Frontend'den gelen soruyu cevaplar. """
    if db_client is None:
         print("Supabase istemcisi yok...")
         return {"reply": "❌ Veritabanı bağlantısı kurulamadığı için cevap verilemiyor."}

    # ask_handler'daki answer_question'a db_client gönderilir
    # ask_handler.py'ın tip hint'i hala "AsyncClient" şeklinde, o şimdilik kalsın.
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
