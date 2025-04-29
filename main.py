# main.py (Client Tipi Debug Log Eklendi)

import os
from typing import Union
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

SUPABASE_URL: str | None = os.environ.get("SUPABASE_URL")
SUPABASE_KEY: str | None = os.environ.get("SUPABASE_KEY")

supabase_client: Union["AsyncClient", None] = None
if SUPABASE_URL and SUPABASE_KEY:
    try:
      supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
      # ---- YENİ EKLENEN DEBUG SATIRI ----
      print(f"DEBUG: Supabase client oluşturuldu. Tipi: {type(supabase_client)}")
      # ------------------------------------
    except Exception as e:
      print(f"❌ Supabase istemcisi oluşturulurken hata: {e}")
      supabase_client = None
else:
    print("⚠️ Uyarı: SUPABASE_URL ve SUPABASE_KEY ortam değişkenleri bulunamadı...")


# ---- Dahili modüller ----
from image_handler import router as image_router
import ask_handler

# ... (Dosyanın geri kalanı aynı) ...

class ChatRequest(BaseModel):
    question: str

app = FastAPI(
    title="SibelGPT Backend",
    version="1.4.1", # Versiyonu güncelledim (debug log)
)

# ... (CORS, get_supabase_client, router, chat, root, startup_event fonksiyonları aynı) ...

async def get_supabase_client() -> Union["AsyncClient", None]:
    return supabase_client

app.include_router(image_router, prefix="", tags=["image"])

@app.post("/chat", tags=["chat"])
async def chat(
    payload: ChatRequest,
    db_client: Union["AsyncClient", None] = Depends(get_supabase_client)
):
    if db_client is None:
         print("Supabase istemcisi yok...")
         return {"reply": "❌ Veritabanı bağlantısı kurulamadığı için cevap verilemiyor."}
    answer = await ask_handler.answer_question(payload.question, db_client)
    return {"reply": answer}

@app.get("/", tags=["meta"])
async def root():
    return {"status": "ok"}

@app.on_event("startup")
async def startup_event():
    # Bu log startup'ta çalışır, istemcinin ilk oluşturulma anındaki tipini görmek daha önemli.
    if supabase_client:
        print(f"✅ Supabase istemcisi startup'ta mevcut. Tipi: {type(supabase_client)}")
    else:
        print("❌ Supabase istemcisi startup'ta mevcut değil.")
