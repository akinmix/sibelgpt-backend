# main.py (CORS allow_methods Güncellemesi)

import os
from typing import Union
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware # CORSMiddleware import edildiğinden emin olun
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client

# ... (load_dotenv, URL/KEY alma, supabase_client oluşturma kısmı aynı) ...
if SUPABASE_URL and SUPABASE_KEY:
    try:
      supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
      print(f"DEBUG: Supabase client oluşturuldu. Tipi: {type(supabase_client)}")
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
    version="1.4.3", # Versiyonu güncelledim (CORS OPTIONS fix)
)

# ---- CORS AYARINI GÜNCELLEME ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.sibelgpt.com", "https://sibelgpt.com", "http://localhost:3000"],
    # allow_methods=["*"], # Eski hali
    allow_methods=["GET", "POST", "OPTIONS", "*"], # YENİ HALİ - OPTIONS açıkça eklendi
    allow_headers=["*"],
)
# ----------------------------------

# ... (Dosyanın geri kalanı aynı: get_supabase_client, image_router, chat, root, startup_event) ...

async def get_supabase_client() -> Union["AsyncClient", None]:
    return supabase_client
app.include_router(image_router, prefix="", tags=["image"])

@app.post("/chat", tags=["chat"])
async def chat(
    payload: ChatRequest,
    db_client: Union["AsyncClient", None] = Depends(get_supabase_client)
):
    print(f"DEBUG: /chat endpoint'ine istek alındı. Soru: {payload.question}")
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
    if supabase_client:
        print(f"✅ Supabase istemcisi startup'ta mevcut. Tipi: {type(supabase_client)}")
    else:
        print("❌ Supabase istemcisi startup'ta mevcut değil.")
