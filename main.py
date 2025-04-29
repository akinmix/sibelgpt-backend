# main.py (Daha Savunmacı Env Var Okuma)

import os
from typing import Union
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from supabase import create_client

load_dotenv()

# İstemciyi başta None olarak tanımla
supabase_client: Union["AsyncClient", None] = None

# Ortam değişkenlerini oku
# Bu satırlarda NameError OLMAMALI, en fazla None dönerler.
supabase_url_from_env = os.environ.get("SUPABASE_URL")
supabase_key_from_env = os.environ.get("SUPABASE_KEY")
print(f"DEBUG: Ortam değişkenleri okundu. URL: {'Var' if supabase_url_from_env else 'Yok'}, Key: {'Var' if supabase_key_from_env else 'Yok'}") # Ekstra debug

# Sadece iki değişken de BAŞARIYLA okunduysa istemciyi oluşturmayı dene
if supabase_url_from_env and supabase_key_from_env:
    try:
        # Okunan değişkenleri doğrudan kullan
        supabase_client = create_client(supabase_url_from_env, supabase_key_from_env)
        print(f"DEBUG: Supabase client oluşturuldu. Tipi: {type(supabase_client)}")
    except Exception as e:
        print(f"❌ Supabase istemcisi oluşturulurken hata: {e}")
        supabase_client = None # Hata durumunda None olduğundan emin ol
else:
    print("⚠️ Uyarı: SUPABASE_URL veya SUPABASE_KEY ortam değişkenlerinden biri veya ikisi de bulunamadı/okunamadı. Supabase bağlantısı kurulmayacak.")
    supabase_client = None # Eksikse None olduğundan emin ol


# ---- Dahili modüller ----
from image_handler import router as image_router
import ask_handler

class ChatRequest(BaseModel):
    question: str

app = FastAPI(
    title="SibelGPT Backend",
    version="1.4.4", # Versiyonu güncelledim (defensive init)
)

# ---- CORS AYARI ---- (Bir önceki adımdaki gibi kalmalı)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.sibelgpt.com", "https://sibelgpt.com", "http://localhost:3000"],
    allow_methods=["GET", "POST", "OPTIONS", "*"],
    allow_headers=["*"],
)

# ---- Supabase İstemcisini Sağlama ----
async def get_supabase_client() -> Union["AsyncClient", None]:
    """ Supabase istemcisini endpoint fonksiyonlarına enjekte eder. """
    return supabase_client # Globaldeki değeri döndürür

# ---- ROUTE KAYDI ----
app.include_router(image_router, prefix="", tags=["image"])

@app.post("/chat", tags=["chat"])
async def chat(
    payload: ChatRequest,
    db_client: Union["AsyncClient", None] = Depends(get_supabase_client)
):
    print(f"DEBUG: /chat endpoint'ine istek alındı. Soru: {payload.question}")
    if db_client is None:
         print("Supabase istemcisi yok veya başlatılamadı...")
         return {"reply": "❌ Veritabanı bağlantısı kurulamadığı için cevap verilemiyor."}
    answer = await ask_handler.answer_question(payload.question, db_client)
    return {"reply": answer}

@app.get("/", tags=["meta"])
async def root():
    """ Health check """
    return {"status": "ok"}

@app.on_event("startup")
async def startup_event():
    # Bu log, yukarıdaki istemci oluşturma mantığı bittikten sonra çalışacak
    if supabase_client:
        print(f"✅ Supabase istemcisi startup sonunda mevcut. Tipi: {type(supabase_client)}")
    else:
        print("❌ Supabase istemcisi startup sonunda mevcut değil veya başlatılamadı.")
