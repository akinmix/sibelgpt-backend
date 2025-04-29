# main.py (Supabase Entegrasyonuna Hazır Sürüm)

import os
from fastapi import FastAPI, Depends # Depends eklendi
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv # .env dosyasını okumak için eklendi
from supabase_py_async import create_client, AsyncClient # Supabase için eklendi

# ---- .env dosyasını yükle (Lokal geliştirme için önemli) ----
load_dotenv()

# ---- Ortam Değişkenlerini Al ----
SUPABASE_URL: str | None = os.environ.get("SUPABASE_URL")
SUPABASE_KEY: str | None = os.environ.get("SUPABASE_KEY")

# ---- Supabase İstemcisini Oluştur ----
# Eğer URL veya Key bulunamazsa None olacak, kontrol ekleyelim
supabase_client: AsyncClient | None = None
if SUPABASE_URL and SUPABASE_KEY:
    supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
else:
    print("⚠️ Uyarı: SUPABASE_URL ve SUPABASE_KEY ortam değişkenleri bulunamadı. Supabase bağlantısı kurulamayacak.")


# ---- Dahili modüller ----
# !!! ask_handler import'unu birazdan değiştireceğiz, şimdilik kalsın
from image_handler import router as image_router
# from ask_handler import answer_question # Bu satırı birazdan güncelleyeceğiz
import ask_handler # ask_handler'ı modül olarak import edelim

class ChatRequest(BaseModel):
    question: str


app = FastAPI(
    title="SibelGPT Backend",
    version="1.1.0", # Versiyonu güncelledim
)

# CORS – Vercel frontend’inden gelen çağrılara izin ver
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.sibelgpt.com", "https://sibelgpt.com", "http://localhost:3000"], # Lokal test için localhost eklendi (opsiyonel)
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Supabase İstemcisini Endpoint'lere Sağlama (Dependency Injection) ----
async def get_supabase_client() -> AsyncClient | None:
    """
    Supabase istemcisini endpoint fonksiyonlarına enjekte eder.
    Eğer istemci başlatılamadıysa None döner.
    """
    return supabase_client

# ---- ROUTE KAYDI ----
# Görsel işleyici router'ı etkilenmeden kalıyor
app.include_router(image_router, prefix="", tags=["image"])


@app.post("/chat", tags=["chat"])
async def chat(
    payload: ChatRequest,
    # Supabase istemcisini Depends ile alıyoruz
    db_client: AsyncClient | None = Depends(get_supabase_client)
):
    """
    Frontend'den gelen soruyu cevaplar. Supabase bağlantısını kullanır.
    """
    # Eğer Supabase istemcisi yoksa hata mesajı dönebiliriz veya normal devam edebiliriz
    if db_client is None:
         # Supabase olmadan eski yöntemle cevap ver (ask_handler'ı buna göre güncelleyeceğiz)
         print("Supabase istemcisi yok, normal cevap deneniyor.")
         # answer = await ask_handler.answer_question_without_rag(payload.question) # Henüz olmayan varsayımsal fonksiyon
         # Şimdilik hata dönelim:
         return {"reply": "❌ Veritabanı bağlantısı kurulamadığı için cevap verilemiyor."}

    # Supabase istemcisini ask_handler'daki fonksiyona gönder
    # ask_handler.answer_question fonksiyonunu da client alacak şekilde güncelleyeceğiz.
    answer = await ask_handler.answer_question(payload.question, db_client)
    return {"reply": answer}

@app.get("/", tags=["meta"])
async def root():
    """
    Render health check ⇒ 200 OK
    """
    return {"status": "ok"}

# Uygulama başlatılırken kontrol mesajı (opsiyonel)
@app.on_event("startup")
async def startup_event():
    if supabase_client:
        print("✅ Supabase istemcisi başarıyla başlatıldı.")
    else:
        print("❌ Supabase istemcisi başlatılamadı. Ortam değişkenlerini kontrol edin.")
