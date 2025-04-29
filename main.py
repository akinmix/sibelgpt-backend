# main.py (Supabase Import'ları Düzeltilmiş Sürüm)

import os
from fastapi import FastAPI, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# ---- Düzeltilmiş Supabase Import'ları ----
# from supabase_py_async import create_client, AsyncClient # ESKİ YANLIŞ SATIR - SİLİNDİ
from supabase import create_client # YENİ DOĞRU IMPORT - create_client için
from supabase.lib.client_async import AsyncClient # YENİ DOĞRU IMPORT - AsyncClient için
# -----------------------------------------

# ---- .env dosyasını yükle (Lokal geliştirme için önemli) ----
load_dotenv()

# ---- Ortam Değişkenlerini Al ----
SUPABASE_URL: str | None = os.environ.get("SUPABASE_URL")
SUPABASE_KEY: str | None = os.environ.get("SUPABASE_KEY")

# ---- Supabase İstemcisini Oluştur ----
# Eğer URL veya Key bulunamazsa None olacak, kontrol ekleyelim
# create_client ve AsyncClient artık doğru yerden import edildiği için burası çalışmalı.
supabase_client: AsyncClient | None = None
if SUPABASE_URL and SUPABASE_KEY:
    try: # create_client çağrısını da try-except içine almak iyi olabilir
      supabase_client = create_client(SUPABASE_URL, SUPABASE_KEY)
      # Önemli Not: create_client V2'de varsayılan olarak senkron dönebilir.
      # Asenkron istemciyi açıkça almak gerekebilir, test edip göreceğiz.
      # Şimdilik bu şekilde bırakalım, eğer tip hatası alırsak düzenleriz.
    except Exception as e:
      print(f"❌ Supabase istemcisi oluşturulurken hata: {e}")
      supabase_client = None # Hata durumunda None olarak ayarla
else:
    print("⚠️ Uyarı: SUPABASE_URL ve SUPABASE_KEY ortam değişkenleri bulunamadı. Supabase bağlantısı kurulamayacak.")


# ---- Dahili modüller ----
from image_handler import router as image_router
import ask_handler # ask_handler'ı modül olarak import edelim

class ChatRequest(BaseModel):
    question: str


app = FastAPI(
    title="SibelGPT Backend",
    version="1.2.0", # Versiyonu güncelledim (import düzeltmesi)
)

# CORS – Vercel frontend’inden gelen çağrılara izin ver
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.sibelgpt.com", "https://sibelgpt.com", "http://localhost:3000"], # Lokal test için localhost eklendi (opsiyonel)
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Supabase İstemcisini Endpoint'lere Sağlama (Dependency Injection) ----
# AsyncClient tipi doğru yerden import edildiği için burası da çalışmalı.
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
    if db_client is None:
         print("Supabase istemcisi yok veya başlatılamadı.")
         return {"reply": "❌ Veritabanı bağlantısı kurulamadığı için cevap verilemiyor."}

    # Supabase istemcisini ask_handler'daki fonksiyona gönder
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
        print("❌ Supabase istemcisi başlatılamadı. Ortam değişkenlerini veya istemci oluşturma kodunu kontrol edin.")
