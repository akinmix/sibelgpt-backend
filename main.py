# main.py - SibelGPT Backend 
import os
import json
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# Supabase import kontrolü
try:
    from supabase import create_client
    from supabase.client import Client
    SUPABASE_AVAILABLE = True
    print("✅ Supabase paketi başarıyla import edildi.")
except ImportError:
    SUPABASE_AVAILABLE = False
    print("❌ Supabase paketi import edilemedi.")

# Ortam değişkenlerini yükle
load_dotenv()

# Dahili modüller
from image_handler import router as image_router
import ask_handler
import search_handler

# ---- Modeller (Pydantic) ----
class ChatRequest(BaseModel):
    question: str
    mode: str = "real-estate"

class WebSearchRequest(BaseModel):
    question: str
    mode: str = "real-estate"

# ---- FastAPI Uygulaması ----
app = FastAPI(
    title="SibelGPT Backend",
    version="1.8.0",
    description="SibelGPT AI Assistant Backend API"
)

# ---- CORS Middleware ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Static Files (Dashboard için) ----
if os.path.exists("public"):
    app.mount("/static", StaticFiles(directory="public"), name="static")
    print("✅ Static klasör mount edildi")
else:
    print("❌ 'public' klasörü bulunamadı")

# ---- Startup Event ----
@app.on_event("startup")
async def startup_event():
    """Uygulama başlangıcında çalışır"""
    print("\n=== SibelGPT Backend Başlatılıyor ===")
    
    # Ortam değişkenlerini kontrol et
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    openai_key = os.getenv("OPENAI_API_KEY")
    google_key = os.getenv("GOOGLE_API_KEY")
    
    print(f"✓ Supabase URL: {'VAR' if supabase_url else 'YOK'}")
    print(f"✓ Supabase Key: {'VAR' if supabase_key else 'YOK'}")
    print(f"✓ OpenAI Key: {'VAR' if openai_key else 'YOK'}")
    print(f"✓ Google Key: {'VAR' if google_key else 'YOK'}")
    
    # Supabase bağlantısını kur
    if SUPABASE_AVAILABLE and supabase_url and supabase_key:
        try:
            app.state.supabase_client = create_client(supabase_url, supabase_key)
            print("✅ Supabase istemcisi oluşturuldu")
            
            # Bağlantı testi
            try:
                test_result = app.state.supabase_client.table('remax_ilanlar').select('id').limit(1).execute()
                print("✅ Supabase bağlantı testi başarılı")
            except Exception as e:
                print(f"⚠️ Supabase bağlantı testi hatası: {e}")
        except Exception as e:
            print(f"❌ Supabase istemcisi oluşturulamadı: {e}")
            app.state.supabase_client = None
    else:
        app.state.supabase_client = None
        print("⚠️ Supabase bağlantısı kurulamadı")
    
    # Dosya yapısını kontrol et
    print(f"\n📁 Çalışma dizini: {os.getcwd()}")
    print(f"📁 Dosyalar: {os.listdir('.')}")
    if os.path.exists('public'):
        print(f"📁 Public klasörü: {os.listdir('public')}")
    
    print("=== Başlatma Tamamlandı ===\n")

# ---- Dependency ----
async def get_supabase_client(request: Request) -> Optional[Client]:
    """Supabase istemcisini döndürür"""
    return getattr(request.app.state, 'supabase_client', None)

# ---- Router Kaydı ----
app.include_router(image_router, prefix="", tags=["image"])

# ---- Ana Endpoint ----
@app.get("/", tags=["meta"])
async def root():
    """API ana endpoint"""
    return {
        "status": "ok",
        "service": "SibelGPT Backend",
        "version": "1.8.0",
        "endpoints": {
            "chat": "/chat",
            "web_search": "/web-search",
            "image": "/image",
            "statistics": "/statistics/dashboard",
            "dashboard": "/dashboard",
            "health": "/health"
        }
    }

# ---- Health Check ----
@app.get("/health", tags=["meta"])
async def health_check(db_client = Depends(get_supabase_client)):
    """Servis sağlık kontrolü"""
    return {
        "status": "healthy",
        "version": "1.8.0",
        "timestamp": datetime.now().isoformat(),
        "services": {
            "supabase": db_client is not None,
            "openai": bool(os.getenv("OPENAI_API_KEY")),
            "google": bool(os.getenv("GOOGLE_API_KEY"))
        }
    }

# ---- Chat Endpoint ----
@app.post("/chat", tags=["chat"])
async def chat(payload: ChatRequest, db_client = Depends(get_supabase_client)):
    """AI sohbet endpoint'i"""
    print(f"📨 Chat isteği: {payload.question[:50]}... (mod: {payload.mode})")
    
    try:
        answer = await ask_handler.answer_question(payload.question, payload.mode)
        return {"reply": answer}
    except Exception as e:
        print(f"❌ Chat hatası: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Sohbet işleminde hata oluştu", "detail": str(e)}
        )

# ---- Web Araması Endpoint ----
@app.post("/web-search", tags=["search"])
async def web_search(payload: WebSearchRequest):
    """Web araması endpoint'i"""
    print(f"🔍 Web arama isteği: {payload.question[:50]}... (mod: {payload.mode})")
    
    try:
        answer = await search_handler.web_search_answer(payload.question, payload.mode)
        return {"reply": answer}
    except Exception as e:
        print(f"❌ Web arama hatası: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Web araması sırasında hata oluştu", "detail": str(e)}
        )

# ---- Dashboard İstatistikleri ----
@app.get("/statistics/dashboard", tags=["statistics"])
async def get_dashboard_statistics(db_client = Depends(get_supabase_client)):
    """Dashboard istatistiklerini döndürür"""
    print("📊 Dashboard istatistikleri istendi")
    
    if not db_client:
        return JSONResponse(
            status_code=503,
            content={"error": "Veritabanı bağlantısı yok"}
        )
    
    try:
        # RPC fonksiyonunu çağır - params parametresi ile
        print("🔄 Supabase RPC çağrısı: get_dashboard_")
        
        # NOT: params parametresi Supabase Python SDK'da zorunlu
        result = db_client.rpc('get_dashboard_statistics', params={}).execute()
        
        print(f"✅ RPC yanıtı alındı: {type(result.data)}")
        
        if result.data:
            # Veri bir liste ise ilk elemanı al
            data = result.data
            if isinstance(data, list) and len(data) > 0:
                data = data[0]
            
            # String JSON ise parse et
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    pass
            
            return {
                "status": "success",
                "statistics": data
            }
        else:
            return JSONResponse(
                status_code=404,
                content={"error": "Veri bulunamadı"}
            )
            
    except Exception as e:
        print(f"❌ Dashboard istatistik hatası: {e}")
        import traceback
        print(traceback.format_exc())
        
        return JSONResponse(
            status_code=500,
            content={
                "error": "İstatistikler alınırken hata oluştu",
                "detail": str(e),
                "type": type(e).__name__
            }
        )

# ---- Test İstatistikleri (Backup) ----
@app.get("/statistics/test", tags=["statistics"])
async def test_statistics():
    """Test amaçlı sabit istatistikler"""
    return {
        "status": "success",
        "statistics": {
            "genel_ozet": {
                "toplam_ilan": 5047,
                "ortalama_fiyat": 13051170.53,
                "en_cok_ilan_ilce": "Kadıköy"
            },
            "ilce_dagilimi": [
                {"ilce": "Kadıköy", "ilan_sayisi": 405, "ortalama_fiyat": 19890138.27},
                {"ilce": "Beylikdüzü", "ilan_sayisi": 304, "ortalama_fiyat": 8759901.32}
            ],
            "fiyat_dagilimi": [
                {"aralik": "5-10M ₺", "ilan_sayisi": 1724, "yuzde": 34.16},
                {"aralik": "0-5M ₺", "ilan_sayisi": 1528, "yuzde": 30.28}
            ],
            "oda_tipi_dagilimi": [
                {"oda_sayisi": "3+1", "ilan_sayisi": 1668, "ortalama_fiyat": 10535730.51},
                {"oda_sayisi": "2+1", "ilan_sayisi": 1574, "ortalama_fiyat": 6540311.82}
            ]
        }
    }

# ---- Dashboard HTML ----
@app.get("/dashboard", tags=["frontend"])
async def serve_dashboard():
    """Dashboard HTML sayfasını serve eder"""
    print("🖥️ Dashboard sayfası istendi")
    
    # Farklı yolları dene
    possible_paths = [
        "public/dashboard.html",
        "./public/dashboard.html",
        Path("public") / "dashboard.html",
        Path(".") / "public" / "dashboard.html"
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"✅ Dashboard bulundu: {path}")
            return FileResponse(path, media_type="text/html")
    
    # Static mount üzerinden dene
    if os.path.exists("public/dashboard.html"):
        print("✅ Dashboard static üzerinden yönlendiriliyor")
        return RedirectResponse(url="/static/dashboard.html")
    
    # Dosya bulunamadı - detaylı hata bilgisi
    current_dir = os.getcwd()
    files_in_root = os.listdir(current_dir)
    
    public_info = {
        "exists": os.path.exists("public"),
        "files": os.listdir("public") if os.path.exists("public") else []
    }
    
    return JSONResponse(
        status_code=404,
        content={
            "error": "Dashboard sayfası bulunamadı",
            "current_directory": current_dir,
            "root_files": files_in_root,
            "public_directory": public_info,
            "tried_paths": [str(p) for p in possible_paths]
        }
    )

# ---- Error Handlers ----
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """404 hatası için özel handler"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Sayfa bulunamadı",
            "path": str(request.url.path),
            "available_endpoints": [
                "/", "/health", "/chat", "/web-search", 
                "/image", "/statistics/dashboard", "/dashboard"
            ]
        }
    )

@app.exception_handler(500)
async def server_error_handler(request, exc):
    """500 hatası için özel handler"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Sunucu hatası",
            "detail": str(exc),
            "type": type(exc).__name__
        }
    )

# ---- Ana Program ----
if __name__ == "__main__":
    import uvicorn
    print("🚀 SibelGPT Backend başlatılıyor...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 10000)),
        reload=True
    )
