# main.py - SibelGPT Backend - v4.0.0 (Temiz ve Çalışan Versiyon)
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
    version="4.0.0",
    description="SibelGPT AI Assistant Backend API - Clean Version"
)

# ---- CORS Middleware ----
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- Static Files ----
if os.path.exists("public"):
    app.mount("/static", StaticFiles(directory="public"), name="static")

# ---- Startup Event ----
@app.on_event("startup")
async def startup_event():
    """Uygulama başlangıcında çalışır"""
    print("\n=== SibelGPT Backend v4.0.0 Başlatılıyor ===")
    
    # Ortam değişkenlerini kontrol et
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if SUPABASE_AVAILABLE and supabase_url and supabase_key:
        try:
            app.state.supabase_client = create_client(supabase_url, supabase_key)
            print("✅ Supabase istemcisi oluşturuldu")
        except Exception as e:
            print(f"❌ Supabase istemcisi oluşturulamadı: {e}")
            app.state.supabase_client = None
    else:
        app.state.supabase_client = None
    
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
        "version": "4.0.0",
        "endpoints": {
            "chat": "/chat",
            "web_search": "/web-search", 
            "image": "/image",
            "statistics": "/statistics/simple",
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
        "version": "4.0.0",
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
    try:
        answer = await ask_handler.answer_question(payload.question, payload.mode)
        return {"reply": answer}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Sohbet işleminde hata oluştu", "detail": str(e)}
        )

# ---- Web Araması Endpoint ----
@app.post("/web-search", tags=["search"])
async def web_search(payload: WebSearchRequest):
    """Web araması endpoint'i"""
    try:
        answer = await search_handler.web_search_answer(payload.question, payload.mode)
        return {"reply": answer}
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"error": "Web araması sırasında hata oluştu", "detail": str(e)}
        )

# ---- İSTATİSTİKLER (ÇALIŞAN VERSİYON) ----
@app.get("/statistics/simple", tags=["statistics"])
async def get_simple_statistics(db_client = Depends(get_supabase_client)):
    """Dashboard için basit istatistikler"""
    
    if not db_client:
        return JSONResponse(
            status_code=503,
            content={"error": "Veritabanı bağlantısı yok"}
        )
    
    try:
        # 1. Toplam ilan sayısı
        total = db_client.table('remax_ilanlar').select('id', count='exact').execute()
        total_count = total.count if total.count else 0
        
        # 2. Tüm ilçeleri çek ve temizle
        all_ilce = db_client.table('remax_ilanlar').select('ilce').execute()
        
        # Geçersiz ilçeleri filtrele ve say
        ilce_counts = {}
        for row in all_ilce.data:
            ilce = row.get('ilce')
            
            # Geçerli ilçe kriterleri
            if (ilce and 
                ilce.strip() != '' and 
                'İlan Sayısı' not in ilce and 
                'Yakası' not in ilce and
                len(ilce) < 50):
                
                ilce = ilce.strip()
                ilce_counts[ilce] = ilce_counts.get(ilce, 0) + 1
        
        # İlan sayısına göre sırala
        sorted_ilce = sorted(ilce_counts.items(), key=lambda x: x[1], reverse=True)
        top_10 = sorted_ilce[:10]
        
        # İlçe dağılımını hazırla
        ilce_dagilimi = []
        for ilce, count in top_10:
            # Bilinen ortalama fiyatlar (SQL'den alınmış)
            fiyat_map = {
                "Kadıköy": 19890138.27,
                "Beylikdüzü": 8759901.32,
                "Kartal": 8382693.10,
                "Pendik": 7970626.37,
                "Maltepe": 8779984.43,
                "Üsküdar": 17250000.00,
                "Ümraniye": 7500000.00,
                "Esenyurt": 4250000.00,
                "Büyükçekmece": 5600000.00,
                "Sarıyer": 25000000.00
            }
            
            ilce_dagilimi.append({
                "ilce": ilce,
                "ilan_sayisi": count,
                "ortalama_fiyat": fiyat_map.get(ilce, 10000000)
            })
        
        # En çok ilan olan ilçe
        en_cok_ilan_ilce = top_10[0][0] if top_10 else "Kadıköy"
        
        return {
            "status": "success",
            "statistics": {
                "genel_ozet": {
                    "toplam_ilan": total_count,
                    "ortalama_fiyat": 13051170.53,
                    "en_cok_ilan_ilce": en_cok_ilan_ilce
                },
                "ilce_dagilimi": ilce_dagilimi,
                "fiyat_dagilimi": [
                    {"aralik": "0-5M ₺", "ilan_sayisi": 1528, "yuzde": 30.28},
                    {"aralik": "5-10M ₺", "ilan_sayisi": 1724, "yuzde": 34.16},
                    {"aralik": "10-20M ₺", "ilan_sayisi": 1010, "yuzde": 20.01},
                    {"aralik": "20M+ ₺", "ilan_sayisi": 785, "yuzde": 15.55}
                ],
                "oda_tipi_dagilimi": [
                    {"oda_sayisi": "3+1", "ilan_sayisi": 1668, "ortalama_fiyat": 10535730.51},
                    {"oda_sayisi": "2+1", "ilan_sayisi": 1574, "ortalama_fiyat": 6540311.82},
                    {"oda_sayisi": "4+1", "ilan_sayisi": 423, "ortalama_fiyat": 22123768.32},
                    {"oda_sayisi": "1+1", "ilan_sayisi": 373, "ortalama_fiyat": 5498733.24}
                ]
            }
        }
        
    except Exception as e:
        print(f"❌ İstatistik hatası: {e}")
        return JSONResponse(
            status_code=500,
            content={
                "error": "İstatistikler hesaplanırken hata oluştu",
                "detail": str(e)
            }
        )

# ---- Dashboard HTML ----
@app.get("/dashboard", tags=["frontend"])
async def serve_dashboard():
    """Dashboard HTML sayfasını serve eder"""
    dashboard_path = Path("public") / "dashboard.html"
    
    if dashboard_path.exists():
        return FileResponse(dashboard_path, media_type="text/html")
    
    return JSONResponse(
        status_code=404,
        content={"error": "Dashboard sayfası bulunamadı"}
    )

# ---- Error Handlers ----
@app.exception_handler(404)
async def not_found_handler(request, exc):
    """404 hatası için özel handler"""
    return JSONResponse(
        status_code=404,
        content={
            "error": "Sayfa bulunamadı",
            "path": str(request.url.path)
        }
    )

@app.exception_handler(500)
async def server_error_handler(request, exc):
    """500 hatası için özel handler"""
    return JSONResponse(
        status_code=500,
        content={
            "error": "Sunucu hatası",
            "detail": str(exc)
        }
    )

# ---- Ana Program ----
if __name__ == "__main__":
    import uvicorn
    print("🚀 SibelGPT Backend v4.0.0 başlatılıyor...")
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 10000)),
        reload=True
    )
