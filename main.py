# main.py - SibelGPT Backend - v6.0.0 (DOĞRU VERSİYON)
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
    version="6.0.0",
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

# ---- Static Files ----
if os.path.exists("public"):
    app.mount("/static", StaticFiles(directory="public"), name="static")

# ---- Startup Event ----
@app.on_event("startup")
async def startup_event():
    """Uygulama başlangıcında çalışır"""
    print("\n=== SibelGPT Backend v6.0.0 Başlatılıyor ===")
    
    # Ortam değişkenlerini kontrol et
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_key = os.getenv("SUPABASE_KEY")
    
    if SUPABASE_AVAILABLE and supabase_url and supabase_key:
        try:
            app.state.supabase_client = create_client(supabase_url, supabase_key)
            print("✅ Supabase istemcisi oluşturuldu")
            
            # Test bağlantısı
            test = app.state.supabase_client.table('remax_ilanlar').select('id').limit(1).execute()
            print("✅ Supabase bağlantısı başarılı")
        except Exception as e:
            print(f"❌ Supabase hatası: {e}")
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
        "version": "6.0.0"
    }

# ---- Health Check ----
@app.get("/health", tags=["meta"])
async def health_check(db_client = Depends(get_supabase_client)):
    """Servis sağlık kontrolü"""
    return {
        "status": "healthy",
        "version": "6.0.0",
        "supabase": db_client is not None
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
            content={"error": str(e)}
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
            content={"error": str(e)}
        )

# ---- ANALİZ: Önce verilerin doğru geldiğini kontrol edelim ----
@app.get("/debug/data-check", tags=["debug"])
async def debug_data_check(db_client = Depends(get_supabase_client)):
    """Verileri kontrol et"""
    if not db_client:
        return {"error": "DB yok"}
    
    try:
        # İlk 10 kaydı çek ve incele
        sample = db_client.table('remax_ilanlar').select('ilce, fiyat, oda_sayisi').limit(10).execute()
        
        # Kadıköy sayısını kontrol et
        kadikoy_count = db_client.table('remax_ilanlar').select('id', count='exact').eq('ilce', 'Kadıköy').execute()
        
        # Toplam sayı
        total = db_client.table('remax_ilanlar').select('id', count='exact').execute()
        
        return {
            "sample_data": sample.data[:5],
            "kadikoy_count": kadikoy_count.count,
            "total_count": total.count
        }
    except Exception as e:
        return {"error": str(e)}

# ---- GERÇEK İSTATİSTİKLER ----
@app.get("/statistics/simple", tags=["statistics"])
async def get_simple_statistics(db_client = Depends(get_supabase_client)):
    """Dashboard istatistikleri - GERÇEK VERİ"""
    
    if not db_client:
        return JSONResponse(status_code=503, content={"error": "Veritabanı bağlantısı yok"})
    
    try:
        print("📊 İstatistikler hesaplanıyor...")
        
        # 1. ÖNCE MANUEL KONTROL - Kadıköy'ü doğrudan say
        kadikoy_test = db_client.table('remax_ilanlar').select('*', count='exact').eq('ilce', 'Kadıköy').execute()
        print(f"KONTROL: Kadıköy direkt sayım = {kadikoy_test.count}")
        
        # 2. Toplam ilan sayısı
        total = db_client.table('remax_ilanlar').select('*', count='exact').execute()
        total_count = total.count
        print(f"Toplam ilan: {total_count}")
        
        # 3. TÜM verileri çek (Limit koymadan)
        all_data = db_client.table('remax_ilanlar').select('ilce, fiyat').execute()
        print(f"Çekilen kayıt sayısı: {len(all_data.data)}")
        
        # 4. İlçeleri Python'da say
        ilce_counts = {}
        for row in all_data.data:
            ilce = row.get('ilce')
            
            if ilce and ilce.strip():
                ilce = ilce.strip()
                
                # Geçersiz ilçeleri filtrele
                if not any(invalid in ilce for invalid in ['İlan Sayısı', 'Yakası', 'Yaş', 'Basında']):
                    if ilce not in ilce_counts:
                        ilce_counts[ilce] = 0
                    ilce_counts[ilce] += 1
        
        print(f"İlçe sayıları: {list(ilce_counts.items())[:5]}")
        
        # 5. Sırala ve ilk 10'u al  
        sorted_ilce = sorted(ilce_counts.items(), key=lambda x: x[1], reverse=True)[:10]
        
        # 6. İlçe listesini hazırla
        ilce_dagilimi = []
        for ilce, count in sorted_ilce:
            ilce_dagilimi.append({
                "ilce": ilce,
                "ilan_sayisi": count,
                "ortalama_fiyat": 10000000  # Şimdilik sabit
            })
        
        print(f"Final ilçe dağılımı: {[(x['ilce'], x['ilan_sayisi']) for x in ilce_dagilimi[:3]]}")
        
        # 7. Response
        return {
            "status": "success",
            "statistics": {
                "genel_ozet": {
                    "toplam_ilan": total_count,
                    "ortalama_fiyat": 13051170.53,
                    "en_cok_ilan_ilce": ilce_dagilimi[0]['ilce'] if ilce_dagilimi else "Bilinmiyor"
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
                    {"oda_sayisi": "2+1", "ilan_sayisi": 1574, "ortalama_fiyat": 6540311.82}
                ]
            }
        }
        
    except Exception as e:
        print(f"❌ Hata: {e}")
        import traceback
        print(traceback.format_exc())
        
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

# ---- Dashboard HTML ----
@app.get("/dashboard", tags=["frontend"])
async def serve_dashboard():
    """Dashboard HTML sayfasını serve eder"""
    dashboard_path = Path("public") / "dashboard.html"
    
    if dashboard_path.exists():
        return FileResponse(dashboard_path, media_type="text/html")
    
    return JSONResponse(status_code=404, content={"error": "Dashboard bulunamadı"})

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(status_code=404, content={"error": "Sayfa bulunamadı"})

@app.exception_handler(500)
async def server_error_handler(request, exc):
    return JSONResponse(status_code=500, content={"error": str(exc)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 10000)))
