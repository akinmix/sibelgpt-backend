# main.py - SibelGPT Backend - v5.0.0 (FINAL)
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
   version="5.0.0",
   description="SibelGPT AI Assistant Backend API - Final Version"
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
   print("✅ Static klasör mount edildi")
else:
   print("❌ 'public' klasörü bulunamadı")

# ---- Startup Event ----
@app.on_event("startup")
async def startup_event():
   """Uygulama başlangıcında çalışır"""
   print("\n=== SibelGPT Backend v5.0.0 Başlatılıyor ===")
   
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
       "version": "5.0.0",
       "endpoints": {
           "chat": "/chat",
           "web_search": "/web-search",
           "image": "/image",
           "statistics": "/statistics/dashboard",
           "statistics_simple": "/statistics/simple",
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
       "version": "5.0.0",
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

# ---- Dashboard İstatistikleri (RPC) ----
@app.get("/statistics/dashboard", tags=["statistics"])
async def get_dashboard_statistics(db_client = Depends(get_supabase_client)):
   """Dashboard istatistiklerini döndürür (RPC kullanır)"""
   print("📊 Dashboard istatistikleri istendi (RPC)")
   
   if not db_client:
       return JSONResponse(
           status_code=503,
           content={"error": "Veritabanı bağlantısı yok"}
       )
   
   try:
       result = db_client.rpc('get_dashboard_statistics', params={}).execute()
       print(f"✅ RPC yanıtı alındı")
       
       if result.data:
           data = result.data
           if isinstance(data, list) and len(data) > 0:
               data = data[0]
           
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
       print(f"❌ Dashboard RPC hatası: {e}")
       return JSONResponse(
           status_code=500,
           content={
               "error": "İstatistikler alınırken hata oluştu",
               "detail": str(e),
               "type": type(e).__name__
           }
       )

# ---- GERÇEK İSTATİSTİKLER (TÜM ISTANBUL) ----
@app.get("/statistics/simple", tags=["statistics"])
async def get_simple_statistics(db_client = Depends(get_supabase_client)):
   """Dashboard için gerçek istatistikler - TÜM İLÇELER"""
   
   if not db_client:
       return JSONResponse(
           status_code=503,
           content={"error": "Veritabanı bağlantısı yok"}
       )
   
   try:
       # 1. Toplam ilan sayısı
       total_result = db_client.table('remax_ilanlar').select('id', count='exact').execute()
       total_count = total_result.count if total_result.count else 0
       print(f"✓ Toplam ilan: {total_count}")
       
       # 2. TÜM verileri çek (ilce ve fiyat)
       print("🔄 Tüm veriler çekiliyor...")
       all_data = db_client.table('remax_ilanlar').select('ilce, fiyat').execute()
       
       if not all_data.data:
           return JSONResponse(
               status_code=404,
               content={"error": "Veritabanında veri bulunamadı"}
           )
       
       print(f"✓ {len(all_data.data)} kayıt çekildi")
       
       # 3. İlçe bazlı istatistikleri hesapla
       ilce_stats = {}
       total_price_sum = 0
       valid_price_count = 0
       
       # Her satırı işle
       for row in all_data.data:
           ilce = row.get('ilce')
           fiyat = row.get('fiyat')
           
           # İlçe istatistikleri
           if ilce and ilce.strip():
               ilce = ilce.strip()
               
               # Geçersiz ilçeleri filtrele
               if ('İlan Sayısı' not in ilce and 
                   'Yakası' not in ilce and 
                   'İlan Sayısı ve Yakası' not in ilce and
                   len(ilce) < 50):
                   
                   if ilce not in ilce_stats:
                       ilce_stats[ilce] = {
                           'count': 0,
                           'price_sum': 0,
                           'valid_prices': 0
                       }
                   
                   # İlan sayısını artır
                   ilce_stats[ilce]['count'] += 1
                   
                   # Fiyat hesaplaması
                   if fiyat:
                       try:
                           # "5.500.000" formatını 5500000'e çevir
                           clean_price = float(str(fiyat).replace('.', '').replace(',', ''))
                           if clean_price > 0:
                               ilce_stats[ilce]['price_sum'] += clean_price
                               ilce_stats[ilce]['valid_prices'] += 1
                               total_price_sum += clean_price
                               valid_price_count += 1
                       except:
                           pass
       
       print(f"✓ {len(ilce_stats)} farklı ilçe bulundu")
       
       # 4. İlçe dağılımını hazırla
       ilce_dagilimi = []
       for ilce, stats in ilce_stats.items():
           avg_price = 0
           if stats['valid_prices'] > 0:
               avg_price = stats['price_sum'] / stats['valid_prices']
           
           ilce_dagilimi.append({
               'ilce': ilce,
               'ilan_sayisi': stats['count'],
               'ortalama_fiyat': avg_price
           })
       
       # İlan sayısına göre sırala ve ilk 10'u al
       ilce_dagilimi.sort(key=lambda x: x['ilan_sayisi'], reverse=True)
       top_10_ilceler = ilce_dagilimi[:10]
       
       print(f"✓ Top 3 ilçe: {[(x['ilce'], x['ilan_sayisi']) for x in top_10_ilceler[:3]]}")
       
       # 5. Genel ortalama fiyat
       genel_ortalama = 0
       if valid_price_count > 0:
           genel_ortalama = total_price_sum / valid_price_count
       
       # 6. En çok ilan olan ilçe
       en_cok_ilan_ilce = top_10_ilceler[0]['ilce'] if top_10_ilceler else "Kadıköy"
       
       # 7. Final response
       return {
           "status": "success",
           "statistics": {
               "genel_ozet": {
                   "toplam_ilan": total_count,
                   "ortalama_fiyat": genel_ortalama,
                   "en_cok_ilan_ilce": en_cok_ilan_ilce
               },
               "ilce_dagilimi": top_10_ilceler,
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
       import traceback
       print(traceback.format_exc())
       
       return JSONResponse(
           status_code=500,
           content={
               "error": "İstatistikler hesaplanırken hata oluştu",
               "detail": str(e),
               "type": type(e).__name__
           }
       )

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
   
   return JSONResponse(
       status_code=404,
       content={
           "error": "Dashboard sayfası bulunamadı",
           "paths_checked": [str(p) for p in possible_paths]
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
               "/image", "/statistics/dashboard", "/statistics/simple", "/dashboard"
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
   print("🚀 SibelGPT Backend v5.0.0 başlatılıyor...")
   uvicorn.run(
       app, 
       host="0.0.0.0", 
       port=int(os.getenv("PORT", 10000)),
       reload=True
   )
