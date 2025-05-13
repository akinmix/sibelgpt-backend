# main.py - SibelGPT Backend - v3.0.0
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
   version="3.0.0",
   description="SibelGPT AI Assistant Backend API - Fully Dynamic"
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
   print("\n=== SibelGPT Backend v3.0.0 Başlatılıyor ===")
   
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
       "version": "3.0.0",
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
       "version": "3.0.0",
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

# ---- TAMAMEN DİNAMİK İSTATİSTİKLER ----
@app.get("/statistics/simple", tags=["statistics"])
async def get_simple_statistics(db_client = Depends(get_supabase_client)):
   """Basit istatistikler - TÜM VERİLER SUPABASE'DEN ÇEKİLİR"""
   print("📊 Dinamik istatistikler istendi")
   
   if not db_client:
       return JSONResponse(
           status_code=503,
           content={"error": "Veritabanı bağlantısı yok"}
       )
   
   try:
       # 1. TÜM VERİYİ ÇEK (ilce, fiyat, oda_sayisi)
       print("🔄 Supabase'den tüm veriler çekiliyor...")
       all_data_result = db_client.table('remax_ilanlar').select('ilce, fiyat, oda_sayisi').execute()
       
       if not all_data_result.data:
           print("⚠️ Hiç veri bulunamadı")
           return JSONResponse(
               status_code=404,
               content={"error": "Veritabanında hiç ilan bulunamadı"}
           )
       
       all_data = all_data_result.data
       total_count = len(all_data)
       print(f"✅ {total_count} kayıt bulundu")
       
       # 2. VERİ YAPILARINI HAZIRLA
       ilce_stats = {}
       oda_stats = {}
       price_ranges = {
           '0-5M': 0,
           '5-10M': 0,
           '10-20M': 0,
           '20M+': 0
       }
       total_price_sum = 0
       valid_price_count = 0
       
       # 3. HER KAYDI İŞLE
       print("🔄 Veriler işleniyor...")
       for row in all_data:
           ilce = row.get('ilce')
           fiyat = row.get('fiyat')
           oda_sayisi = row.get('oda_sayisi')
           
           # İLÇE İSTATİSTİKLERİ
           if ilce:
               if ilce not in ilce_stats:
                   ilce_stats[ilce] = {
                       'count': 0,
                       'price_sum': 0,
                       'valid_prices': 0
                   }
               ilce_stats[ilce]['count'] += 1
               
               # FİYAT İŞLEME
               if fiyat:
                   try:
                       # Fiyat temizleme: "5.500.000" -> 5500000
                       fiyat_str = str(fiyat)
                       clean_price = float(fiyat_str.replace('.', '').replace(',', ''))
                       
                       if clean_price > 0:
                           ilce_stats[ilce]['price_sum'] += clean_price
                           ilce_stats[ilce]['valid_prices'] += 1
                           total_price_sum += clean_price
                           valid_price_count += 1
                           
                           # FİYAT ARALIĞI BELİRLE
                           if clean_price < 5000000:
                               price_ranges['0-5M'] += 1
                           elif clean_price < 10000000:
                               price_ranges['5-10M'] += 1
                           elif clean_price < 20000000:
                               price_ranges['10-20M'] += 1
                           else:
                               price_ranges['20M+'] += 1
                   except Exception as e:
                       pass  # Geçersiz fiyat verisi, atla
           
           # ODA SAYISI İSTATİSTİKLERİ
           if oda_sayisi:
               if oda_sayisi not in oda_stats:
                   oda_stats[oda_sayisi] = {
                       'count': 0,
                       'price_sum': 0,
                       'valid_prices': 0
                   }
               oda_stats[oda_sayisi]['count'] += 1
               
               # Oda tipi için ortalama fiyat
               if fiyat:
                   try:
                       clean_price = float(str(fiyat).replace('.', '').replace(',', ''))
                       if clean_price > 0:
                           oda_stats[oda_sayisi]['price_sum'] += clean_price
                           oda_stats[oda_sayisi]['valid_prices'] += 1
                   except:
                       pass
       
       # 4. İLÇE DAĞILIMINI HAZIRLA
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
       
       # İlan sayısına göre sırala, ilk 10'u al
       ilce_dagilimi.sort(key=lambda x: x['ilan_sayisi'], reverse=True)
       top_10_ilceler = ilce_dagilimi[:10]
       
       print(f"✅ {len(ilce_stats)} ilçe bulundu, ilk 10'u alındı")
       
       # 5. ODA TİPİ DAĞILIMINI HAZIRLA
       oda_tipi_dagilimi = []
       for oda, stats in oda_stats.items():
           avg_price = 0
           if stats['valid_prices'] > 0:
               avg_price = stats['price_sum'] / stats['valid_prices']
           
           oda_tipi_dagilimi.append({
               'oda_sayisi': oda,
               'ilan_sayisi': stats['count'],
               'ortalama_fiyat': avg_price
           })
       
       # İlan sayısına göre sırala, ilk 6'yı al
       oda_tipi_dagilimi.sort(key=lambda x: x['ilan_sayisi'], reverse=True)
       top_6_oda = oda_tipi_dagilimi[:6]
       
       # 6. FİYAT ARALIĞI YÜZDELERİNİ HESAPLA
       fiyat_dagilimi = []
       total_with_price = sum(price_ranges.values())
       
       for aralik, count in price_ranges.items():
           yuzde = 0
           if total_with_price > 0:
               yuzde = (count / total_with_price) * 100
           
           fiyat_dagilimi.append({
               'aralik': aralik + ' ₺',
               'ilan_sayisi': count,
               'yuzde': round(yuzde, 2)
           })
       
       # 7. GENEL İSTATİSTİKLER
       genel_ortalama = 0
       if valid_price_count > 0:
           genel_ortalama = total_price_sum / valid_price_count
       
       en_cok_ilan_ilce = "Bilinmiyor"
       if top_10_ilceler:
           en_cok_ilan_ilce = top_10_ilceler[0]['ilce']
       
       print("✅ Tüm hesaplamalar tamamlandı")
       
       # 8. SONUCU DÖNDÜR
       return {
           "status": "success",
           "statistics": {
               "genel_ozet": {
                   "toplam_ilan": total_count,
                   "ortalama_fiyat": genel_ortalama,
                   "en_cok_ilan_ilce": en_cok_ilan_ilce
               },
               "ilce_dagilimi": top_10_ilceler,
               "fiyat_dagilimi": fiyat_dagilimi,
               "oda_tipi_dagilimi": top_6_oda
           }
       }
       
   except Exception as e:
       print(f"❌ Kritik hata: {e}")
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
   
   # Olası dosya yolları
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
   print("🚀 SibelGPT Backend v3.0.0 başlatılıyor...")
   uvicorn.run(
       app, 
       host="0.0.0.0", 
       port=int(os.getenv("PORT", 10000)),
       reload=True
   )
