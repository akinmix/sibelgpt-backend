# main.py - SibelGPT Backend - v7.0.0 (FIXED VERSION)
import os
import json
import time
import asyncio
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Dict
from contextlib import asynccontextmanager

from fastapi import FastAPI, Depends, Request, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv

# Supabase import kontrolü
try:
    from supabase import create_client
    from supabase.client import Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

# Ortam değişkenlerini yükle
load_dotenv()

# ============= FIXED PERFORMANCE MIDDLEWARE =============
async def performance_middleware(request: Request, call_next):
    """Fixed middleware function - correct signature"""
    start_time = time.time()
    
    # Request ID for tracking
    request_id = id(request)
    
    response = await call_next(request)
    
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Request-ID"] = str(request_id)
    
    # Log slow requests (>2 seconds)
    if process_time > 2.0:
        print(f"⚠️  Slow request: {request.method} {request.url} - {process_time:.2f}s")
    
    return response

# ============= SIMPLIFIED STARTUP/SHUTDOWN =============
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("\n🚀 SibelGPT Backend v7.0.0 - Starting...")
    
    # Initialize Supabase if available
    if SUPABASE_AVAILABLE:
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if supabase_url and supabase_key:
            try:
                app.state.supabase_client = create_client(supabase_url, supabase_key)
                print("✅ Supabase client initialized")
            except Exception as e:
                print(f"❌ Supabase error: {e}")
                app.state.supabase_client = None
        else:
            app.state.supabase_client = None
    else:
        app.state.supabase_client = None
    
    print("✅ Startup complete")
    
    yield
    
    # Shutdown
    print("🔄 Shutting down...")
    print("👋 SibelGPT Backend shutdown complete")

# ============= FASTAPI APP INITIALIZATION =============
app = FastAPI(
    title="SibelGPT Backend",
    version="7.0.0",
    description="SibelGPT AI Assistant Backend API - Fixed Version",
    lifespan=lifespan,
)

# ============= MIDDLEWARE STACK =============
# 1. Performance monitoring (FIXED)
app.middleware("http")(performance_middleware)

# 2. GZIP compression
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 3. CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= SIMPLE RATE LIMITING =============
from collections import defaultdict

rate_limit_store = defaultdict(list)
RATE_LIMIT_REQUESTS = 60
RATE_LIMIT_WINDOW = 60

async def rate_limit_check(request: Request):
    """Simple rate limiting"""
    client_ip = request.client.host
    current_time = time.time()
    
    # Clean old entries
    rate_limit_store[client_ip] = [
        timestamp for timestamp in rate_limit_store[client_ip]
        if current_time - timestamp < RATE_LIMIT_WINDOW
    ]
    
    # Check limit
    if len(rate_limit_store[client_ip]) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded"
        )
    
    # Add current request
    rate_limit_store[client_ip].append(current_time)

# ============= MODELS =============
class ChatRequest(BaseModel):
    question: str
    mode: str = "real-estate"
    conversation_history: List[Dict] = []

class WebSearchRequest(BaseModel):
    question: str
    mode: str = "real-estate"

# ============= DEPENDENCY =============
async def get_supabase_client(request: Request) -> Optional[Client]:
    return getattr(request.app.state, 'supabase_client', None)

# ============= STATIC FILES =============
if os.path.exists("public"):
    app.mount("/static", StaticFiles(directory="public"), name="static")

# ============= ENDPOINTS =============
@app.get("/", tags=["meta"])
async def root():
    return {
        "status": "ok",
        "service": "SibelGPT Backend",
        "version": "7.0.0"
    }

@app.get("/health", tags=["meta"])
async def health_check(db_client = Depends(get_supabase_client)):
    return {
        "status": "healthy",
        "version": "7.0.0",
        "supabase": db_client is not None,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/chat", tags=["chat"])
async def chat(
    payload: ChatRequest, 
    background_tasks: BackgroundTasks,
    db_client = Depends(get_supabase_client),
    _rate_limit = Depends(rate_limit_check)
):
    """Chat endpoint"""
    start_time = time.time()
    
    try:
        import ask_handler
        
        answer = await ask_handler.answer_question(
            payload.question, 
            payload.mode, 
            payload.conversation_history
        )
        
        # Log slow queries in background
        duration = time.time() - start_time
        if duration > 1.0:
            print(f"📊 Slow chat query: {duration:.2f}s - Mode: {payload.mode}")
        
        return {"reply": answer}
        
    except Exception as e:
        print(f"❌ Chat error: {str(e)}")
        return JSONResponse(
            status_code=500, 
            content={"error": f"Chat processing failed: {str(e)}"}
        )

@app.post("/web-search", tags=["search"])
async def web_search(
    payload: WebSearchRequest,
    background_tasks: BackgroundTasks,
    _rate_limit = Depends(rate_limit_check)
):
    """Web search endpoint"""
    start_time = time.time()
    
    try:
        import search_handler
        
        answer = await search_handler.web_search_answer(payload.question, payload.mode)
        
        duration = time.time() - start_time
        if duration > 2.0:
            print(f"📊 Slow web search: {duration:.2f}s")
        
        return {"reply": answer}
        
    except Exception as e:
        print(f"❌ Web search error: {str(e)}")
        return JSONResponse(
            status_code=500, 
            content={"error": f"Web search failed: {str(e)}"}
        )

# ============= STATISTICS ENDPOINT =============
@app.get("/statistics/simple", tags=["statistics"])
async def get_simple_statistics():
    """Dashboard statistics"""
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
                {"ilce": "Beylikdüzü", "ilan_sayisi": 304, "ortalama_fiyat": 8759901.32},
                {"ilce": "Kartal", "ilan_sayisi": 290, "ortalama_fiyat": 8382693.10},
                {"ilce": "Pendik", "ilan_sayisi": 273, "ortalama_fiyat": 7970626.37},
                {"ilce": "Maltepe", "ilan_sayisi": 257, "ortalama_fiyat": 8779984.43}
            ],
            "fiyat_dagilimi": [
                {"aralik": "0-5M ₺", "ilan_sayisi": 2327, "yuzde": 46.11},
                {"aralik": "5-10M ₺", "ilan_sayisi": 1582, "yuzde": 31.34},
                {"aralik": "10-20M ₺", "ilan_sayisi": 1024, "yuzde": 20.29},
                {"aralik": "20M+ ₺", "ilan_sayisi": 114, "yuzde": 2.26}
            ]
        }
    }

# ============= DASHBOARD =============
@app.get("/dashboard", tags=["frontend"])
async def serve_dashboard():
    dashboard_path = Path("public") / "dashboard.html"
    
    if dashboard_path.exists():
        return FileResponse(dashboard_path, media_type="text/html")
    
    return JSONResponse(status_code=404, content={"error": "Dashboard bulunamadı"})

# ============= ERROR HANDLERS =============
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(status_code=404, content={"error": "Sayfa bulunamadı"})

@app.exception_handler(500)
async def server_error_handler(request, exc):
    return JSONResponse(status_code=500, content={"error": str(exc)})

# ============= ROUTER LOADING =============
@app.on_event("startup")
async def setup_routers():
    """Load routers after startup"""
    try:
        from image_handler import router as image_router
        from pdf_handler import router as pdf_router
        from elevenlabs_handler import router as elevenlabs_router
        
        app.include_router(image_router, prefix="", tags=["image"])
        app.include_router(pdf_router, prefix="", tags=["pdf"])
        app.include_router(elevenlabs_router, prefix="", tags=["speech"])
        
        print("✅ All routers loaded")
    except Exception as e:
        print(f"⚠️  Router loading error: {e}")

# ============= PERFORMANCE ENDPOINT =============
@app.get("/performance", tags=["monitoring"])
async def get_performance_stats():
    """Performance statistics"""
    return {
        "status": "active",
        "rate_limiting": {
            "active_ips": len(rate_limit_store),
            "limit_per_minute": RATE_LIMIT_REQUESTS
        },
        "system": {
            "version": "7.0.0",
            "uptime": "Available after full startup"
        }
    }

# ============= MAIN EXECUTION =============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 10000))
    )
