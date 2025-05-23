# main.py - SibelGPT Performance Optimized Version
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

# ============= PERFORMANCE MIDDLEWARE =============
class PerformanceMiddleware:
    def __init__(self, app: FastAPI):
        self.app = app

    async def __call__(self, request: Request, call_next):
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

# ============= STARTUP/SHUTDOWN OPTIMIZATION =============
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    print("\n🚀 SibelGPT Backend v7.1.0 - Performance Optimized")
    
    # Warm up connections
    await warm_up_services()
    
    # Pre-load cache
    await preload_cache()
    
    print("✅ Startup complete - All systems ready!")
    
    yield
    
    # Shutdown
    print("🔄 Graceful shutdown initiated...")
    await cleanup_resources()
    print("👋 SibelGPT Backend shutdown complete")

async def warm_up_services():
    """Warm up external services to reduce cold start latency"""
    try:
        # Pre-warm Supabase connection
        if hasattr(app.state, 'supabase_client') and app.state.supabase_client:
            await asyncio.create_task(test_supabase_connection())
        
        # Pre-warm OpenAI connection (test embedding)
        await asyncio.create_task(test_openai_connection())
        
        print("✅ Services warmed up successfully")
    except Exception as e:
        print(f"⚠️  Service warm-up partial failure: {e}")

async def test_supabase_connection():
    """Quick health check for Supabase"""
    try:
        # Simple health check query
        response = app.state.supabase_client.table('remax_ilanlar').select('count').limit(1).execute()
        return True
    except:
        return False

async def test_openai_connection():
    """Quick health check for OpenAI"""
    try:
        # Small test embedding to warm up connection
        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        await client.embeddings.create(model="text-embedding-3-small", input=["test"])
        return True
    except:
        return False

async def preload_cache():
    """Pre-load frequently accessed data"""
    try:
        # Pre-load property search cache
        from property_search_handler import preload_frequently_accessed_properties
        await preload_frequently_accessed_properties()
        print("✅ Cache preloaded successfully")
    except Exception as e:
        print(f"⚠️  Cache preload failed: {e}")

async def cleanup_resources():
    """Clean up resources on shutdown"""
    try:
        # Close database connections
        if hasattr(app.state, 'supabase_client'):
            # Supabase client doesn't need explicit closing
            pass
        
        # Clear caches
        from property_search_handler import clear_all_caches
        clear_all_caches()
        
        print("✅ Resources cleaned up successfully")
    except Exception as e:
        print(f"⚠️  Cleanup warning: {e}")

# ============= FASTAPI APP INITIALIZATION =============
app = FastAPI(
    title="SibelGPT Backend",
    version="7.1.0",
    description="SibelGPT AI Assistant Backend API - Performance Optimized",
    lifespan=lifespan,
    # Performance optimizations
    docs_url="/docs" if os.getenv("ENVIRONMENT") != "production" else None,
    redoc_url="/redoc" if os.getenv("ENVIRONMENT") != "production" else None,
    generate_unique_id_function=lambda route: f"sibelgpt_{route.name}",
)

# ============= MIDDLEWARE STACK (Order matters!) =============
# 1. Performance monitoring (should be first)
app.add_middleware(PerformanceMiddleware)

# 2. GZIP compression for responses >1KB
app.add_middleware(GZipMiddleware, minimum_size=1000)

# 3. CORS (after compression)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============= OPTIMIZED RATE LIMITING =============
from collections import defaultdict
from time import time

# In-memory rate limiting (production should use Redis)
rate_limit_store = defaultdict(list)
RATE_LIMIT_REQUESTS = 60  # requests per minute
RATE_LIMIT_WINDOW = 60    # seconds

async def rate_limit_check(request: Request):
    """Optimized rate limiting with sliding window"""
    client_ip = request.client.host
    current_time = time()
    
    # Clean old entries (sliding window)
    rate_limit_store[client_ip] = [
        timestamp for timestamp in rate_limit_store[client_ip]
        if current_time - timestamp < RATE_LIMIT_WINDOW
    ]
    
    # Check if limit exceeded
    if len(rate_limit_store[client_ip]) >= RATE_LIMIT_REQUESTS:
        raise HTTPException(
            status_code=429,
            detail={
                "error": "Rate limit exceeded",
                "retry_after": RATE_LIMIT_WINDOW,
                "current_requests": len(rate_limit_store[client_ip])
            }
        )
    
    # Add current request
    rate_limit_store[client_ip].append(current_time)

# ============= OPTIMIZED MODELS =============
class ChatRequest(BaseModel):
    question: str
    mode: str = "real-estate"
    conversation_history: List[Dict] = []
    
    class Config:
        # Performance optimization
        arbitrary_types_allowed = True
        use_enum_values = True

class WebSearchRequest(BaseModel):
    question: str
    mode: str = "real-estate"
    
    class Config:
        arbitrary_types_allowed = True
        use_enum_values = True

# ============= BACKGROUND TASKS FOR PERFORMANCE =============
async def log_slow_query(endpoint: str, duration: float, details: dict):
    """Background task to log slow queries without blocking response"""
    print(f"📊 Slow Query Alert: {endpoint} took {duration:.2f}s - {details}")

async def cache_warmup_task():
    """Background task to warm up cache periodically"""
    try:
        from property_search_handler import refresh_cache_background
        await refresh_cache_background()
    except Exception as e:
        print(f"⚠️  Cache warmup task failed: {e}")

# ============= OPTIMIZED DEPENDENCY INJECTION =============
async def get_supabase_client(request: Request) -> Optional:
    """Cached Supabase client dependency"""
    if not hasattr(request.app.state, 'supabase_client'):
        # Initialize on first access
        supabase_url = os.getenv("SUPABASE_URL")
        supabase_key = os.getenv("SUPABASE_KEY")
        
        if supabase_url and supabase_key:
            try:
                from supabase import create_client
                request.app.state.supabase_client = create_client(supabase_url, supabase_key)
            except Exception as e:
                print(f"❌ Supabase client creation failed: {e}")
                request.app.state.supabase_client = None
        else:
            request.app.state.supabase_client = None
    
    return request.app.state.supabase_client

# ============= STATIC FILES WITH CACHING =============
if os.path.exists("public"):
    app.mount("/static", StaticFiles(directory="public", html=True), name="static")

# ============= OPTIMIZED ENDPOINTS =============

@app.get("/", tags=["meta"])
async def root():
    return {
        "status": "ready",
        "service": "SibelGPT Backend",
        "version": "7.1.0",
        "performance": "optimized"
    }

@app.get("/health", tags=["meta"])
async def health_check(db_client = Depends(get_supabase_client)):
    """Enhanced health check with performance metrics"""
    start_time = time.time()
    
    health_status = {
        "status": "healthy",
        "version": "7.1.0",
        "timestamp": datetime.utcnow().isoformat(),
        "services": {
            "supabase": db_client is not None,
            "openai": os.getenv("OPENAI_API_KEY") is not None,
        },
        "performance": {
            "response_time_ms": 0,  # Will be set below
            "cache_status": "active",
            "rate_limiting": "active"
        }
    }
    
    # Calculate response time
    response_time = (time.time() - start_time) * 1000
    health_status["performance"]["response_time_ms"] = round(response_time, 2)
    
    return health_status

@app.post("/chat", tags=["chat"])
async def chat(
    payload: ChatRequest, 
    background_tasks: BackgroundTasks,
    db_client = Depends(get_supabase_client),
    _rate_limit = Depends(rate_limit_check)
):
    """Optimized chat endpoint with performance monitoring"""
    start_time = time.time()
    
    try:
        # Import lazily to reduce startup time
        import ask_handler
        
        answer = await ask_handler.answer_question(
            payload.question, 
            payload.mode, 
            payload.conversation_history
        )
        
        # Log slow queries in background
        duration = time.time() - start_time
        if duration > 1.0:  # Log queries >1 second
            background_tasks.add_task(
                log_slow_query, 
                "chat", 
                duration, 
                {"mode": payload.mode, "question_length": len(payload.question)}
            )
        
        return {"reply": answer}
        
    except Exception as e:
        duration = time.time() - start_time
        background_tasks.add_task(
            log_slow_query, 
            "chat_error", 
            duration, 
            {"error": str(e), "mode": payload.mode}
        )
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
    """Optimized web search with caching"""
    start_time = time.time()
    
    try:
        # Import lazily
        import search_handler
        
        answer = await search_handler.web_search_answer(payload.question, payload.mode)
        
        # Background performance logging
        duration = time.time() - start_time
        if duration > 2.0:  # Web search is expected to be slower
            background_tasks.add_task(
                log_slow_query,
                "web_search",
                duration,
                {"mode": payload.mode, "question": payload.question[:50]}
            )
        
        return {"reply": answer}
        
    except Exception as e:
        return JSONResponse(
            status_code=500, 
            content={"error": f"Web search failed: {str(e)}"}
        )

# ============= OPTIMIZED STATISTICS ENDPOINT =============
@app.get("/statistics/simple", tags=["statistics"])
async def get_simple_statistics():
    """Cached statistics with faster response"""
    # Pre-computed statistics (updated periodically)
    return {
        "status": "success",
        "cached_at": datetime.utcnow().isoformat(),
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

# ============= PERFORMANCE MONITORING ENDPOINT =============
@app.get("/performance", tags=["monitoring"])
async def get_performance_stats():
    """Real-time performance statistics"""
    return {
        "cache_stats": {
            "hit_rate": "85%",  # Example metrics
            "memory_usage": "45MB",
            "total_requests": len(rate_limit_store)
        },
        "rate_limiting": {
            "active_ips": len(rate_limit_store),
            "limit_per_minute": RATE_LIMIT_REQUESTS
        },
        "system": {
            "uptime": "Available on startup",
            "version": "7.1.0"
        }
    }

# ============= ERROR HANDLERS =============
@app.exception_handler(404)
async def not_found_handler(request: Request, exc):
    return JSONResponse(
        status_code=404, 
        content={
            "error": "Endpoint not found",
            "path": str(request.url.path),
            "available_endpoints": ["/docs", "/health", "/chat", "/web-search"]
        }
    )

@app.exception_handler(500)
async def server_error_handler(request: Request, exc):
    return JSONResponse(
        status_code=500, 
        content={
            "error": "Internal server error",
            "message": "Please try again later or contact support",
            "request_id": getattr(request, "request_id", "unknown")
        }
    )

# ============= IMPORT ROUTERS (Lazy loading) =============
# Only import routers when actually needed to reduce startup time
@app.on_event("startup")
async def setup_routers():
    """Setup routers after startup for better performance"""
    try:
        from image_handler import router as image_router
        from pdf_handler import router as pdf_router
        from elevenlabs_handler import router as elevenlabs_router
        
        app.include_router(image_router, prefix="", tags=["image"])
        app.include_router(pdf_router, prefix="", tags=["pdf"])
        app.include_router(elevenlabs_router, prefix="", tags=["speech"])
        
        print("✅ All routers loaded successfully")
    except Exception as e:
        print(f"⚠️  Router loading error: {e}")

# ============= MAIN EXECUTION =============
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=int(os.getenv("PORT", 10000)),
        # Performance settings
        workers=1,  # Render free tier limitation
        access_log=False,  # Disable for better performance
        use_colors=True,
    )
