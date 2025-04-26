# main.py  (TEMİZ SÜRÜM)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ---- Dahili modüller ----
from image_handler import router as image_router
from routes.ilan_detay import router as ilan_router
from ask_handler import answer_question  # Chat endpoint için

app = FastAPI(
    title="SibelGPT Backend",
    version="1.0.0",
)

# CORS – Vercel frontend’inden gelen çağrılara izin ver
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://www.sibelgpt.com", "https://sibelgpt.com"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---- ROUTE KAYDI ----
app.include_router(image_router, prefix="/image", tags=["image"])
app.include_router(ilan_router, prefix="/api", tags=["ilan"])

@app.post("/chat", tags=["chat"])
async def chat(question: str):
    """
    Frontend'den gelen soruyu RAG (ask_handler) aracılığıyla cevaplar.
    """
    answer = await answer_question(question)
    return {"reply": answer}
