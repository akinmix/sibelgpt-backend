# main.py  (TEMİZ SÜRÜM)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# ---- Dahili modüller ----
from image_handler import router as image_router
from ask_handler import answer_question  # Chat endpoint için
from pydantic import BaseModel

class ChatRequest(BaseModel):
    question: str


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
app.include_router(image_router, prefix="", tags=["image"])


@app.post("/chat", tags=["chat"])
async def chat(payload: ChatRequest):
    """
    Frontend'den gelen soruyu cevaplar.
    """
    answer = await answer_question(payload.question)
    return {"reply": answer}

@app.get("/", tags=["meta"])
async def root():
    """
    Render health check ⇒ 200 OK
    """
    return {"status": "ok"}

