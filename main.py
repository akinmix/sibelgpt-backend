from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI()

# CORS ayarları – Tüm kaynaklara izin veriyoruz
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # veya sadece "https://www.sibelgpt.com"
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    question: str

@app.post("/chat")
async def chat(query: Query):
    user_question = query.question
    return {
        "reply": f"Senin sorun: '{user_question}' — SibelGPT henüz eğitilmediği için örnek cevap döndürüyor."
    }
