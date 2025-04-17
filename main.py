from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from openai import OpenAI
import os

# API anahtarını ortam değişkeninden al
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

# CORS ayarı (her yerden erişime izin veriyoruz, istersen sıkılaştırabiliriz)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

class ChatRequest(BaseModel):
    question: str

@app.post("/chat")
async def chat_endpoint(request: ChatRequest):
    try:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Sen SibelGPT adında, deneyimli ve sıcak tarzda konuşan bir yapay zeka danışmanısın. "
                        "Uzmanlık alanların: gayrimenkul,borsa,numeroloji, astroloji, finans ve yaşam rehberliği. "
                        "Teknik bilgi sunarken samimi ve yardımcı ol, tıpkı İstanbul Erenköy’de Remax Sonuç'ta Bağdat Caddesi’nde "
                        "yer alan bir danışman ofisindeymiş gibi konuş. "
                        "Kullanıcıya \"Ben SibelGPT’yim, İstanbul Erenköy’de, Bağdat Caddesi’nde konumlandım.\" "
                        "diyerek kendini tanıtabilirsin. "
                        "Tercihen kadın sesi tonuyla, açık, sade, teknik ama dostça anlat. "
                        "Gerektiğinde örnekler sun ve \"İstersen daha detaylı anlatabilirim.\" gibi esnek yanıtlar ver."
                    )
                },
                {"role": "user", "content": request.question}
            ],
            temperature=0.7
        )
        return {"reply": completion.choices[0].message.content.strip()}
    
    except Exception as e:
        return {"error": f"SibelGPT: Hata oluştu — {str(e)}"}
