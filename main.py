from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import os

# API key'i ortam değişkeninden al
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# İstek modeli
class Query(BaseModel):
    question: str

@app.post("/chat")
async def chat(query: Query):
    try:
        completion = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "Sen SibelGPT’sin. İstanbul'da yaşayan, yapay zekadan anlayan, sıcak ve bilgili bir kadın danışmansın. Kullanıcıdan gelen sorulara içtenlikle ve bilgece cevap ver."},
                {"role": "user", "content": query.question}
            ]
        )
        reply = completion.choices[0].message["content"]
        return {"reply": reply}
    except Exception as e:
        return {"error": str(e)}
