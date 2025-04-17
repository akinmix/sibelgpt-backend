from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import openai
import os
from dotenv import load_dotenv

load_dotenv()  # .env dosyasındaki API anahtarını al

openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

# CORS ayarları: Vercel'den gelen istekleri kabul et
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Can be restricted to your frontend URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    question = data.get("question")

    if not question:
        return {"reply": "Lütfen bir soru yazınız."}

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "SibelGPT adında bir yapay zeka danışmansın. Kullanıcılara gayrimenkul, numeroloji, finans gibi konularda sıcak, profesyonel ve yardımcı bir tonda yanıt veriyorsun."},
                {"role": "user", "content": question}
            ]
        )
        answer = response['choices'][0]['message']['content']
        return {"reply": answer}

    except Exception as e:
        return {"reply": f"Hata oluştu: {str(e)}"}

@app.get("/")
def root():
    return {"message": "SibelGPT API aktif."}
