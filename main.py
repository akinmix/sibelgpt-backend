from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os

# .env dosyasından API anahtarını al
from dotenv import load_dotenv
load_dotenv()

app = FastAPI()

# CORS ayarları
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# OpenAI istemcisini başlat
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# System Prompt: Kişiselleştirilmiş başlangıç mesajı
system_message = {
    "role": "system",
    "content": """
Sen SibelGPT adında, deneyimli bir yapay zeka danışmanısın.
Uzmanlık alanların: gayrimenkul, numeroloji, astroloji, finans ve yaşam tavsiyeleri.
Tarzın sıcak, içten, bilgi dolu ve kullanıcı dostu.
Cevaplarında kadın sesi gibi samimi bir dil kullan. Teknik terimleri gerektiğinde kullan ama sadeleştir.
İstanbul'un Kadıköy ilçesi, Erenköy Mahallesi, Bağdat Caddesi'nde bir ofistesin.

Kullanıcının yanında olduğunu hissettirecek şekilde yaz.
Noktalama ve duraksamalara dikkat et. Yanıtların doğal, sade ve doğrudan olsun.
"""
}

@app.post("/chat")
async def chat(request: Request):
    try:
        body = await request.json()
        question = body.get("question")

        if not question:
            return {"reply": "Lütfen bir soru yazın."}

        messages = [system_message, {"role": "user", "content": question}]
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages
        )

        reply = response.choices[0].message.content
        return {"reply": reply}

    except Exception as e:
        return {"reply": f"SibelGPT: Hata oluştu. {str(e)}"}
