from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    question = data.get("question")

    if not question:
        return {"reply": "Lütfen bir soru yazın."}

    try:
        completion = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "SibelGPT adında bir yapay zeka danışmansın. Kullanıcılara gayrimenkul, numeroloji, finans gibi konularda sıcak, profesyonel ve yardımcı bir tonda yanıt veriyorsun."},
                {"role": "user", "content": question}
            ]
        )
        return {"reply": completion.choices[0].message.content}
    except Exception as e:
        return {"reply": f"Hata oluştu: {str(e)}"}

@app.get("/")
def root():
    return {"message": "SibelGPT aktif!"}
