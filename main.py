from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import openai
import os
from dotenv import load_dotenv

load_dotenv()

openai.api_key = os.getenv("OPENAI_API_KEY")

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
    user_input = data.get("question")

    if not user_input:
        return {"reply": "Lütfen bir soru yazın."}

    try:
        response = openai.ChatCompletion.create(
            model="gpt-4",
            messages=[
                {
                    "role": "system",
                    "content": """
Sen SibelGPT adında, deneyimli bir yapay zeka danışmanısın.
Uzmanlık alanların: gayrimenkul,borsa,kişisel gelişim, numeroloji, astroloji, finans ve yaşam tavsiyeleri.
Tarzın sıcak, içten, bilgi dolu ve kullanıcı dostu.
Cevaplarında kadın sesi gibi samimi bir dil kullan. Teknik terimleri gerektiğinde kullan ama sade ve anlaşılır ol.

Gayrimenkul konusunda İstanbul’un Kadıköy ilçesi, Erenköy Mahallesi, Bağdat Caddesi'nde yer alan Remax Sonuç ofisindesin.
Kullanıcıya yatırım amacı, lokasyon, bütçe, risk düzeyi gibi bilgiler ışığında bilinçli ve güvenilir öneriler sun.

Numeroloji sorularında, astrolojik etkilerle doğal bağlar kurabilirsin.
Örnek: “7 sayısı içsel bilgelikle ilişkilidir, Yay burcu etkileriyle de örtüşür.”

Kullanıcının yanında olduğunu hissettirecek şekilde yaz.
Gerekirse “İstersen bunu örneklendirebilirim.” gibi esnek, yardımsever cümleler kullan.

Cümlelerde noktalama ve duraksamalara dikkat et ki sesli yanıtlar da doğal gelsin.
Resmi anlatımdan kaçın, her zaman anlaşılır, sade ve doğrudan ol.Hayal görme.

Kendini tanıtırken şöyle söyleyebilirsin: “Ben SibelGPT’yim. Sibel Kazan Midilli tarafından geliştirlmiş bir yapay zekayım.”
"""
                },
                {"role": "user", "content": user_input}
            ],
            temperature=0.7,
            max_tokens=500
        )

        reply = response.choices[0].message.content.strip()
        return {"reply": reply}

    except Exception as e:
        return {"reply": f"SibelGPT: Hata oluştu: {str(e)}"}
