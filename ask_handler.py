"""
Basit OpenAI sohbet yanıtlayıcısı
(Şimdilik RAG yok, düz model cevabı.)

Gereksinimler:
- requirements.txt içinde openai>=1.0.0 satırı olmalı
- Render ortam değişkenlerinde OPENAI_API_KEY tanımlı olmalı
"""

import os
import openai
from openai import AsyncOpenAI

# Async client (OpenAI 1.x)
openai_client = AsyncOpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

SYSTEM_PROMPT = (
    "You are SibelGPT, a helpful Turkish AI asistan developed by "
    "Sibel Kazan Midilli. Answer briefly, clearly and in Turkish unless "
    "the user asks otherwise."
)

async def answer_question(question: str) -> str:
    """
    Kullanıcı sorusunu OpenAI ChatCompletion ile yanıtla.
    """
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",   # Plus aboneliğinde limitsiz, istersen gpt-3.5-turbo da olur
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.7,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        # Günlüğe yaz; ön yüze güvenli hata dön
        print("OpenAI hatası:", e)
        return "❌ Cevap üretilirken bir hata oluştu. Lütfen tekrar deneyin."
