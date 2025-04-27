"""
ask_handler.py
--------------
Basit OpenAI sohbet yanıtlayıcısı.
(İleride RAG eklemek istersen bu fonksiyonun içini genişletebilirsin.)

Ön-koşullar
-----------
• requirements.txt → openai>=1.0.0 satırı var.  
• Render ortam değişkenlerinde OPENAI_API_KEY ayarlı.
"""

import os
from openai import AsyncOpenAI

# ── OpenAI istemcisi ───────────────────────────────────────────────────────────
openai_client = AsyncOpenAI(
    api_key=os.environ.get("OPENAI_API_KEY")  # Render'da tanımlı olmalı
)

# Sistem talimatı (modelin kişiliği)
SYSTEM_PROMPT = (
    "Sen SibelGPT'sin: Sibel Kazan Midilli tarafından geliştirilen, "
    "Türkçe yanıt veren yardımsever bir yapay zeka asistanısın. "
    "Yanıtlarını kısa, açık ve anlaşılır tut."
)

# Ana fonksiyon – backend'de main.py çağırıyor
async def answer_question(question: str) -> str:
    """
    Kullanıcıdan gelen soruyu OpenAI ChatCompletion ile yanıtla.
    """
    try:
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",      # Plus kullanıcıları için limitsiz/ekonomik
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            temperature=0.7,
            max_tokens=512,
        )
        return response.choices[0].message.content.strip()

    except Exception as e:
        # Hata olursa logla ve kullanıcıya güvenli mesaj gönder
        print("OpenAI hatası:", e)
        return "❌ Cevap üretilirken bir hata oluştu. Lütfen tekrar deneyin."
