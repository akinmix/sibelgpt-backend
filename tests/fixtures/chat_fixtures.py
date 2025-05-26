# tests/fixtures/chat_fixtures.py
"""Chat testleri için sahte veriler"""

SAMPLE_QUESTIONS = [
    "Kadıköy'de ev arıyorum",
    "Konut kredisi nasıl alınır",
    "Numeroloji hakkında bilgi",
    "Bitcoin fiyatı nasıl",
    "Merhaba nasılsın"
]

SAMPLE_RESPONSES = [
    "Gayrimenkul uzmanı yanıtı",
    "Zihin koçu yanıtı", 
    "Finans uzmanı yanıtı"
]

SAMPLE_CONVERSATION_HISTORY = [
    {
        "role": "user",
        "text": "Merhaba"
    },
    {
        "role": "assistant", 
        "text": "Size nasıl yardımcı olabilirim?"
    }
]

CHAT_TEST_SCENARIOS = [
    {
        "name": "Gayrimenkul sorusu",
        "question": "Kadıköy'de 3+1 daire",
        "mode": "real-estate",
        "expected_in_response": ["daire", "Kadıköy"]
    },
    {
        "name": "Finans sorusu", 
        "question": "Bitcoin analizi",
        "mode": "finance",
        "expected_in_response": ["Bitcoin", "analiz"]
    }
]
