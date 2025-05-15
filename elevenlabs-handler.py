# elevenlabs_handler.py
import os
import httpx
from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import io

router = APIRouter()

# ElevenLabs API ayarları
ELEVENLABS_API_KEY = os.getenv("ELEVENLABS_API_KEY")
ELEVENLABS_VOICE_ID = "EJGs6dWlD5VrB3llhBqB"  # Sibel Hanım'ın klonlanmış ses ID'si
ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech"

class SpeechRequest(BaseModel):
    text: str
    voice_settings: dict = {
        "stability": 0.75,
        "similarity_boost": 0.85
    }

@router.post("/generate-speech")
async def generate_speech(request: SpeechRequest):
    """Metni sese dönüştürür"""
    
    if not ELEVENLABS_API_KEY:
        raise HTTPException(status_code=500, detail="ElevenLabs API key bulunamadı")
    
    # API endpoint
    url = f"{ELEVENLABS_API_URL}/{ELEVENLABS_VOICE_ID}"
    
    headers = {
        "Accept": "audio/mpeg",
        "Content-Type": "application/json",
        "xi-api-key": ELEVENLABS_API_KEY
    }
    
    payload = {
        "text": request.text,
        "model_id": "eleven_multilingual_v2",
        "voice_settings": request.voice_settings
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(url, json=payload, headers=headers)
            response.raise_for_status()
            
            # Ses verisini stream olarak döndür
            audio_content = response.content
            
            return StreamingResponse(
                io.BytesIO(audio_content),
                media_type="audio/mpeg",
                headers={
                    "Content-Disposition": "inline; filename=speech.mp3",
                    "Cache-Control": "public, max-age=3600"  # 1 saat önbellek
                }
            )
            
        except httpx.HTTPError as e:
            if "quota" in str(e).lower():
                raise HTTPException(status_code=429, detail="Aylık ses kotası doldu")
            else:
                raise HTTPException(status_code=500, detail=f"Ses oluşturma hatası: {str(e)}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Beklenmeyen hata: {str(e)}")

@router.get("/voice-info")
async def get_voice_info():
    """Ses bilgilerini döndürür"""
    return {
        "voice_id": ELEVENLABS_VOICE_ID,
        "voice_name": "Sibel",
        "model": "eleven_multilingual_v2",
        "language": "tr-TR"
    }
