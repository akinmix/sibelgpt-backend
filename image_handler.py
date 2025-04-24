# image_handler.py
from fastapi import APIRouter, Request
from pydantic import BaseModel
import openai
import os

router = APIRouter()

openai.api_key = os.getenv("OPENAI_API_KEY")

class ImageRequest(BaseModel):
    prompt: str

@router.post("/image")
async def generate_image(req: ImageRequest):
    try:
        response = openai.images.generate(
            model="dall-e-3",
            prompt=req.prompt,
            n=1,
            quality="standard",  # hızlı ve düşük maliyetli üretim
            size="1024x1024"
        )
        image_url = response.data[0].url
        return {"image_url": image_url}
    except Exception as e:
        return {"error": str(e)}
