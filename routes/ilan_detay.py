from fastapi import APIRouter, Request
from controllers.ilan_controller import prepare_ilan_dosyasi

router = APIRouter()

@router.post("/api/ilan-detay")
async def ilan_detay(payload: dict):
    ilan_no = payload.get("ilan_no")
    if not ilan_no:
        return {"error": "ilan_no eksik"}

    result = prepare_ilan_dosyasi(ilan_no)
    return result
