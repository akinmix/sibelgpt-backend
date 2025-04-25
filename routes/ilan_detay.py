from fastapi import APIRouter
from pydantic import BaseModel
from controllers.ilan_controller import prepare_ilan_dosyasi

router = APIRouter()

class IlanDetayRequest(BaseModel):
    ilan_no: str

@router.post("/api/ilan-detay")
async def ilan_detay(payload: IlanDetayRequest):
    ilan_no = payload.ilan_no
    result = prepare_ilan_dosyasi(ilan_no)
    return result
