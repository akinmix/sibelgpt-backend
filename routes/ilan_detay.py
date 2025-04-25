import os
import requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

router = APIRouter()

# 🔹 Model
class IlanRequest(BaseModel):
    ilan_no: str

# 🔹 Yardımcı Fonksiyon: Veriyi filtrele
def filtrele_scrape_verisi(firecrawl_json: dict):
    try:
        md_text = firecrawl_json['data']['markdown']
        sonuc = {}

        if "## İlan Açıklaması" in md_text:
            aciklama = md_text.split("## İlan Açıklaması")[1].split("##")[0].strip()
            sonuc["aciklama"] = aciklama

        if "![KADIKÖY" in md_text:
            ilk_foto = md_text.split("![")[1].split("](")[1].split(")")[0]
            sonuc["fotograflar"] = [ilk_foto]

        for satir in md_text.splitlines():
            if "₺" in satir:
                sonuc["fiyat"] = satir.strip()
            if "Oda Sayısı" in satir:
                sonuc["oda"] = satir.split("**")[-1]
            if "m² (Brüt)" in satir:
                sonuc["m2"] = satir.split("**")[-1]

        return sonuc

    except Exception as e:
        raise ValueError(f"Firecrawl verisi işlenemedi: {e}")

# 🔹 Endpoint
@router.post("/api/ilan-detay")
def ilan_detay_cek(req: IlanRequest):
    FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
    headers = {
        "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
        "Content-Type": "application/json"
    }

    url = f"https://www.remax.com.tr/portfoy/{req.ilan_no}"
    payload = {
        "url": url,
        "options": {
            "render": True,
            "waitFor": 2000,
            "formats": ["markdown", "html", "links", "rawHtml"]
        }
    }

    try:
        res = requests.post("https://api.firecrawl.dev/scrape", headers=headers, json=payload)
        if res.status_code != 200:
            raise HTTPException(status_code=500, detail="Firecrawl'dan veri alınamadı")

        filtered_data = filtrele_scrape_verisi(res.json())
        return {"ilan_no": req.ilan_no, "veri": filtered_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
