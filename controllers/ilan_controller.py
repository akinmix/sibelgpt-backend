import csv
import os
from controllers.remax_scraper import scrape_remax_listing, save_to_markdown
from controllers.convert_md_to_pdf import convert_md_to_pdf  # varsa yoksa ekleriz

CSV_PATH = "markdowns/ilanlar.csv"
MARKDOWN_DIR = "markdowns"

def get_url_from_csv(ilan_no):
    try:
        with open(CSV_PATH, newline='', encoding="utf-8") as f:
            reader = csv.DictReader(f)
            print("[DEBUG] CSV Başlıkları:", reader.fieldnames)
            for row in reader:
                if row["ilan_no"] == ilan_no:
                    return row["URL"]
    except Exception as e:
        print(f"[HATA][get_url_from_csv] CSV okuma hatası: {e}")
    return None

def prepare_ilan_dosyasi(ilan_no):
    try:
        url = get_url_from_csv(ilan_no)
        if not url:
            return {"error": "İlan bulunamadı."}

        data = scrape_remax_listing(url)

        md_file = os.path.join(MARKDOWN_DIR, f"{ilan_no}.md")
        pdf_file = os.path.join(MARKDOWN_DIR, f"{ilan_no}.pdf")

        save_to_markdown(data, md_file)
        convert_md_to_pdf(md_file, pdf_file)

        return {
            "success": True,
            "ilan_no": ilan_no,
            "url": url,
            "md_path": f"/markdowns/{ilan_no}.md",
            "pdf_path": f"/markdowns/{ilan_no}.pdf"
        }
    except Exception as e:
        return {"error": f"Hata oluştu: {str(e)}"}
import os
import requests
from fastapi import HTTPException
from dotenv import load_dotenv

load_dotenv()

def prepare_ilan_dosyasi_firecrawl(ilan_no):
    firecrawl_api_key = os.getenv("FIRECRAWL_API_KEY")
    if not firecrawl_api_key:
        raise HTTPException(status_code=500, detail="Firecrawl API anahtarı eksik")

    url = f"https://www.remax.com.tr/portfoy/{ilan_no}"
    payload = {
        "url": url,
        "options": {
            "extractOnlyMainContent": True,
            "outputFormat": ["markdown"],
            "excludeTags": ["script", ".ad", "#footer"],
            "waitFor": 1000,
            "timeout": 30000
        }
    }

    response = requests.post(
        "https://api.firecrawl.dev/scrape",
        headers={"Authorization": f"Bearer {firecrawl_api_key}"},
        json=payload
    )

    print("🔥 Firecrawl response status:", response.status_code)
    print("🔥 Firecrawl response body:", response.text)

    if response.status_code != 200:
        raise HTTPException(status_code=500, detail="Firecrawl'dan veri alınamadı")

    data = response.json()
    markdown = data.get("markdown", "")

    return {
        "ilan_no": ilan_no,
        "veri": {
            "aciklama": markdown[:1500]
        }
    }
