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

