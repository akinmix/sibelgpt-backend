import pdfkit

def convert_md_to_pdf(md_path, pdf_path):
    try:
        with open(md_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        # Basit dönüştürme (örnek)
        pdfkit.from_string(html_content, pdf_path)
        print(f"PDF başarıyla oluşturuldu: {pdf_path}")
    except Exception as e:
        print(f"PDF oluşturulamadı: {e}")

