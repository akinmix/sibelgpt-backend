# pdf_handler.py - REMAX İlanlarından PDF Oluşturma (Tek Sayfa, Kompakt)
import os
import httpx
import io
import re
from datetime import datetime
from typing import Dict, List, Optional
from pathlib import Path
from fastapi import APIRouter, Response, HTTPException 
from fastapi.responses import StreamingResponse 
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import HexColor
from PIL import Image

# ---- PDF Saklama Dizini ----
APP_ROOT = Path(__file__).parent
PDF_STORAGE_DIR = Path('./pdf_storage')
os.makedirs(PDF_STORAGE_DIR, exist_ok=True)
router = APIRouter()

# FireCrawl API ayarları
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
FIRECRAWL_URL = "https://api.firecrawl.dev/v0/scrape"

def clean_turkish_chars(text: str) -> str:
    """Türkçe karakterleri ASCII'ye çevirir"""
    replacements = {
        'ç': 'c', 'Ç': 'C',
        'ğ': 'g', 'Ğ': 'G',
        'ı': 'i', 'İ': 'I',
        'ö': 'o', 'Ö': 'O',
        'ş': 's', 'Ş': 'S',
        'ü': 'u', 'Ü': 'U'
    }
    for turkish, ascii_char in replacements.items():
        text = text.replace(turkish, ascii_char)
    return text

async def scrape_property_with_firecrawl(property_id: str) -> Dict:
    """FireCrawl kullanarak REMAX ilan verilerini çeker"""
    
    if not FIRECRAWL_API_KEY:
        raise HTTPException(status_code=500, detail="FireCrawl API anahtarı eksik")
    
    url = f"https://remax.com.tr/portfoy/{property_id}"
    
    headers = {
        "Authorization": f"Bearer {FIRECRAWL_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "url": url,
        "formats": ["markdown", "links", "html"],
        "extractMainContent": True,
        "waitFor": 5000,
        "screenshot": False,
        "onlyMainContent": False
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(FIRECRAWL_URL, json=payload, headers=headers)
            response.raise_for_status()
            return response.json()
        except httpx.HTTPError as e:
            raise HTTPException(status_code=500, detail=f"FireCrawl hatası: {str(e)}")

def parse_property_data(firecrawl_data: Dict) -> Dict:
    """FireCrawl verisinden gerekli bilgileri parse eder"""
    
    content = firecrawl_data.get("data", {}).get("markdown", "")
    links = firecrawl_data.get("data", {}).get("links", [])
    
    # Başlık
    lines = content.split("\n")
    title = ""
    for line in lines[:5]:
        if line.strip() and not line.startswith("#") and not line.startswith("["):
            # "Giriş Yap" ve benzeri gereksiz metinleri atla
            if line.strip().lower() not in ["giriş yap", "giris yap", "login", "sign in", "üye ol"]:
                title = line.strip()
                break
    
    # Portföy No
    portfoy_match = re.search(r'Portföy No:\s*([A-Z0-9]+)', content)
    portfoy_no = portfoy_match.group(1) if portfoy_match else ""
    
    # Fiyat
    price_match = re.search(r'\*\*\s*([0-9.]+)\s*₺\s*\*\*', content)
    price = price_match.group(1) if price_match else "Belirtilmemiş"
    
    # Teknik Özellikler
    specs = {}
    spec_patterns = {
        'Emlak Tipi': r'Emlak Tipi\*\*\s*([\w\s]+)',
        'm² (Brüt)': r'm² \(Brüt\)\*\*\s*(\d+)',
        'm² (Net)': r'm² \(Net\)\*\*\s*(\d+)',
        'Oda Sayısı': r'Oda Sayısı\*\*\s*([\d+\w\s]+)',
        'Bina Yaşı': r'Bina Yaşı\*\*\s*([\d\-\s\w]+)',
        'Bulunduğu Kat': r'Bulunduğu Kat\*\*\s*([\w\s]+)',
        'Kat Sayısı': r'Kat Sayısı\*\*\s*(\d+)',
        'Isıtma': r'Isıtma\*\*\s*([\w\s\(\)]+)',
        'Banyo Sayısı': r'Banyo Sayısı\*\*\s*(\d+)',
        'Balkon': r'Balkon\*\*\s*([\w\s]+)',
        'Kullanım Durumu': r'Kullanım Durumu\*\*\s*([\w\s]+)',
        'Site İçerisinde': r'Site İçerisinde\*\*\s*([\w\s]+)',
        'Krediye Uygun': r'Krediye Uygun\*\*\s*([\w\s]+)'
    }
    
    for key, pattern in spec_patterns.items():
        match = re.search(pattern, content)
        if match:
            specs[key] = match.group(1).strip()
    
    # İlan Açıklaması
    desc_match = re.search(r'## İlan Açıklaması\s*(.*?)(?=##|\n\n\n|$)', content, re.DOTALL)
    description = desc_match.group(1).strip() if desc_match else ""
    
    return {
        'title': title,
        'portfoy_no': portfoy_no,
        'price': price,
        'specs': specs,
        'description': description
    }

async def create_compact_pdf(property_data: Dict) -> bytes:
    """Parse edilmiş veriden kompakt tek sayfalık PDF oluşturur"""
    
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    # Renkler
    primary_color = HexColor('#1976d2')
    secondary_color = HexColor('#757575')
    
    # UTF-8 encoding için
    c._doc.setProducer('SibelGPT PDF Generator')
    c._doc.setTitle('REMAX İlan Detayı')
    c._doc.setSubject('Gayrimenkul İlanı')
    
    # Header (Sibel foto ve REMAX logo ile)
    header_height = 100
    c.setFillColor(primary_color)
    c.rect(0, height-header_height, width, header_height, fill=1)
    
    # Sibel Hanım'ın fotoğrafı - SOL TARAFA
    try:
        photo_url = "https://www.sibelgpt.com/sibel-kazan-midilli.jpg"
        async with httpx.AsyncClient() as client:
            response = await client.get(photo_url, timeout=10)
        
        if response.status_code == 200:
            photo_data = io.BytesIO(response.content)
            img = ImageReader(photo_data)
            
            # Fotoğrafı header içinde sol tarafa yerleştir
            photo_height = 70
            photo_width = 52  # 3:4 oran korunarak
            photo_x = 20
            photo_y = height - header_height + 15
            
            c.drawImage(img, photo_x, photo_y, width=photo_width, height=photo_height, mask='auto')
    except Exception as e:
        print(f"Fotoğraf yüklenemedi: {e}")
    
    # Başlık metni - ORTADA
    c.setFillColor('white')
    c.setFont("Helvetica-Bold", 22)
    c.drawCentredString(width/2, height-40, clean_turkish_chars("SİBEL KAZAN MİDİLLİ"))
    c.setFont("Helvetica", 14)
    c.drawCentredString(width/2, height-60, clean_turkish_chars("REMAX SONUÇ | Gayrimenkul Danışmanı"))
    c.setFont("Helvetica", 12)
    c.drawCentredString(width/2, height-80, "Tel: 532 687 84 64 | sibel@sibelizim.com")
    
    # REMAX logosu - SAĞ TARAFA
    try:
        logo_url = "https://www.sibelgpt.com/remax-logo.png"
        async with httpx.AsyncClient() as client:
            response = await client.get(logo_url, timeout=10)
        
        if response.status_code == 200:
            logo_data = io.BytesIO(response.content)
            logo_img = ImageReader(logo_data)
            
            # Logoyu header içinde sağ tarafa yerleştir
            logo_height = 50
            logo_width = 75  # 200:133 oran korunarak
            logo_x = width - logo_width - 20
            logo_y = height - header_height + 25
            
            c.drawImage(logo_img, logo_x, logo_y, width=logo_width, height=logo_height, mask='auto')
    except Exception as e:
        print(f"Logo yüklenemedi: {e}")
    
    # İlan Başlığı
    c.setFillColor('black')
    c.setFont("Helvetica-Bold", 16)
    y_pos = height - header_height - 30
    
    # Başlığı satırlara böl
    title_lines = []
    words = clean_turkish_chars(property_data['title']).split()
    current_line = ""
    for word in words:
        test_line = current_line + " " + word if current_line else word
        if c.stringWidth(test_line, "Helvetica-Bold", 16) < width - 100:
            current_line = test_line
        else:
            if current_line:
                title_lines.append(current_line)
            current_line = word
    if current_line:
        title_lines.append(current_line)
    
    for line in title_lines[:2]:  # Maksimum 2 satır
        c.drawString(50, y_pos, line)
        y_pos -= 20
    
    # Portföy No ve Fiyat
    y_pos -= 10
    c.setFont("Helvetica", 12)
    c.setFillColor(secondary_color)
    c.drawString(50, y_pos, f"Portfoy No: {property_data['portfoy_no']}")
    
    c.setFont("Helvetica-Bold", 20)
    c.setFillColor(primary_color)
    c.drawString(width - 200, y_pos, f"{property_data['price']} TL")
    
    # Özellikler Tablosu
    y_pos -= 40
    c.setFont("Helvetica-Bold", 14)
    c.setFillColor('black')
    c.drawString(50, y_pos, clean_turkish_chars("TEKNİK ÖZELLİKLER"))
    y_pos -= 5
    c.setStrokeColor(primary_color)
    c.setLineWidth(2)
    c.line(50, y_pos, 200, y_pos)
    
    # Özellikleri 2 sütun halinde göster
    y_pos -= 25
    specs = property_data['specs']
    spec_keys = list(specs.keys())
    
    c.setFont("Helvetica", 11)
    col1_x = 50
    col2_x = width/2 + 20
    
    for i, key in enumerate(spec_keys):
        if i < len(spec_keys) / 2:
            x = col1_x
            y = y_pos - (i * 22)
        else:
            x = col2_x
            y = y_pos - ((i - len(spec_keys)//2) * 22)
        
        c.setFillColor(secondary_color)
        c.drawString(x, y, f"{clean_turkish_chars(key)}:")
        c.setFillColor('black')
        c.drawString(x + 100, y, clean_turkish_chars(specs[key]))
    
    # İlan Açıklaması
    y_pos -= (len(spec_keys)//2 + 1) * 22 + 20
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_pos, clean_turkish_chars("İLAN AÇIKLAMASI"))
    y_pos -= 5
    c.setLineWidth(2)
    c.line(50, y_pos, 200, y_pos)
    
    # Açıklama metni
    y_pos -= 20
    c.setFont("Helvetica", 10)
    desc_lines = clean_turkish_chars(property_data['description']).split('\n')
    
    # Sayfa sonuna kadar olan kalan alanı hesapla
    remaining_space = y_pos - 80  # 80 px footer için
    line_height = 14
    max_lines = int(remaining_space / line_height)
    
    lines_written = 0
    for line in desc_lines:
        if lines_written >= max_lines:
            break
        if line.strip():
            while len(line) > 85 and lines_written < max_lines:
                c.drawString(50, y_pos, line[:85])
                line = line[85:]
                y_pos -= line_height
                lines_written += 1
            if lines_written < max_lines:
                c.drawString(50, y_pos, line)
                y_pos -= line_height
                lines_written += 1
    
    # Alt bilgi kutusu
    footer_height = 50
    c.setFillColor(HexColor('#f8f9fa'))
    c.rect(0, 0, width, footer_height, fill=1)
    
    c.setFillColor(primary_color)
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(width/2, 30, clean_turkish_chars("REMAX SONUÇ - Remax Sonuç'ta hiçbir işiniz SONUÇSUZ kalmaz"))
    
    c.setFillColor('black')
    c.setFont("Helvetica", 10)
    c.drawCentredString(width/2, 15, f"Bu PDF {datetime.now().strftime('%d.%m.%Y %H:%M')} tarihinde olusturulmustur.")
    
    c.save()
    buffer.seek(0)
    return buffer.read()

@router.get("/generate-property-pdf/{property_id}")
async def generate_property_pdf(property_id: str):
    """REMAX ilan ID'si ile PDF oluşturur"""
    
    # PDF'in depolama yolunu belirle
    pdf_path = PDF_STORAGE_DIR / f"{property_id}.pdf"
    
    # PDF var mı kontrol et
    if pdf_path.exists():
        # PDF varsa doğrudan döndür
        print(f"✅ Önceden oluşturulmuş PDF bulundu: {property_id}.pdf")
        with open(pdf_path, 'rb') as f:
            content = f.read()
        return StreamingResponse(
            io.BytesIO(content),
            media_type="application/pdf",
            headers={
                "Content-Disposition": f"attachment; filename={property_id}_ilan.pdf"
            }
        )
    
    # PDF yoksa oluştur ve kaydet
    try:
        print(f"🔍 PDF bulunamadı, yeni PDF oluşturuluyor: {property_id}")
        # 1. FireCrawl ile veriyi çek
        firecrawl_data = await scrape_property_with_firecrawl(property_id)
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Veri çekme hatası: {str(e)}")
    
    # 2. Veriyi parse et
    try:
        property_data = parse_property_data(firecrawl_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Veri işleme hatası: {str(e)}")
    
    # 3. Kompakt PDF oluştur
    try:
        pdf_bytes = await create_compact_pdf(property_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF oluşturma hatası: {str(e)}")
    
    # 4. PDF'i dosyaya kaydet
    try:
        with open(pdf_path, "wb") as f:
            f.write(pdf_bytes)
        print(f"💾 PDF başarıyla kaydedildi: {property_id}.pdf")
    except Exception as e:
        print(f"❌ PDF kaydetme hatası: {str(e)}")
    
    # 5. PDF'i döndür
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename={property_id}_ilan.pdf"
        }
    )

@router.get("/test-firecrawl/{property_id}")
async def test_firecrawl(property_id: str):
    """FireCrawl'dan gelen veriyi test eder"""
    
    try:
        firecrawl_data = await scrape_property_with_firecrawl(property_id)
        
        data_keys = list(firecrawl_data.get("data", {}).keys())
        links = firecrawl_data.get("data", {}).get("links", [])
        
        return {
            "available_data_keys": data_keys,
            "total_links": len(links),
            "sample_links": links[:20]
        }
    except Exception as e:
        return {"error": str(e), "type": type(e).__name__}
