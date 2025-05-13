# pdf_handler.py - REMAX İlanlarından PDF Oluşturma
import os
import httpx
import io
import re
from datetime import datetime
from typing import Dict, List, Optional
from fastapi import APIRouter, Response, HTTPException
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib.utils import ImageReader
from reportlab.lib.colors import HexColor

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

# FireCrawl API ayarları
FIRECRAWL_API_KEY = os.getenv("FIRECRAWL_API_KEY")
FIRECRAWL_URL = "https://api.firecrawl.dev/v0/scrape"

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
        "formats": ["markdown", "links"],
        "extractMainContent": True
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
    for line in lines[:5]:  # İlk 5 satırda başlığı ara
        if line.strip() and not line.startswith("#") and not line.startswith("["):
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
        'Banyo Sayısı': r'Banyo Sayısı\*\*\s*(\d+)'
    }
    
    for key, pattern in spec_patterns.items():
        match = re.search(pattern, content)
        if match:
            specs[key] = match.group(1).strip()
    
    # İlan Açıklaması
    desc_match = re.search(r'## İlan Açıklaması\s*(.*?)(?=##|\n\n\n|$)', content, re.DOTALL)
    description = desc_match.group(1).strip() if desc_match else ""
    
    # Fotoğraf URL'leri
    image_urls = []
    for link in links:
        if "i.remax.com.tr/photos" in link and (link.endswith('.jpg') or link.endswith('.png')):
            # Thumbnail'ları (T/) orjinal (L/) ile değiştir
            if '/T/' in link:
                link = link.replace('/T/', '/L/')
            image_urls.append(link)
    
    # Tekrarlı URL'leri kaldır ve ilk 12'yi al
    unique_images = list(dict.fromkeys(image_urls))[:12]
    
    return {
        'title': title,
        'portfoy_no': portfoy_no,
        'price': price,
        'specs': specs,
        'description': description,
        'images': unique_images
    }

def create_pdf(property_data: Dict) -> bytes:
    """Parse edilmiş veriden PDF oluşturur"""
    
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    # Renkler
    primary_color = HexColor('#1976d2')  # Mavi
    secondary_color = HexColor('#757575')  # Gri
    
    # UTF-8 encoding için
    c._doc.setProducer('SibelGPT PDF Generator')
    c._doc.setTitle('REMAX İlan Detayı')
    c._doc.setSubject('Gayrimenkul İlanı')
    
    # SAYFA 1: Başlık ve Genel Bilgiler
    # Header
    c.setFillColor(primary_color)
    c.rect(0, height-80, width, 80, fill=1)
    
    c.setFillColor('white')
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, height-50, clean_turkish_chars("SİBEL KAZAN MİDİLLİ"))
    c.setFont("Helvetica", 14)
    c.drawString(50, height-70, clean_turkish_chars("REMAX SONUÇ | Gayrimenkul Danışmanı"))
    
    # İlan Başlığı
    c.setFillColor('black')
    c.setFont("Helvetica-Bold", 16)
    y_pos = height - 120
    
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
    
    for line in title_lines[:3]:  # Maksimum 3 satır
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
    y_pos -= 30
    specs = property_data['specs']
    spec_keys = list(specs.keys())
    
    c.setFont("Helvetica", 11)
    col1_x = 50
    col2_x = width/2 + 20
    
    for i, key in enumerate(spec_keys):
        if i < len(spec_keys) / 2:
            x = col1_x
            y = y_pos - (i * 25)
        else:
            x = col2_x
            y = y_pos - ((i - len(spec_keys)//2) * 25)
        
        c.setFillColor(secondary_color)
        c.drawString(x, y, f"{clean_turkish_chars(key)}:")
        c.setFillColor('black')
        c.drawString(x + 100, y, clean_turkish_chars(specs[key]))
    
    # İlan Açıklaması
    y_pos -= (len(spec_keys)//2 + 1) * 25 + 20
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_pos, clean_turkish_chars("İLAN AÇIKLAMASI"))
    y_pos -= 5
    c.setLineWidth(2)
    c.line(50, y_pos, 200, y_pos)
    
    # Açıklama metni
    y_pos -= 20
    c.setFont("Helvetica", 10)
    desc_lines = clean_turkish_chars(property_data['description']).split('\n')
    for line in desc_lines:
        if y_pos < 100:  # Sayfa sonu kontrolü
            break
        if line.strip():
            # Uzun satırları kes
            while len(line) > 90:
                c.drawString(50, y_pos, line[:90])
                line = line[90:]
                y_pos -= 15
            c.drawString(50, y_pos, line)
            y_pos -= 15
    
    # Footer - İletişim
    c.setFillColor(primary_color)
    c.rect(0, 0, width, 60, fill=1)
    c.setFillColor('white')
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 30, clean_turkish_chars("İLETİŞİM: 532 687 84 64"))
    c.setFont("Helvetica", 12)
    c.drawString(50, 15, "sibelkazan@remax.com.tr | www.sibelgpt.com")
    
    # SAYFA 2: Fotoğraflar
    c.showPage()
    
    # Header (tekrar)
    c.setFillColor(primary_color)
    c.rect(0, height-60, width, 60, fill=1)
    c.setFillColor('white')
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height-40, clean_turkish_chars("FOTOĞRAFLAR"))
    
    # Fotoğrafları 3x4 grid halinde yerleştir
    if property_data['images']:
        images_to_show = property_data['images'][:12]
        
        grid_x = 50
        grid_y = height - 100
        img_width = (width - 120) / 3
        img_height = img_width * 0.75  # 4:3 oran
        
        for i, img_url in enumerate(images_to_show):
            row = i // 3
            col = i % 3
            
            x = grid_x + (col * (img_width + 10))
            y = grid_y - (row * (img_height + 10))
            
            # Fotoğraf çerçevesi
            c.setStrokeColor(secondary_color)
            c.setLineWidth(1)
            c.rect(x, y - img_height, img_width, img_height)
            
            # Fotoğraf numarası
            c.setFillColor(primary_color)
            c.setFont("Helvetica", 10)
            c.drawString(x + 5, y - img_height + 5, f"Foto {i+1}")
    
    # Footer (tekrar)
    c.setFillColor(primary_color)
    c.rect(0, 0, width, 60, fill=1)
    c.setFillColor('white')
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 30, clean_turkish_chars("İLETİŞİM: 532 687 84 64"))
    c.setFont("Helvetica", 12)
    c.drawString(50, 15, f"Olusturulma Tarihi: {datetime.now().strftime('%d.%m.%Y %H:%M')}")
    
    c.save()
    buffer.seek(0)
    return buffer.read()

@router.get("/generate-property-pdf/{property_id}")
async def generate_property_pdf(property_id: str):
    """REMAX ilan ID'si ile PDF oluşturur"""
    
    # 1. FireCrawl ile veriyi çek
    try:
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
    
    # 3. PDF oluştur
    try:
        pdf_bytes = create_pdf(property_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF oluşturma hatası: {str(e)}")
    
    # 4. PDF'i döndür
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename={property_id}_ilan.pdf"
        }
    )
