# pdf_handler.py - REMAX İlanlarından PDF Oluşturma (Fotoğraflı + Debug)
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
from PIL import Image

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
        "waitFor": 5000,  # 5 saniye bekle
        "screenshot": False,
        "onlyMainContent": False  # Tüm içeriği tara
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
    
    print(f"DEBUG: Toplam link sayısı: {len(links)}")  # Debug
    
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
        # Tüm resim linklerini al
        if any(ext in link.lower() for ext in ['.jpg', '.jpeg', '.png', '.webp']):
            print(f"DEBUG: Fotoğraf URL bulundu: {link}")
            # REMAX logo değilse ekle
            if "logo" not in link.lower():
                image_urls.append(link)
            
    # FireCrawl'ın bulduğu REMAX fotoğraf linklerini kontrol et
    for link in links:
        if "remax.com.tr/Content/Images" in link and link not in image_urls:
            print(f"DEBUG: REMAX fotoğraf URL bulundu: {link}")
            image_urls.append(link)
    
    # Tekrarlı URL'leri kaldır ve ilk 12'yi al
    unique_images = list(dict.fromkeys(image_urls))[:12]
    print(f"DEBUG: Toplam {len(unique_images)} benzersiz fotoğraf bulundu")  # Debug
    
    # Eğer fotoğraf bulunamazsa boş bırak
    if not unique_images:
        print("DEBUG: Fotoğraf bulunamadı, PDF fotoğrafsız oluşturulacak")
        unique_images = []
    
    return {
        'title': title,
        'portfoy_no': portfoy_no,
        'price': price,
        'specs': specs,
        'description': description,
        'images': unique_images
    }

async def download_image(url: str) -> Optional[bytes]:
    """Fotoğrafı indirir ve bytes olarak döndürür"""
    print(f"DEBUG: Fotoğraf indiriliyor: {url}")
    
    # REMAX URL'leri için sadece tek proxy kullan
    if "remax.com.tr" in url:
        # Sadece images.weserv.nl kullan
        proxy_url = f"https://images.weserv.nl/?url={url}"
        print(f"DEBUG: Proxy kullanılıyor: {proxy_url}")
        
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                response = await client.get(proxy_url)
                if response.status_code == 200:
                    print(f"DEBUG: Proxy başarılı: {len(response.content)} bytes")
                    return response.content
                else:
                    print(f"DEBUG: Proxy başarısız: {response.status_code}")
                    return None
        except Exception as e:
            print(f"DEBUG: Proxy hatası: {e}")
            return None
    
    # Diğer URL'ler için normal indir
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            print(f"DEBUG: Fotoğraf başarıyla indirildi: {len(response.content)} bytes")
            return response.content
    except Exception as e:
        print(f"DEBUG: Fotoğraf indirme hatası: {e}")
        return None

def create_pdf(property_data: Dict) -> bytes:
    """Parse edilmiş veriden PDF oluşturur (fotoğrafsız)"""
    
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
    
   # SAYFA 2: İletişim Bilgileri ve Fotoğraflar
    c.showPage()
    
    # Header
    c.setFillColor(primary_color)
    c.rect(0, height-80, width, 80, fill=1)
    c.setFillColor('white')
    c.setFont("Helvetica-Bold", 24)
    c.drawString(50, height-50, clean_turkish_chars("İLETİŞİM BİLGİLERİ"))
    
    # Sibel Hanım'ın fotoğrafı
    try:
        photo_url = "https://www.sibelgpt.com/sibel-kazan-midilli.jpg"
        response = httpx.get(photo_url, timeout=10)
        
        if response.status_code == 200:
            photo_data = io.BytesIO(response.content)
            img = ImageReader(photo_data)
            
            # Fotoğrafı sola yerleştir
            photo_width = 150
            photo_height = 200  # 3:4 oran
            photo_x = 50
            photo_y = height - 300
            
            c.drawImage(img, photo_x, photo_y, width=photo_width, height=photo_height, mask='auto')
            
            # Fotoğraf çerçevesi
            c.setStrokeColor(primary_color)
            c.setLineWidth(2)
            c.rect(photo_x-2, photo_y-2, photo_width+4, photo_height+4)
    except Exception as e:
        print(f"Fotoğraf yüklenemedi: {e}")
    
    # REMAX logosu
    try:
        logo_url = "https://www.sibelgpt.com/remax-logo.png"
        response = httpx.get(logo_url, timeout=10)
        
        if response.status_code == 200:
            logo_data = io.BytesIO(response.content)
            logo_img = ImageReader(logo_data)
            
            # Logoyu sağ üste yerleştir
            logo_width = 140
            logo_height = 93  # 200:133 oranını koruyarak
            logo_x = width - logo_width - 50
            logo_y = height - 180
            
            c.drawImage(logo_img, logo_x, logo_y, width=logo_width, height=logo_height, mask='auto')
    except Exception as e:
        print(f"Logo yüklenemedi: {e}")
    
    # İletişim bilgileri - sağ tarafta
    info_x = 250
    y_pos = height - 250
    
    c.setFont("Helvetica-Bold", 24)
    c.setFillColor(primary_color)
    c.drawString(info_x, y_pos, clean_turkish_chars("SİBEL KAZAN MİDİLLİ"))
    
    y_pos -= 30
    c.setFont("Helvetica", 16)
    c.setFillColor('black')
    c.drawString(info_x, y_pos, clean_turkish_chars("Gayrimenkul Danışmanı"))
    
    y_pos -= 40
    contact_info = [
        ("Telefon", "532 687 84 64"),
        ("E-posta", "sibelkazan@remax.com.tr"),
        ("Web", "www.sibelgpt.com"),
        ("Ofis", clean_turkish_chars("REMAX SONUÇ"))
    ]
    
    c.setFont("Helvetica", 14)
    for label, value in contact_info:
        c.setFillColor(secondary_color)
        c.drawString(info_x, y_pos, f"{label}:")
        c.setFillColor('black')
        c.drawString(info_x + 70, y_pos, value)
        y_pos -= 25
    
    # Alt bilgi kutusu
    box_y = 120
    c.setFillColor(HexColor('#f8f9fa'))
    c.rect(50, box_y, width-100, 100, fill=1)
    
    c.setFillColor(primary_color)
    c.setFont("Helvetica-Bold", 12)
    c.drawCentredString(width/2, box_y + 75, clean_turkish_chars("20 Yıllık Deneyim | Güvenilir Danışmanlık | Doğru Yatırım"))
    
    # Çalışma alanları
    y_pos = box_y + 50
    areas = [
        clean_turkish_chars("▸ Konut Alım-Satım"),
        clean_turkish_chars("▸ Yatırım Danışmanlığı"),
        clean_turkish_chars("▸ Değerleme Hizmetleri"),
        clean_turkish_chars("▸ Gayrimenkul Hukuku")
    ]
    
    c.setFont("Helvetica", 11)
    c.setFillColor('black')
    for i, area in enumerate(areas):
        x_offset = -150 if i % 2 == 0 else 50
        y_offset = -20 if i >= 2 else 0
        c.drawCentredString(width/2 + x_offset, y_pos + y_offset, area)
    
    # Footer
    c.setFillColor(primary_color)
    c.rect(0, 0, width, 60, fill=1)
    c.setFillColor('white')
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 30, clean_turkish_chars("İLETİŞİM: 532 687 84 64"))
    c.setFont("Helvetica", 12)
    c.drawString(50, 15, f"Olusturulma Tarihi: {datetime.now().strftime('%d.%m.%Y %H:%M')}")

async def create_pdf_with_images(property_data: Dict) -> bytes:
    """Parse edilmiş veriden fotoğraflı PDF oluşturur"""
    
    print(f"DEBUG: PDF oluşturma başlıyor, {len(property_data.get('images', []))} fotoğraf var")  # Debug
    
    # Önce fotoğrafları indirelim
    downloaded_images = []
    if property_data.get('images'):
        for i, img_url in enumerate(property_data['images'][:4]):  # İlk 4 fotoğraf
            print(f"DEBUG: Fotoğraf {i+1} indiriliyor: {img_url}")  # Debug
            img_data = await download_image(img_url)
            if img_data:
                print(f"DEBUG: Fotoğraf {i+1} başarıyla indirildi: {len(img_data)} bytes")  # Debug
                downloaded_images.append(img_data)
            else:
                print(f"DEBUG: Fotoğraf {i+1} indirilemedi")  # Debug
                downloaded_images.append(None)  # Placeholder için
    else:
        print("DEBUG: Hiç fotoğraf URL'si bulunamadı")  # Debug
    
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    # Renkler
    primary_color = HexColor('#1976d2')  # Mavi
    secondary_color = HexColor('#757575')  # Gri
    
    # SAYFA 1: Başlık ve Genel Bilgiler (öncekiyle aynı)
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
    
    for line in title_lines[:3]:
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
    
    # İlan Açıklaması (kısaltılmış)
    y_pos -= (len(spec_keys)//2 + 1) * 25 + 20
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y_pos, clean_turkish_chars("İLAN AÇIKLAMASI"))
    y_pos -= 5
    c.setLineWidth(2)
    c.line(50, y_pos, 200, y_pos)
    
    # Açıklama metni (daha kısa)
    y_pos -= 20
    c.setFont("Helvetica", 10)
    desc_lines = clean_turkish_chars(property_data['description'])[:300].split('\n')  # İlk 300 karakter
    for line in desc_lines:
        if y_pos < 100:
            break
        if line.strip():
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
    
    # Header
    c.setFillColor(primary_color)
    c.rect(0, height-60, width, 60, fill=1)
    c.setFillColor('white')
    c.setFont("Helvetica-Bold", 18)
    c.drawString(50, height-40, clean_turkish_chars("FOTOGRAFLAR"))
    
    # Fotoğrafları yerleştir
    grid_x = 50
    grid_y = height - 100
    img_width = (width - 120) / 2
    img_height = img_width * 0.75
    
    for i, img_data in enumerate(downloaded_images[:4]):
        row = i // 2
        col = i % 2
        
        x = grid_x + (col * (img_width + 20))
        y = grid_y - (row * (img_height + 20))
        
        # Çerçeve
        c.setStrokeColor(secondary_color)
        c.setLineWidth(1)
        c.rect(x, y - img_height, img_width, img_height)
        
        if img_data:
            try:
                # Fotoğrafı PDF'e ekle
                img_buffer = io.BytesIO(img_data)
                img = Image.open(img_buffer)
                
                # Boyutlandırma
                img_ratio = img.width / img.height
                pdf_ratio = img_width / img_height
                
                if img_ratio > pdf_ratio:
                    # Geniş fotoğraf - genişliğe göre sığdır
                    new_width = img_width
                    new_height = img_width / img_ratio
                else:
                    # Uzun fotoğraf - yüksekliğe göre sığdır
                    new_height = img_height
                    new_width = img_height * img_ratio
                
                # Ortala
                x_offset = x + (img_width - new_width) / 2
                y_offset = y - img_height + (img_height - new_height) / 2
                
                # PDF'e ekle
                img_buffer.seek(0)
                img_reader = ImageReader(img_buffer)
                c.drawImage(img_reader, x_offset, y_offset, new_width, new_height)
                
                print(f"DEBUG: Fotoğraf {i+1} PDF'e eklendi")  # Debug
                
            except Exception as e:
                print(f"HATA: Fotoğraf {i+1} ekleme hatası: {e}")
                # Hata durumunda placeholder
                c.setFillColor(secondary_color)
                c.setFont("Helvetica", 12)
                c.drawCentredString(x + img_width/2, y - img_height/2, "Fotograf yuklenemedi")
        else:
            # Placeholder
            c.setFillColor(secondary_color)
            c.setFont("Helvetica", 12)
            c.drawCentredString(x + img_width/2, y - img_height/2, "Fotograf bulunamadi")
        
        # Fotoğraf numarası
        c.setFillColor(primary_color)
        c.setFont("Helvetica-Bold", 10)
        c.drawString(x + 5, y - 15, f"Foto {i+1}")
    
    # Footer
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

# Test endpoint
@router.get("/test-firecrawl/{property_id}")
async def test_firecrawl(property_id: str):
    """FireCrawl'dan gelen veriyi test eder"""
    
    try:
        firecrawl_data = await scrape_property_with_firecrawl(property_id)
        
        # Tüm veri yapısını incele
        data_keys = list(firecrawl_data.get("data", {}).keys())
        links = firecrawl_data.get("data", {}).get("links", [])
        html_content = firecrawl_data.get("data", {}).get("html", "")
        
        # HTML'de fotoğraf ara
        photo_patterns = []
        if html_content:
            import re
            patterns = [
                r'src=["\']([^"\']+remax[^"\']+photos[^"\']+)["\']',
                r'data-src=["\']([^"\']+remax[^"\']+photos[^"\']+)["\']',
                r'(https?://[^"\'\s]+remax[^"\'\s]+photos[^"\'\s]+)',
                r'(https?://[^"\'\s]+\.jpg)',
                r'(https?://[^"\'\s]+\.png)'
            ]
            
            for pattern in patterns:
                matches = re.findall(pattern, html_content[:5000])  # İlk 5000 karakter
                if matches:
                    photo_patterns.append({
                        "pattern": pattern,
                        "count": len(matches),
                        "samples": matches[:3]
                    })
        
        photo_links = []
        for link in links:
            if "photos" in link or ".jpg" in link or ".png" in link:
                photo_links.append(link)
        
        return {
            "available_data_keys": data_keys,
            "total_links": len(links),
            "photo_links_count": len(photo_links),
            "photo_links": photo_links[:10],
            "html_available": bool(html_content),
            "html_length": len(html_content) if html_content else 0,
            "html_photo_patterns": photo_patterns,
            "sample_links": links[:20]
        }
    except Exception as e:
        return {"error": str(e), "type": type(e).__name__}

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
    
    # 3. PDF oluştur (fotoğraflı)
    try:
        pdf_bytes = await create_pdf_with_images(property_data)
    except Exception as e:
        # Hata durumunda fotoğrafsız PDF oluştur
        print(f"Fotoğraflı PDF hatası: {e}")
        pdf_bytes = create_pdf(property_data)
    
    # 4. PDF'i döndür
    return Response(
        content=pdf_bytes,
        media_type="application/pdf",
        headers={
            "Content-Disposition": f"attachment; filename={property_id}_ilan.pdf"
        }
    )
    # Test PDF bileşenleri endpoint
    @router.get("/test-pdf-components/{property_id}")
    async def test_pdf_components(property_id: str):
    """PDF bileşenlerini test eder"""
    
    test_results = {
        "sibel_photo": False,
        "remax_logo": False,
        "property_data": False
    }
    
    # Fotoğraf testi
    try:
        photo_response = httpx.get("https://www.sibelgpt.com/sibel-kazan-midilli.jpg")
        test_results["sibel_photo"] = photo_response.status_code == 200
    except:
        pass
    
    # Logo testi
    try:
        logo_response = httpx.get("https://www.sibelgpt.com/remax-logo.png")
        test_results["remax_logo"] = logo_response.status_code == 200
    except:
        pass
    
    # İlan verisi testi
    try:
        data = await scrape_property_with_firecrawl(property_id)
        test_results["property_data"] = bool(data)
    except:
        pass
    
    return test_results
