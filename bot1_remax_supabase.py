# -*- coding: utf-8 -*-
import time
import os
import re # Fiyat ve Lokasyon temizleme için
import random
from urllib.parse import urljoin, urlparse
from datetime import datetime, timezone # son_gorulme_tarihi için eklendi

# --- Gerekli Kütüphaneler (Önce yüklenmeli: pip install selenium webdriver-manager openai supabase python-dotenv) ---
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager # ChromeDriver'ı otomatik yönetmek için
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException, StaleElementReferenceException

from openai import OpenAI # OpenAI embedding için
from supabase import create_client, Client # Supabase bağlantısı için
from dotenv import load_dotenv # API anahtarlarını güvenli yönetmek için

# --- Ayarlar ---
load_dotenv() # .env dosyasındaki değişkenleri yükle

# -- Remax Ayarları --
START_URL = "https://www.remax.com.tr/ofis/detay/sonuc?t=satilik&type=1" # Senin Remax Sonuç ofisinin sayfası
BASE_URL = "https://www.remax.com.tr"
MAX_PAGES_TO_SCRAPE = 25 # Güvenlik için maksimum sayfa sınırı
MAX_LISTINGS_PER_RUN = 10 # Test için veya API limitlerini aşmamak için (None yaparsan hepsini işler)

# -- OpenAI Ayarları --
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-ada-002"

# -- Supabase Ayarları --
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_ANON_KEY") # Anon key kullanıyoruz
SUPABASE_TABLE_NAME = "ilanlar" # Tablo adımız

# --- Kontroller ---
if not OPENAI_API_KEY:
    print("HATA: OPENAI_API_KEY ortam değişkeni bulunamadı. Lütfen .env dosyasını kontrol edin.")
    exit()
if not SUPABASE_URL or not SUPABASE_KEY:
    print("HATA: SUPABASE_URL veya SUPABASE_ANON_KEY ortam değişkeni bulunamadı. Lütfen .env dosyasını kontrol edin.")
    exit()

# --- Yardımcı Fonksiyonlar ---

def clean_price(price_str):
    """Fiyat metnini temizler ve sayısal (float) değere dönüştürür."""
    if not price_str or price_str == "N/A" or "****" in price_str:
        return None
    cleaned_price = re.sub(r'[₺£$€.,]', '', price_str).strip()
    try:
        return float(cleaned_price)
    except ValueError:
        print(f"Uyarı: Fiyat dönüştürülemedi: '{price_str}'")
        return None

def extract_ilan_id(url):
    """URL'den ilan ID'sini (örn: P12345678) çıkarır."""
    try:
        path_parts = urlparse(url).path.split('/')
        if len(path_parts) > 1 and path_parts[-2] == 'portfoy':
            ilan_id = path_parts[-1]
            if re.match(r'^[A-Za-z]?\d+$', ilan_id):
                 return ilan_id
    except Exception as e:
        print(f"Uyarı: URL'den ilan ID'si çıkarılamadı ({url}): {e}")
    return None

def parse_location_string(location_str):
    """ 'Şehir / İlçe / Mahalle / Site' formatındaki metni ayrıştırır ve temizler. """
    if not location_str or location_str == "N/A":
        return {'sehir': None, 'ilce': None, 'mahalle': None, 'site_adi': None}

    parts = [part.strip() for part in location_str.split('/') if part.strip()]

    sehir = parts[0] if len(parts) > 0 else None
    ilce = parts[1] if len(parts) > 1 else None
    mahalle_raw = parts[2] if len(parts) > 2 else None
    site_adi_raw = parts[3] if len(parts) > 3 else None

    mahalle = None
    if mahalle_raw:
        mahalle = re.sub(r'\s+(Mah\.|Mh\.)$', '', mahalle_raw, flags=re.IGNORECASE).strip()

    site_adi = site_adi_raw.strip() if site_adi_raw else None

    return {'sehir': sehir, 'ilce': ilce, 'mahalle': mahalle, 'site_adi': site_adi}


def get_openai_embedding(text, client):
    """Verilen metin için OpenAI'dan embedding alır."""
    if not text or not isinstance(text, str): return None
    text = text.replace("\n", " ").strip()
    if not text: return None
    try:
        response = client.embeddings.create(input=[text], model=EMBEDDING_MODEL)
        return response.data[0].embedding
    except Exception as e:
        print(f"HATA: OpenAI embedding alınırken hata: {e}")
        return None

def init_supabase_client():
    """Supabase istemcisini başlatır."""
    try:
        print("Supabase istemcisi başlatılıyor...")
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        print("Supabase istemcisi başarıyla başlatıldı.")
        return supabase
    except Exception as e:
        print(f"HATA: Supabase istemcisi başlatılamadı: {e}")
        exit()

def scrape_listing_details(url):
    """Verilen URL'deki ilan detay sayfasını ziyaret edip ilanın açıklamasını alır."""
    print(f"   Detay sayfası ziyaret ediliyor: {url}")
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')
    options.add_experimental_option('excludeSwitches', ['enable-logging'])

    detail_driver = None
    description = None
    try:
        service = Service(ChromeDriverManager().install())
        detail_driver = webdriver.Chrome(service=service, options=options)
        detail_driver.get(url)
        wait = WebDriverWait(detail_driver, 30)

        try:
            description_header_xpath = "//h2[normalize-space()='İlan Açıklaması']"
            description_header = wait.until(EC.visibility_of_element_located((By.XPATH, description_header_xpath)))
            description_div_xpath = f"{description_header_xpath}/following-sibling::div[@class='content'][1]"
            description_div = wait.until(EC.presence_of_element_located((By.XPATH, description_div_xpath)))
            time.sleep(random.uniform(0.5, 1.0))
            description = description_div.text.strip()
            print(f"   Açıklama bulundu ({len(description)} karakter).")
        except (NoSuchElementException, TimeoutException):
            print(f"   Uyarı: İlan Açıklaması bulunamadı: {url}")
            description = None

    except Exception as e:
        print(f"HATA: Detay sayfası ({url}) çekilirken hata: {e}")
        description = None
    finally:
        if detail_driver:
            detail_driver.quit()

    return description

def insert_to_supabase(supabase_client, data):
    """Veriyi Supabase'e ekler/günceller."""
    try:
        # >>> YENİ EKLENEN SATIR: Son görülme tarihini ekle <<<
        data['son_gorulme_tarihi'] = datetime.now(timezone.utc).isoformat()

        response = supabase_client.table(SUPABASE_TABLE_NAME).upsert(
            data,
            on_conflict='ilan_id' # ilan_id'ye göre güncelle/ekle
        ).execute()
        if response.data or (hasattr(response, 'status_code') and 200 <= response.status_code < 300):
             print(f"   Başarılı: İlan ID {data.get('ilan_id')} Supabase'e eklendi/güncellendi.")
             return True
        else:
             print(f"   HATA: İlan ID {data.get('ilan_id')} Supabase'e yazılamadı. Yanıt: {response}")
             return False

    except Exception as e:
        print(f"HATA: Supabase'e veri yazılırken hata oluştu (ID: {data.get('ilan_id')}): {e}")
        import traceback
        traceback.print_exc()
        return False

# --- Ana Script ---
if __name__ == "__main__":
    print("Bot 1 - Remax Scraper, Embedder ve Supabase Writer Başlatılıyor...")

    driver = None
    scraped_urls = set()
    listings_processed_in_this_run = 0

    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    supabase_client = init_supabase_client()

    try:
        chrome_options = Options()
        user_agent = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        print(f"Kullanılan User Agent: {user_agent}")
        chrome_options.add_argument(f'user-agent={user_agent}')
        chrome_options.add_argument("--headless") # Render'da çalıştırırken headless OLMALI
        chrome_options.add_argument("--disable-gpu"); chrome_options.add_argument("--window-size=1920x1080") # Headless için boyut belirtmek iyi olabilir
        chrome_options.add_argument("--no-sandbox"); chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_experimental_option('excludeSwitches', ['enable-logging'])

        print("WebDriver (ChromeDriverManager ile) başlatılıyor...")
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)

        print(f"Başlangıç URL'sine gidiliyor: {START_URL}")
        driver.get(START_URL)
        initial_wait = random.uniform(5, 8)
        print(f"İlk yükleme için {initial_wait:.1f} saniye bekleniyor...")
        time.sleep(initial_wait)

        page_count = 1
        while page_count <= MAX_PAGES_TO_SCRAPE:
            print(f"\n--- Sayfa {page_count} taranıyor ---")
            wait = WebDriverWait(driver, 60)

            if MAX_LISTINGS_PER_RUN is not None and listings_processed_in_this_run >= MAX_LISTINGS_PER_RUN:
                print(f"Bu çalıştırma için maksimum ilan sayısına ({MAX_LISTINGS_PER_RUN}) ulaşıldı. Durduruluyor.")
                break

            try:
                list_container_selector = "div.result-list"
                print(f"İlan listesi konteynerı ({list_container_selector}) bekleniyor...")
                list_container = wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, list_container_selector)))
                print("İlan listesi konteyneri bulundu.")

                first_listing_item_selector = f"{list_container_selector} > div.item"
                print(f"İlk ilan birimi ({first_listing_item_selector}) bekleniyor...")
                # Varlığını beklemek yerine görünürlüğünü beklemek daha garanti olabilir
                wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, first_listing_item_selector)))
                print("İlk ilan birimi bulundu ve görünür.")

                render_wait = random.uniform(1, 3) # Bekleme süresini biraz azalttım
                print(f"İlanların render olması için {render_wait:.1f} saniye bekleniyor...")
                time.sleep(render_wait)

                listing_elements = driver.find_elements(By.CSS_SELECTOR, first_listing_item_selector)
                print(f"{len(listing_elements)} adet ilan birimi (div.item) bulundu.")

                for item_div in listing_elements:
                    if MAX_LISTINGS_PER_RUN is not None and listings_processed_in_this_run >= MAX_LISTINGS_PER_RUN: break

                    try:
                        info_link_element = item_div.find_element(By.CSS_SELECTOR, "a.info")
                        relative_url = info_link_element.get_attribute('href')
                        detail_url = urljoin(BASE_URL, relative_url) if relative_url else None

                        if detail_url and detail_url not in scraped_urls:
                            scraped_urls.add(detail_url)
                            print(f"\n-> Yeni İlan Bulundu: {detail_url}")

                            ilan_id = extract_ilan_id(detail_url)
                            if not ilan_id: print("   Uyarı: İlan ID alınamadı, atlanıyor."); continue

                            try:
                                lokasyon_str = info_link_element.find_element(By.CSS_SELECTOR, "div.breadcrumbs").text.strip()
                                parsed_location = parse_location_string(lokasyon_str)
                                print(f"   Lokasyon (Ham): '{lokasyon_str}' -> Parsed: {parsed_location}")
                            except NoSuchElementException: parsed_location = parse_location_string(None); print("   Lokasyon ana sayfada bulunamadı.")

                            try:
                                fiyat_str = info_link_element.find_element(By.CSS_SELECTOR, "div.price-container strong").text.strip()
                                fiyat_num = clean_price(fiyat_str)
                                print(f"   Fiyat (Ana Sayfa): {fiyat_str} -> {fiyat_num}")
                            except NoSuchElementException: fiyat_num = None; print("   Fiyat ana sayfada bulunamadı.")

                            ozet_text = scrape_listing_details(detail_url)

                            if ozet_text:
                                print("   Embedding oluşturuluyor...")
                                embedding_vector = get_openai_embedding(ozet_text, openai_client)

                                if embedding_vector:
                                    data_to_insert = {
                                        'ilan_id': ilan_id,
                                        'ozet': ozet_text,
                                        'url': detail_url,
                                        'embedding': embedding_vector,
                                        'fiyat': fiyat_num,
                                        'sehir': parsed_location.get('sehir'),
                                        'ilce': parsed_location.get('ilce'),
                                        'mahalle': parsed_location.get('mahalle'),
                                        'site_adi': parsed_location.get('site_adi')
                                        # son_gorulme_tarihi insert_to_supabase içinde eklenecek
                                    }
                                    if insert_to_supabase(supabase_client, data_to_insert):
                                         listings_processed_in_this_run += 1
                                else: print(f"   Uyarı: Embedding oluşturulamadığı için ilan ({ilan_id}) atlanıyor.")
                            else: print(f"   Uyarı: Detay sayfasından açıklama alınamadığı için ilan ({ilan_id}) atlanıyor.")
                        # else: pass
                    except NoSuchElementException: continue
                    except StaleElementReferenceException: print("   Uyarı: İlan elementi eskidi, atlanıyor."); continue
                    except Exception as inner_e: print(f"   Uyarı: İlan işlenirken hata: {inner_e}"); continue

                print(f"\nSayfa {page_count} tamamlandı. Bu çalıştırmada işlenen toplam yeni ilan: {listings_processed_in_this_run}")

            except TimeoutException: print(f"Hata: Sayfa {page_count} yüklenirken zaman aşımı."); break
            except Exception as page_e: print(f"Hata: Sayfa {page_count} işlenirken beklenmedik hata: {page_e}"); break

            if MAX_LISTINGS_PER_RUN is not None and listings_processed_in_this_run >= MAX_LISTINGS_PER_RUN: break

            try:
                pagination_selector = "#satiliklar > div.pagination"
                next_page_selector = f"{pagination_selector} > a.next"
                print(f"Pagination ({pagination_selector}) bekleniyor...")
                # Pagination konteynerinin görünür olmasını beklemek daha iyi olabilir
                pagination_container = WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.CSS_SELECTOR, pagination_selector)))
                print("Pagination bulundu.")
                # next_page_link = pagination_container.find_element(By.CSS_SELECTOR, "a.next") # Bazen class değişebilir, doğrudan 'Sonraki' metnini arayabiliriz
                next_page_link = pagination_container.find_element(By.LINK_TEXT, "Sonraki")

                print("Sonraki sayfa butonuna tıklanıyor...")
                driver.execute_script("arguments[0].scrollIntoView(true);", next_page_link); time.sleep(0.5)
                driver.execute_script("arguments[0].click();", next_page_link)
                page_count += 1
                page_load_wait = random.uniform(6, 10)
                print(f"Yeni sayfanın yüklenmesi için {page_load_wait:.1f} saniye bekleniyor...")
                time.sleep(page_load_wait)
            except NoSuchElementException: print("\nSonraki sayfa butonu bulunamadı. Son sayfaya ulaşıldı."); break
            except TimeoutException: print("\nPagination konteyneri bulunamadı veya zaman aşımına uğradı."); break
            except Exception as next_e: print(f"\nSonraki sayfaya geçerken hata: {next_e}"); break

    except TimeoutException: print(f"HATA: Başlangıç sayfası yüklenirken zaman aşımı: {START_URL}")
    except Exception as e: print(f"Beklenmedik bir ana hata oluştu: {e}"); import traceback; traceback.print_exc()
    finally:
        if driver: print("Tarayıcı kapatılıyor."); driver.quit()
        print(f"\n--- Tarama Tamamlandı ---")
        print(f"Bu çalıştırmada toplam {listings_processed_in_this_run} yeni ilan işlendi ve Supabase'e kaydedildi/güncellendi.")