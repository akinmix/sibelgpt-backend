import time
import json # JSON formatında yazdırmak isterseniz
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import NoSuchElementException, TimeoutException

# Önceki decode_cfemail fonksiyonu buraya gelecek
def decode_cfemail(encodedString):
    """
    Cloudflare'ın e-posta gizleme tekniğini çözmeye çalışır.
    Not: Bu yöntem her zaman çalışmayabilir veya gelecekte değişebilir.
    """
    try:
        k = int(encodedString[:2], 16)
        decodedString = ''.join([chr(int(encodedString[i:i+2], 16) ^ k) for i in range(2, len(encodedString), 2)])
        return decodedString
    except Exception as e:
        #print(f"Cloudflare email çözülemedi: {e}") # Hata ayıklama için açık kalabilir
        return f"[E-posta çözülemedi: {encodedString}]" # Hata durumunda kodlu hali veya bir not göster


# Önceki scrape_remax_listing fonksiyonu buraya gelecek
# (Fonksiyonun tamamını buraya kopyalayın)
def scrape_remax_listing(url):
    """
    Belirtilen Remax ilan URL'sinden bilgileri çeker.
    """
    print(f"İlan sayfası yükleniyor: {url}")

    # WebDriver Options (User-Agent ekleme)
    options = webdriver.ChromeOptions()
    # Headless modu (tarayıcı penceresi olmadan çalıştırır) - test ederken kapatabilirsiniz
    # options.add_argument('--headless') # Markdown çıktısı alırken headless daha iyi olabilir
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    # Akıllı User-Agent: Tarayıcı gibi görünmek için yaygın bir User-Agent kullanır
    options.add_argument('user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')

    driver = None
    listing_data = {}
    listing_data['URL'] = url # URL'yi de kaydedelim

    try:
        # WebDriver'ı başlat (WebDriver Manager otomatik olarak uygun driver'ı indirir)
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)

        driver.get(url)

        # Dinamik içeriğin yüklenmesini beklemek için explicit wait kullanma
        wait = WebDriverWait(driver, 15) # Bekleme süresini biraz ayarlayabilirsiniz

        # Başlığın yüklenmesini bekle ve çek
        try:
            title_element = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, 'h1#propertyTitle')))
            listing_data['Başlık'] = title_element.text.strip()
        except TimeoutException:
            listing_data['Başlık'] = 'Başlık Bulunamadı (Zaman Aşımı)'
        except NoSuchElementException:
            listing_data['Başlık'] = 'Başlık Elementi Bulunamadı'
        # print(f"Başlık: {listing_data.get('Başlık', 'Yok')}") # Debug çıktısı

        # Fiyatın yüklenmesini bekle ve çek
        try:
            price_element = wait.until(EC.visibility_of_element_located((By.CSS_SELECTOR, 'strong.price-share')))
            listing_data['Fiyat'] = price_element.text.strip()
        except TimeoutException:
             listing_data['Fiyat'] = 'Fiyat Bulunamadı (Zaman Aşımı)'
        except NoSuchElementException:
             listing_data['Fiyat'] = 'Fiyat Elementi Bulunamadı'
        # print(f"Fiyat: {listing_data.get('Fiyat', 'Yok')}") # Debug çıktısı


        # Konum bilgilerini çek
        try:
            location_element = driver.find_element(By.CSS_SELECTOR, '.breadcrumbs')
            listing_data['Konum'] = location_element.text.strip()
        except NoSuchElementException:
            listing_data['Konum'] = 'Konum Bulunamadı'
        # print(f"Konum: {listing_data.get('Konum', 'Yok')}") # Debug çıktısı


        # Ana özellikler (Spotlight Props) çekme
        listing_data['Ana Özellikler'] = {}
        try:
            # İlk spotlight slide'ı bulalım
            spotlight_ul = driver.find_element(By.CSS_SELECTOR, '.spotlight-props ul')
            feature_items = spotlight_ul.find_elements(By.TAG_NAME, 'li')
            for item in feature_items:
                try:
                    label_element = item.find_element(By.TAG_NAME, 'strong')
                    value_element = item.find_element(By.TAG_NAME, 'span')
                    label = label_element.text.strip()
                    value = value_element.text.strip()
                    if label and value: # Boş etiketleri veya değerleri atla
                         listing_data['Ana Özellikler'][label] = value
                except NoSuchElementException:
                    # Bazen li içinde strong/span olmayabilir, atla
                    pass
        except NoSuchElementException:
             listing_data['Ana Özellikler'] = 'Bulunamadı veya yapı farklı'
        # print("Ana Özellikler:", listing_data.get('Ana Özellikler', 'Yok')) # Debug çıktısı


        # İlan Açıklaması çekme
        try:
            # İlan Açıklaması başlığını bul
            description_header = driver.find_element(By.XPATH, "//h2[text()='İlan Açıklaması']")
            # Açıklama div'i genellikle h2'den sonraki ilk div.content
            description_div = description_header.find_element(By.XPATH, "./following-sibling::div[@class='content'][1]")
            listing_data['İlan Açıklaması'] = description_div.text.strip() # Sadece metni al
        except NoSuchElementException:
            listing_data['İlan Açıklaması'] = 'Açıklama Bulunamadı'
        # print(f"İlan Açıklaması (İlk 100 karakter): {listing_data.get('İlan Açıklaması', 'Yok')[:100]}...") # Debug çıktısı


        # Danışman Bilgileri çekme
        listing_data['Danışman Bilgileri'] = {}
        try:
            agent_info_container = driver.find_element(By.CSS_SELECTOR, '.agent-info')

            # Danışman Adı ve Soyadı
            try:
                agent_name_element = agent_info_container.find_element(By.CSS_SELECTOR, '.user-info strong a')
                listing_data['Danışman Bilgileri']['Ad Soyad'] = agent_name_element.text.strip()
            except NoSuchElementException:
                 listing_data['Danışman Bilgileri']['Ad Soyad'] = 'Bulunamadı'

            # Ofis Adı
            try:
                office_name_element = agent_info_container.find_element(By.CSS_SELECTOR, '.user-info span a')
                listing_data['Danışman Bilgileri']['Ofis'] = office_name_element.text.strip()
            except NoSuchElementException:
                 listing_data['Danışman Bilgileri']['Ofis'] = 'Bulunamadı'


            # Telefon Numaraları
            try:
                phone_elements = agent_info_container.find_elements(By.CSS_SELECTOR, '.contact-info.active a[href^="tel:"]')
                phone_numbers = [phone.text.strip() for phone in phone_elements]
                listing_data['Danışman Bilgileri']['Telefon Numaraları'] = phone_numbers if phone_numbers else ['Bulunamadı']
            except NoSuchElementException:
                 listing_data['Danışman Bilgileri']['Telefon Numaraları'] = ['Bulunamadı']

            # E-posta (Cloudflare gizlemesi olabilir)
            try:
                email_span = agent_info_container.find_element(By.CSS_SELECTOR, '.contact-info.active span.__cf_email__')
                encoded_email = email_span.get_attribute('data-cfemail')
                if encoded_email:
                    listing_data['Danışman Bilgileri']['E-posta'] = decode_cfemail(encoded_email)
                else:
                     # Bazen doğrudan metin olabilir veya farklı bir yapı olabilir
                     # Doğrudan a[href^="mailto:"] elementini kontrol et
                     try:
                         email_link = agent_info_container.find_element(By.CSS_SELECTOR, '.contact-info.active a[href^="mailto:"]')
                         listing_data['Danışman Bilgileri']['E-posta'] = email_link.text.strip()
                     except NoSuchElementException:
                         listing_data['Danışman Bilgileri']['E-posta'] = 'Bulunamadı'
            except NoSuchElementException:
                 listing_data['Danışman Bilgileri']['E-posta'] = 'Bulunamadı'

        except NoSuchElementException:
            listing_data['Danışman Bilgileri'] = 'Danışman Bilgileri Konteyneri Bulunamadı'
        # print("Danışman Bilgileri:", listing_data.get('Danışman Bilgileri', 'Yok')) # Debug çıktısı


        # --- Ek Özellikleri Çekme (İlan Özellikleri) ---
        listing_data['Detaylı Özellikler'] = {}
        try:
            # "İlan Özellikleri" başlığını içeren section elementini bul
            properties_section = driver.find_element(By.XPATH, "//h2[text()='İlan Özellikleri']/parent::section")
            # Bu section içindeki tüm h3 başlıklarını (özellik kategorileri) bul
            feature_categories = properties_section.find_elements(By.TAG_NAME, 'h3')

            for category_h3 in feature_categories:
                category_name = category_h3.text.strip()
                if category_name: # Boş kategori isimlerini atla
                    # h3'ü takip eden ilk div.properties-container.fluid elementini bul
                    try:
                        properties_div = category_h3.find_element(By.XPATH, "./following-sibling::div[@class='properties-container fluid'][1]")
                        # Bu div içindeki tüm span.active elementlerini (özellikler) bul
                        feature_spans = properties_div.find_elements(By.CSS_SELECTOR, 'span.active')
                        features_list = [span.text.strip() for span in feature_spans if span.text.strip()] # Boş özellikleri atla
                        listing_data['Detaylı Özellikler'][category_name] = features_list
                    except NoSuchElementException:
                        listing_data['Detaylı Özellikler'][category_name] = ['Özellikler Bulunamadı'] # Kategori bulundu ama özellik divi bulunamadı
        except NoSuchElementException:
             listing_data['Detaylı Özellikler'] = 'Detaylı Özellikler Bölümü Bulunamadı'
        # print("Detaylı Özellikler:", listing_data.get('Detaylı Özellikler', 'Yok')) # Debug çıktısı


        # İlan Görüntülenme ve İletişim sayısı (Analytics)
        try:
            analytics_div = driver.find_element(By.CSS_SELECTOR, '.analytics')
            # Görüntülenme sayısını içeren strong elementini bul (metni 'Görüntülenme' olan strong'un önceki kardeşi)
            try:
                view_count_label = analytics_div.find_element(By.XPATH, ".//strong[text()='Görüntülenme']")
                view_count_element = view_count_label.find_element(By.XPATH, "./preceding-sibling::strong")
                listing_data['Görüntülenme'] = view_count_element.text.strip()
            except NoSuchElementException:
                 listing_data['Görüntülenme'] = 'Bulunamadı'

            # İletişim sayısını içeren strong elementini bul
            try:
                contact_count_label = analytics_div.find_element(By.XPATH, ".//strong[text()='İletişim']")
                contact_count_element = contact_count_label.find_element(By.XPATH, "./preceding-sibling::strong")
                listing_data['İletişim Sayısı'] = contact_count_element.text.strip()
            except NoSuchElementException:
                 listing_data['İletişim Sayısı'] = 'Bulunamadı'

        except NoSuchElementException:
            listing_data['Görüntülenme'] = 'Analitik Bilgileri Bulunamadı'
            listing_data['İletişim Sayısı'] = 'Analitik Bilgileri Bulunamadı'

        # print(f"Görüntülenme: {listing_data.get('Görüntülenme', 'Yok')}, İletişim Sayısı: {listing_data.get('İletişim Sayısı', 'Yok')}") # Debug çıktısı


    except TimeoutException:
        print(f"Hata: Sayfa yüklenirken zaman aşımı: {url}")
        listing_data['Hata'] = 'Zaman Aşımı'
    except Exception as e:
        print(f"Beklenmeyen bir hata oluştu ({url}): {e}")
        listing_data['Hata'] = f'Beklenmeyen Hata: {e}'
    finally:
        # Tarayıcıyı kapat
        if driver:
            driver.quit()
            print(f"Tarayıcı kapatıldı: {url}")
        #print(f"URL {url} için çekme tamamlandı.") # Debug çıktısı

    return listing_data

# Markdown dosyasına yazma fonksiyonu
def save_to_markdown(data_list, filename="remax_listings_report.md"):
    """
    List of listing dictionaries'ı Markdown formatında bir dosyaya kaydeder.
    """
    print(f"\nVeriler '{filename}' dosyasına yazılıyor...")

    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# RE/MAX İlan Raporu\n\n")
        f.write(f"Oluşturulma Tarihi: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("---\n\n") # Başlangıç ayırıcısı

        if not data_list:
            f.write("Hiç ilan verisi çekilemedi.\n")
            return

        for listing in data_list:
            # Hata varsa belirt
            if 'Hata' in listing:
                f.write(f"## Hata Oluşan İlan: {listing.get('URL', 'URL Yok')}\n\n")
                f.write(f"**Hata Mesajı:** {listing['Hata']}\n\n")
                f.write("---\n\n") # İlan ayırıcısı
                continue # Bu ilanı atla, sıradakine geç

            # Başlık (H2)
            f.write(f"## {listing.get('Başlık', 'Başlık Bulunamadı')}\n\n")

            # URL
            f.write(f"**URL:** [{listing.get('URL', 'URL Yok')}]({listing.get('URL', '#')})\n\n")

            # Fiyat ve Konum
            f.write(f"**Fiyat:** {listing.get('Fiyat', 'Belirtilmemiş')}\n\n")
            f.write(f"**Konum:** {listing.get('Konum', 'Belirtilmemiş')}\n\n")

            # Ana Özellikler
            if 'Ana Özellikler' in listing and isinstance(listing['Ana Özellikler'], dict):
                f.write("### Ana Özellikler\n\n")
                if listing['Ana Özellikler']:
                    for key, value in listing['Ana Özellikler'].items():
                        f.write(f"- **{key}:** {value}\n")
                else:
                    f.write("- Ana Özellikler bulunamadı.\n")
                f.write("\n") # Özellikler listesi sonrası boş satır
            elif isinstance(listing.get('Ana Özellikler'), str):
                 f.write(f"### Ana Özellikler\n\n{listing['Ana Özellikler']}\n\n")


            # İlan Açıklaması
            f.write("### İlan Açıklaması\n\n")
            description = listing.get('İlan Açıklaması', 'Açıklama Bulunamadı')
            # HTML etiketlerini veya fazla boşlukları temel düzeyde temizle (isteğe bağlı)
            # description = description.replace('<br>', '\n').replace('<p>', '\n\n').replace('</p>', '') # Örnek temizlik
            # description = ' '.join(description.split()) # Çoklu boşlukları tek boşluğa indir
            f.write(f"{description}\n\n")

            # Detaylı Özellikler
            if 'Detaylı Özellikler' in listing and isinstance(listing['Detaylı Özellikler'], dict):
                f.write("### Detaylı Özellikler\n\n")
                if listing['Detaylı Özellikler']:
                    for category, features in listing['Detaylı Özellikler'].items():
                        f.write(f"#### {category}\n\n") # Kategori Başlığı (H4)
                        if features and isinstance(features, list):
                            for feature in features:
                                f.write(f"- {feature}\n") # Özellikleri liste olarak ekle
                        elif isinstance(features, str): # Kategori bulundu ama özellik listesi yerine string (hata mesajı) geldi
                             f.write(f"- {features}\n")
                        else:
                             f.write("- Bu kategoriye ait özellik bulunamadı.\n")
                        f.write("\n") # Kategori sonrası boş satır
                else:
                    f.write("Detaylı özellikler bulunamadı.\n\n")
            elif isinstance(listing.get('Detaylı Özellikler'), str):
                 f.write(f"### Detaylı Özellikler\n\n{listing['Detaylı Özellikler']}\n\n")


            # Danışman Bilgileri
            if 'Danışman Bilgileri' in listing and isinstance(listing['Danışman Bilgileri'], dict):
                f.write("### Danışman Bilgileri\n\n")
                if listing['Danışman Bilgileri']:
                    for key, value in listing['Danışman Bilgileri'].items():
                        if isinstance(value, list):
                            f.write(f"- **{key}:**\n") # Liste başlığı
                            for item in value:
                                f.write(f"  - {item}\n") # Listenin her öğesi girintili liste öğesi
                        else:
                            f.write(f"- **{key}:** {value}\n")
                else:
                    f.write("Danışman bilgileri bulunamadı.\n")
                f.write("\n") # Danışman bilgileri sonrası boş satır
            elif isinstance(listing.get('Danışman Bilgileri'), str):
                 f.write(f"### Danışman Bilgileri\n\n{listing['Danışman Bilgileri']}\n\n")


            # İstatistikler (Görüntülenme, İletişim)
            if 'Görüntülenme' in listing or 'İletişim Sayısı' in listing:
                f.write("### İstatistikler\n\n")
                f.write(f"- **Görüntülenme:** {listing.get('Görüntülenme', 'Bulunamadı')}\n")
                f.write(f"- **İletişim Sayısı:** {listing.get('İletişim Sayısı', 'Bulunamadı')}\n")
                f.write("\n") # İstatistikler sonrası boş satır

            # İlan ayırıcısı
            f.write("---\n\n")

    print(f"Markdown dosyası başarıyla oluşturuldu: '{filename}'")

# --- Kullanım Örneği (Önceki kodun sonuna ekleyin veya güncelleyin) ---
if __name__ == "__main__":
    listing_urls = [
        "https://www.remax.com.tr/portfoy/P73663481", # Verdiğiniz örnek ilan
        # Buraya çekmek istediğiniz diğer Remax ilan URL'lerini ekleyebilirsiniz
        # "https://www.remax.com.tr/portfoy/B08780841", # Başka bir örnek
        # "https://www.remax.com.tr/portfoy/P35062245", # Başka bir örnek
         "https://www.remax.com.tr/portfoy/gecersiz-bir-url" # Hata test etmek için
    ]

    all_listings_data = []

    for url in listing_urls:
        data = scrape_remax_listing(url)
        all_listings_data.append(data)
        print("-" * 50) # İlanlar arasına ayırıcı çizgi (konsol çıktısı için)

    # Tarama bittikten sonra, toplanan veriyi Markdown dosyasına kaydet
    save_to_markdown(all_listings_data, "remax_ilan_sonuclari.md")

    # İsteğe bağlı: Konsola JSON olarak da yazdırmaya devam etmek isterseniz
    # import json
    # print("\n--- Toplanan Tüm İlan Verileri (JSON Konsol Çıktısı) ---")
    # print(json.dumps(all_listings_data, indent=4, ensure_ascii=False))