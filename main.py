import os
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv

# LangChain ve ilgili kütüphaneler
from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma # Chroma'yı community'den almak daha güncel olabilir
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- Yapılandırma ve Başlangıç ---

# Loglama ayarları
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ortam değişkenlerini yükle
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("OpenAI API anahtarı .env dosyasında bulunamadı veya ayarlanmadı!")
    raise ValueError("OPENAI_API_KEY ortam değişkeni ayarlanmalı.")

# Sabitler
MARKDOWN_DIRECTORY = "markdowns" # Bu hala göreceli olabilir, main.py'nin yanındaki klasör
# ----- DEĞİŞİKLİK BURADA -----
PERSIST_DIRECTORY = "/var/data/chroma_db" # Render'daki Mount Path ile aynı olmalı!
# -----------------------------
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- Embedding Modeli ve Vektör Deposu ---

try:
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
except Exception as e:
    logger.error(f"OpenAI Embeddings başlatılamadı: {e}")
    raise

db = None
try:
    # Veritabanı YOLUNU kontrol et (Mutlak yol)
    if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
        logger.info(f"Mevcut ChromaDB veritabanı yükleniyor: {PERSIST_DIRECTORY}")
        db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
        logger.info("✅ ChromaDB başarıyla yüklendi.")
    else:
        # Eğer dizin yoksa, Render'daki disk bağlantısı henüz aktif olmamış olabilir
        # veya ilk deploy. Dizini oluşturmayı deneyelim.
        if not os.path.exists(PERSIST_DIRECTORY):
             logger.warning(f"'{PERSIST_DIRECTORY}' dizini bulunamadı. Oluşturuluyor...")
             try:
                 os.makedirs(PERSIST_DIRECTORY) # Dizini oluştur
                 logger.info(f"'{PERSIST_DIRECTORY}' dizini başarıyla oluşturuldu.")
             except OSError as e:
                 logger.error(f"'{PERSIST_DIRECTORY}' dizini oluşturulamadı: {e}. İzinleri kontrol edin.")
                 raise # Dizini oluşturamazsa devam edemez

        # Dizini oluşturduktan veya zaten var olduktan sonra tekrar kontrol et
        if not os.path.exists(PERSIST_DIRECTORY) or not os.listdir(PERSIST_DIRECTORY):
             logger.info(f"'{PERSIST_DIRECTORY}' boş veya yeni oluşturuldu. Yeni veritabanı oluşturuluyor...")
             if not os.path.exists(MARKDOWN_DIRECTORY):
                 logger.error(f"Markdown dosyalarının olması beklenen '{MARKDOWN_DIRECTORY}' dizini bulunamadı!")
                 raise FileNotFoundError(f"'{MARKDOWN_DIRECTORY}' dizini mevcut değil.")

             logger.info(f"'{MARKDOWN_DIRECTORY}' içindeki .md dosyaları yükleniyor...")
             loader = DirectoryLoader(MARKDOWN_DIRECTORY, glob="*.md", show_progress=True)
             documents = loader.load()

             if not documents:
                 logger.warning(f"'{MARKDOWN_DIRECTORY}' içinde işlenecek .md dosyası bulunamadı.")
             else:
                 logger.info(f"{len(documents)} adet döküman yüklendi.")
                 text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
                 texts = text_splitter.split_documents(documents)
                 logger.info(f"Dökümanlar {len(texts)} parçaya (chunk) ayrıldı.")

                 logger.info("Embedding'ler oluşturuluyor ve ChromaDB'ye kaydediliyor...")
                 # Kaydederken MUTLAK YOLU kullan
                 db = Chroma.from_documents(
                     documents=texts,
                     embedding=embeddings,
                     persist_directory=PERSIST_DIRECTORY # Mutlak yolu kullan
                 )
                 logger.info(f"✅ ChromaDB oluşturuldu ve '{PERSIST_DIRECTORY}' içine kaydedildi.")

except Exception as e:
    logger.error(f"ChromaDB yüklenirken/oluşturulurken hata oluştu: {e}", exc_info=True)
    raise

if db:
    retriever = db.as_retriever()
    logger.info("Retriever başarıyla oluşturuldu.")
else:
    logger.error("Veritabanı nesnesi (db) başlatılamadığı için retriever oluşturulamadı.")
    retriever = None

# --- Prompt ve QA Zinciri ---

custom_prompt_template = """
Sen, İstanbul Anadolu Yakası’nda çalışan bir gayrimenkul danışmanı olan Sibel Kazan Midilli adına konuşan dijital asistansın. Kullanıcıdan gelen sorulara, sana sağlanan aşağıdaki bağlam (context) bilgilerine dayanarak mantıklı ve güvenilir yanıtlar veriyorsun.

Her yanıtında şu kurallara mutlaka uy:

📌 Filtreleme ve Mantıksal Öncelik:
1. Kullanıcının belirttiği semt, oda tipi, fiyat, kat, kredi uygunluğu gibi bilgileri bağlam içinde ara ve dikkate al.
2. Eğer bağlamda tam eşleşen ilan(lar) varsa, önce “Tam olarak aradığınız kriterlere uygun şu ilanları buldum:” diyerek onları sun.
3. Eğer bağlamda tam eşleşen ilan bulunamazsa, bunu açıkça belirt:
   > “Verdiğiniz kriterlere tam olarak uyan ilan şu anda elimdeki verilerde bulunamadı. Ancak benzer olabilecek birkaç alternatif şunlar olabilir:”
4. Benzer ilanları, yalnızca bağlamdaki bilgilerden yola çıkarak, konu dışına çıkmadan (örneğin yakın semt, benzer oda sayısı, yakın fiyat aralığı gibi mantıklı yakınlıkta) sun. Asla bağlam dışı bilgi uydurma.

📌 Cevap Formatı – Her İlanı Aşağıdaki Gibi Listele (Bilgi yoksa boş bırakma, "Belirtilmemiş" yaz):
- **İlan No:** (Bağlamdaki ilan no)
- **Lokasyon:** (Bağlamdaki İl/İlçe/Mahalle)
- **Oda Sayısı:** (Bağlamdaki oda sayısı)
- **m²:** (Bağlamdaki metrekare)
- **Kat:** (Bağlamdaki kat bilgisi)
- **Fiyat:** (Bağlamdaki fiyat)
- **Ekstra:** (Bağlamdaki ek bilgiler - krediye uygun, deniz manzaralı, yeni bina vb. Varsa belirt, yoksa bu satırı ekleme)

📌 Yanıt Yapısı:
- 1. Paragraf: Kullanıcının isteğine kısa bir yanıt (örneğin: "İstediğiniz özelliklerde 3 ilan buldum." veya "Tam eşleşen ilan bulamadım ama benzerleri var.")
- 2. Paragraf: Bulunan uygun ilanlar (eğer varsa, yukarıdaki formatta, en fazla 3 tane, her biri arasında bir boş satır bırakarak).
- 3. Paragraf: Kibar bir kapanış ve kullanıcıyı daha fazla soru sormaya teşvik eden bir cümle.

Örnek Kapanış:
> Dilersen farklı bir semt veya bütçe için de arama yapabiliriz ya da mevcut ilanlar hakkında daha fazla detay sorabilirsin. Nasıl devam etmek istersin?

📌 Şunları Asla Yapma:
- Kullanıcıyı dışarıdaki web sitelerine, başka danışmanlara veya Remax'a yönlendirme.
- Bağlamda olmayan bilgileri uydurma veya tutarsız cevap verme.
- Yanıtları gereksiz uzun paragraflara boğma.
- "Bilmiyorum", "Emin değilim" gibi kaçamak veya yetersiz cevaplar verme. Sadece sağlanan bağlamı kullan.

📌 Konuşma Tarzı:
- Profesyonel, yardımsever ve samimi.
- Güven verici ve bilgili.
- Kibar ve kullanıcı dostu.

Bağlam (Context):
{context}

Soru (Question):
{question}

Cevap:
"""

prompt = PromptTemplate(
    template=custom_prompt_template,
    input_variables=["context", "question"]
)

try:
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.7)
    if retriever is None:
         raise ValueError("Retriever başlatılamadığı için QA zinciri oluşturulamıyor.")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )
    logger.info("✅ RetrievalQA zinciri başarıyla oluşturuldu.")
except Exception as e:
    logger.error(f"QA Zinciri oluşturulurken hata: {e}", exc_info=True)
    qa_chain = None

# --- FastAPI Uygulaması ---

app = FastAPI(
    title="SibelGPT Gayrimenkul Asistanı API",
    description="LangChain ve OpenAI kullanarak emlak sorularına cevap veren API.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Production için değiştirin: ["https://www.sibelgpt.com"]
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"message": "SibelGPT Backend API çalışıyor!"}

@app.post("/chat")
async def chat_endpoint(request: Request):
    if qa_chain is None or retriever is None:
         logger.error("/chat endpoint çağrıldı ancak QA zinciri veya retriever hazır değil.")
         raise HTTPException(status_code=503, detail="Servis şu anda hazır değil, lütfen daha sonra tekrar deneyin.")

    try:
        data = await request.json()
        message = data.get("question")

        if not message or not isinstance(message, str) or not message.strip():
            logger.warning(f"Geçersiz veya eksik soru alındı: {message}")
            raise HTTPException(status_code=400, detail="Lütfen geçerli bir soru (question) gönderin.")

        logger.info(f"📥 Soru alındı: {message}")

        logger.info("QA zinciri çalıştırılıyor...")
        result = qa_chain.invoke({"query": message})
        answer = result.get("result")

        if not answer:
             logger.warning("QA zinciri 'result' anahtarı olmayan bir sonuç döndürdü.")
             answer = "Üzgünüm, sorunuzu işlerken bir sorun oluştu."

        logger.info(f"✅ Cevap üretildi (ilk 100 karakter): {answer[:100]}...")
        return {"reply": answer}

    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"'/chat' endpoint'inde beklenmedik hata: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Mesajınız işlenirken dahili bir sunucu hatası oluştu.")
