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

# Loglama ayarları (isteğe bağlı ama önerilir)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ortam değişkenlerini yükle (.env dosyasından)
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    logger.error("OpenAI API anahtarı .env dosyasında bulunamadı veya ayarlanmadı!")
    raise ValueError("OPENAI_API_KEY ortam değişkeni ayarlanmalı.")

# Sabitler
MARKDOWN_DIRECTORY = "markdowns"
PERSIST_DIRECTORY = "chroma_db" # Vektör deposunun kaydedileceği/yükleneceği dizin
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# --- Embedding Modeli ve Vektör Deposu ---

# OpenAI Embedding modeli
try:
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
except Exception as e:
    logger.error(f"OpenAI Embeddings başlatılamadı: {e}")
    raise

# ChromaDB'yi Yükle veya Oluştur (EN ÖNEMLİ DEĞİŞİKLİK)
db = None
try:
    if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
        logger.info(f"Mevcut ChromaDB veritabanı yükleniyor: {PERSIST_DIRECTORY}")
        db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
        logger.info("✅ ChromaDB başarıyla yüklendi.")
    else:
        logger.info(f"'{PERSIST_DIRECTORY}' bulunamadı veya boş. Yeni veritabanı oluşturuluyor...")
        if not os.path.exists(MARKDOWN_DIRECTORY):
             logger.error(f"Markdown dosyalarının olması beklenen '{MARKDOWN_DIRECTORY}' dizini bulunamadı!")
             raise FileNotFoundError(f"'{MARKDOWN_DIRECTORY}' dizini mevcut değil.")

        logger.info(f"'{MARKDOWN_DIRECTORY}' içindeki .md dosyaları yükleniyor...")
        loader = DirectoryLoader(MARKDOWN_DIRECTORY, glob="*.md", show_progress=True)
        documents = loader.load()

        if not documents:
            logger.warning(f"'{MARKDOWN_DIRECTORY}' içinde işlenecek .md dosyası bulunamadı.")
            # İsteğe bağlı: Boş bir veritabanı oluşturulabilir veya hata verilebilir
            # Şimdilik devam et, ancak uygulama işlevsiz olabilir.
        else:
            logger.info(f"{len(documents)} adet döküman yüklendi.")
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
            texts = text_splitter.split_documents(documents)
            logger.info(f"Dökümanlar {len(texts)} parçaya (chunk) ayrıldı.")

            logger.info("Embedding'ler oluşturuluyor ve ChromaDB'ye kaydediliyor (Bu işlem biraz sürebilir)...")
            db = Chroma.from_documents(
                documents=texts,
                embedding=embeddings,
                persist_directory=PERSIST_DIRECTORY
            )
            logger.info(f"✅ ChromaDB oluşturuldu ve '{PERSIST_DIRECTORY}' içine kaydedildi.")

except Exception as e:
    logger.error(f"ChromaDB yüklenirken/oluşturulurken hata oluştu: {e}", exc_info=True)
    # Uygulama vektör deposu olmadan çalışamaz, bu yüzden burada durmak mantıklı olabilir.
    raise

# Retriever'ı oluştur (db nesnesi artık dolu olmalı veya hata vermiş olmalı)
if db:
    retriever = db.as_retriever()
    logger.info("Retriever başarıyla oluşturuldu.")
else:
    logger.error("Veritabanı nesnesi (db) başlatılamadığı için retriever oluşturulamadı.")
    # Uygulamanın bu noktada çalışmaya devam etmemesi gerekebilir
    # veya chat endpoint'i bunu kontrol etmeli.
    retriever = None # Hata durumunu belirtmek için

# --- Prompt ve QA Zinciri ---

# Detaylı ve yapılandırılmış custom prompt
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

# Prompt Template oluştur
prompt = PromptTemplate(
    template=custom_prompt_template,
    input_variables=["context", "question"]
)

# QA Zinciri oluştur (Güncel yöntemle)
try:
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.7) # Temperature ayarını deneyebilirsiniz
    if retriever is None:
         raise ValueError("Retriever başlatılamadığı için QA zinciri oluşturulamıyor.")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff", # Token limitini aşma riskine dikkat!
        retriever=retriever,
        return_source_documents=False, # Kaynak dokümanları döndürme (isteğe bağlı)
        chain_type_kwargs={"prompt": prompt}
    )
    logger.info("✅ RetrievalQA zinciri başarıyla oluşturuldu.")
except Exception as e:
    logger.error(f"QA Zinciri oluşturulurken hata: {e}", exc_info=True)
    qa_chain = None # Hata durumunu belirt

# --- FastAPI Uygulaması ---

app = FastAPI(
    title="SibelGPT Gayrimenkul Asistanı API",
    description="LangChain ve OpenAI kullanarak emlak sorularına cevap veren API.",
    version="1.0.0"
)

# CORS Ayarları (Production için allow_origins'i daraltın)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Örnek: ["https://www.sibelgpt.com", "http://localhost:3000"]
    allow_credentials=True,
    allow_methods=["POST", "GET"], # Sadece gerekli metodlara izin verin
    allow_headers=["*"], # Veya spesifik başlıklar: ["Content-Type", "Authorization"]
)

@app.get("/")
async def read_root():
    """ API'nin çalıştığını gösteren basit bir hoşgeldin mesajı. """
    return {"message": "SibelGPT Backend API çalışıyor!"}

@app.post("/chat")
async def chat_endpoint(request: Request):
    """ Kullanıcıdan gelen soruları alır, QA zincirini çalıştırır ve cevabı döndürür. """
    if qa_chain is None or retriever is None:
         logger.error("/chat endpoint çağrıldı ancak QA zinciri veya retriever hazır değil.")
         raise HTTPException(status_code=503, detail="Servis şu anda hazır değil, lütfen daha sonra tekrar deneyin.")

    try:
        data = await request.json()
        message = data.get("question") # None dönebilir

        if not message or not isinstance(message, str) or not message.strip():
            logger.warning(f"Geçersiz veya eksik soru alındı: {message}")
            raise HTTPException(status_code=400, detail="Lütfen geçerli bir soru (question) gönderin.")

        logger.info(f"📥 Soru alındı: {message}")

        # --- DEBUG için ilgili dokümanları görme (isteğe bağlı) ---
        # try:
        #     relevant_docs = retriever.get_relevant_documents(message)
        #     logger.info(f"🔎 Bulunan ilgili döküman sayısı: {len(relevant_docs)}")
        #     for i, doc in enumerate(relevant_docs[:3], 1): # İlk 3 dokümanı logla
        #         logger.debug(f"--- İlgili Döküman {i} İçeriği (ilk 300 karakter) ---\n{doc.page_content[:300]}\n...")
        # except Exception as e_retriever:
        #     logger.warning(f"Debug için doküman alınırken hata: {e_retriever}")
        # --- DEBUG SONU ---

        logger.info("QA zinciri çalıştırılıyor...")
        # Zinciri çalıştır (Doğru invoke kullanımı)
        result = qa_chain.invoke({"query": message})
        answer = result.get("result")

        if not answer:
             logger.warning("QA zinciri 'result' anahtarı olmayan bir sonuç döndürdü.")
             answer = "Üzgünüm, sorunuzu işlerken bir sorun oluştu." # Varsayılan cevap

        logger.info(f"✅ Cevap üretildi (ilk 100 karakter): {answer[:100]}...")
        return {"reply": answer}

    except HTTPException as http_exc:
        # Zaten bilinen HTTP hatalarını tekrar yükselt
        raise http_exc
    except Exception as e:
        # Beklenmedik diğer tüm hataları yakala ve logla
        logger.error(f"'/chat' endpoint'inde beklenmedik hata: {e}", exc_info=True)
        # Kullanıcıya genel bir hata mesajı gönder
        raise HTTPException(status_code=500, detail="Mesajınız işlenirken dahili bir sunucu hatası oluştu.")

# --- Uvicorn ile Çalıştırma (opsiyonel, genellikle dışarıdan yapılır) ---
# Genellikle `uvicorn main:app --reload --host 0.0.0.0 --port 8000` gibi çalıştırılır.
# if __name__ == "__main__":
#     import uvicorn
#     logger.info("Uygulama doğrudan çalıştırılıyor (uvicorn)...")
#     uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
