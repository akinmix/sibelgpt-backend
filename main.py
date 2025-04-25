# -*- coding: utf-8 -*-
import os
import logging
from fastapi import FastAPI, Request, HTTPException
from routes import ilan_detay
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from image_handler import router as image_router

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
MARKDOWN_DIRECTORY = "markdowns"
PERSIST_DIRECTORY = "/var/data/chroma_db" # Render'daki Mount Path ile aynı
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
    # Render diski bu yola bağladığı için, dizinin var olması beklenir.
    if os.path.exists(PERSIST_DIRECTORY):
         # Dizin varsa ve boş değilse yükle
         # Not: Boş dizin kontrolü önemlidir, çünkü Render diski boş olarak bağlayabilir.
         if os.listdir(PERSIST_DIRECTORY):
             logger.info(f"Mevcut ChromaDB veritabanı yükleniyor: {PERSIST_DIRECTORY}")
             db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
             logger.info("✅ ChromaDB başarıyla yüklendi.")
         else:
             # Dizin var ama içi boşsa, yeni oluşturma adımına geç
             logger.info(f"'{PERSIST_DIRECTORY}' mevcut ancak boş. Yeni veritabanı oluşturulacak.")
             # db hala None, aşağıdaki blok çalışacak.
    else:
         # Bu durumun olmaması gerekir eğer disk doğru bağlandıysa, ama yine de loglayalım.
         logger.warning(f"'{PERSIST_DIRECTORY}' bulunamadı! Render disk bağlantısını kontrol edin. Yeni veritabanı oluşturulmaya çalışılacak...")
         # db hala None, aşağıdaki blok çalışacak.

    # Eğer yukarıdaki yükleme başarılı olmadıysa (db hala None ise), oluşturmayı dene
    if db is None:
         logger.info(f"Yeni veritabanı '{PERSIST_DIRECTORY}' içinde oluşturuluyor...")

         # ----> os.makedirs çağrısı burada YOK! <----
         # Render diski bu yola bağladığı için klasör zaten var olmalı.

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
             # ChromaDB DOĞRUDAN Render'ın bağladığı bu yola yazacak
             db = Chroma.from_documents(
                 documents=texts,
                 embedding=embeddings,
                 persist_directory=PERSIST_DIRECTORY # Mutlak yolu kullan
             )
             logger.info(f"✅ ChromaDB oluşturuldu ve '{PERSIST_DIRECTORY}' içine kaydedildi.")

except Exception as e:
    # Bu blok hem yükleme hem de oluşturma hatalarını yakalar
    logger.error(f"ChromaDB yüklenirken/oluşturulurken genel hata: {e}", exc_info=True)
    # Eğer hata PermissionError ise, muhtemelen disk bağlantısı/izin sorunu vardır.
    if isinstance(e, PermissionError):
         logger.error("İzin Hatası (Permission Denied)! Render disk ayarlarını ve Mount Path'i kontrol edin.")
    raise # Hatayı tekrar yükselt ki uygulama başlamasın

# Retriever oluştur
if db:
    retriever = db.as_retriever()
    logger.info("Retriever başarıyla oluşturuldu.")
else:
    logger.warning("Veritabanı nesnesi (db) başlatılamadı veya boş. Retriever None olarak ayarlandı.")
    retriever = None # Endpoint'in bunu kontrol etmesi önemli

# --- Prompt ve QA Zinciri ---

custom_prompt_template = """
Sen SibelGPT'sin. Sibel Kazan Midilli adına konuşan çok yönlü dijital bir asistansın.

Kullanıcıdan gelen sorulara, konunun içeriğine göre en uygun uzman kimliğinle yanıt verirsin.
Uzmanlık alanların şunlardır:
- İstanbul Anadolu Yakası’nda gayrimenkul danışmanlığı (özellikle Kadıköy, Suadiye, Erenköy, Maltepe, Kartal bölgeleri)
- Numeroloji ve kişisel farkındalık
- Finansal analiz, borsa ve yatırım
- Yapay zeka uygulamaları ve teknolojik trendler
- Genel kültür ve bilgilendirici yanıtlar

---

🎯 Eğer gelen soru; daire tipi, semt, fiyat, m², oda sayısı, kredi, iskan gibi gayrimenkule özgü veriler içeriyorsa:
→ Bir emlak danışmanı gibi davran ve aşağıdaki kurallara göre yanıt ver:

📌 **Emlak Filtreleme ve Mantıksal Yanıt Kuralları**
1. Kullanıcının belirttiği semt, oda tipi, fiyat, kat, kredi uygunluğu gibi bilgileri bağlam içinde ara.
2. Tam eşleşen ilan(lar) varsa: “İstediğiniz özelliklerde şu ilan(lar) mevcut.” diye sun.
3. Tam eşleşme yoksa: “Verdiğiniz kriterlere tam uyan ilan bulunamadı ama benzerler var.” diyerek yakın öneriler sun.
4. Bağlam dışına çıkma. Sadece verilerle sınırlı kal.

📄 **Cevap Formatı – Her ilan için şu şekilde yanıt ver:**
- **İlan No:** [id]
- **Lokasyon:** [ilçe / mahalle]
- **Oda Sayısı:** [örnek: 3+1]
- **m²:** [brüt metrekare]
- **Kat:** [örnek: 3. Kat, Yüksek Giriş]
- **Fiyat:** [örnek: 13.900.000 TL]
- **Ekstra:** [manzara, yeni bina, krediye uygun vb.]

✍️ **Yanıt Yapısı:**
1. Paragraf: Kullanıcıyı selamla ve kısa açıklama yap
2. Paragraf: Eşleşen veya benzer ilanları listele
3. Paragraf: Devam etmek ister misiniz? gibi soruyla konuşmayı açık bırak

---

📚 Diğer konularda (numeroloji, genel sorular, yapay zeka, borsa) gelen her soruya açık, samimi ve bilgilendirici şekilde yanıt ver.
Eğer kullanıcı ne hakkında konuşmak istediğini netleştirmediyse, konuyu nazikçe anlamaya çalış.

✨ Samimi ama bilgi dolu konuş. Gereksiz tekrar yapma. Profesyonel ama içten bir danışman gibi davran.

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

# QA Zinciri oluştur
qa_chain = None # Başlangıçta None
try:
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.7)
    if retriever is not None: # Sadece retriever varsa zinciri oluştur
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=False,
            chain_type_kwargs={"prompt": prompt}
        )
        logger.info("✅ RetrievalQA zinciri başarıyla oluşturuldu.")
    else:
        logger.warning("Retriever başlatılamadığı için QA zinciri oluşturulamadı. /chat endpoint cevap veremeyebilir.")

except Exception as e:
    logger.error(f"QA Zinciri oluşturulurken hata: {e}", exc_info=True)
    # qa_chain zaten None kalacak

# --- FastAPI Uygulaması ---

app = FastAPI(
    title="SibelGPT Gayrimenkul Asistanı API",
    description="LangChain ve OpenAI kullanarak emlak sorularına cevap veren API.",
    version="1.0.1"
)
app.include_router(ilan_detay.router)

@app.post("/ask")
async def ask_route(request: Request):
    try:
        data = await request.json()
        question = data.get("question")
        if not question:
            raise HTTPException(status_code=400, detail="Soru verilmedi.")
        response = ask(question)
        return {"reply": response}
    except Exception as e:
        logging.exception("RAG sisteminde hata oluştu")
        raise HTTPException(status_code=500, detail=str(e))

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Production için değiştirin: z.B. ["https://www.sibelgpt.com", "https://sibel-landing.vercel.app"]
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)
app.include_router(image_router)

@app.get("/")
async def read_root():
    return {"message": "SibelGPT Backend API çalışıyor!"}

@app.post("/chat")
async def chat_endpoint(request: Request):
    # Zincirin varlığını her istekte kontrol et
    if qa_chain is None:
         logger.error("/chat endpoint çağrıldı ancak QA zinciri hazır değil (başlangıçta hata oluşmuş olabilir).")
         raise HTTPException(status_code=503, detail="Servis şu anda tam olarak hazır değil, lütfen daha sonra tekrar deneyin veya yönetici ile iletişime geçin.")

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
             logger.warning("QA zinciri 'result' anahtarı olmayan bir sonuç döndürdü veya boş cevap verdi.")
             answer = "Üzgünüm, sorunuza uygun bir cevap bulamadım veya işlerken bir sorun oluştu."

        logger.info(f"✅ Cevap üretildi (ilk 100 karakter): {answer[:100]}...")
        return {"reply": answer}

    except HTTPException as http_exc:
        # FastAPI tarafından oluşturulan bilinen hataları tekrar yükselt
        raise http_exc
    except Exception as e:
        # Diğer tüm beklenmedik hataları yakala
        logger.error(f"'/chat' endpoint'inde beklenmedik hata: {e}", exc_info=True) 
        raise HTTPException(status_code=500, detail="Mesajınız işlenirken dahili bir sunucu hatası oluştu.")

# Uvicorn ile çalıştırma kısmı genellikle Render'ın Start Command'ında belirtilir.
# if __name__ == "__main__":
#     import uvicorn
#     port = int(os.getenv("PORT", 8000)) # Render genellikle PORT ortam değişkenini ayarlar
#     logger.info(f"Uygulama doğrudan çalıştırılıyor (uvicorn) - Port: {port}")
#     uvicorn.run(app, host="0.0.0.0", port=port)
