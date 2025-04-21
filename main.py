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
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Loglama
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Ortam değişkenleri
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("OPENAI_API_KEY ortam değişkeni tanımlanmalı.")

# Sabitler
MARKDOWN_DIRECTORY = "markdowns"
PERSIST_DIRECTORY = "/mnt/chroma_db"  # Kalıcı disk mount path
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

# Embeddings
try:
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
except Exception as e:
    logger.error(f"Embedding başlatılamadı: {e}")
    raise

# ChromaDB yükle/oluştur
db = None
try:
    if os.path.exists(PERSIST_DIRECTORY) and os.listdir(PERSIST_DIRECTORY):
        logger.info(f"Mevcut ChromaDB yükleniyor: {PERSIST_DIRECTORY}")
        db = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=embeddings)
    else:
        if not os.path.exists(MARKDOWN_DIRECTORY):
            raise FileNotFoundError(f"'{MARKDOWN_DIRECTORY}' dizini bulunamadı.")
        loader = DirectoryLoader(MARKDOWN_DIRECTORY, glob="*.md", show_progress=True)
        documents = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
        texts = text_splitter.split_documents(documents)
        db = Chroma.from_documents(
            documents=texts,
            embedding=embeddings,
            persist_directory=PERSIST_DIRECTORY
        )
        logger.info(f"Yeni ChromaDB oluşturuldu: {PERSIST_DIRECTORY}")
except Exception as e:
    logger.error(f"ChromaDB başlatılamadı: {e}")
    raise

retriever = db.as_retriever() if db else None

# Prompt Template
custom_prompt_template = """Sen, İstanbul Anadolu Yakası’nda çalışan bir gayrimenkul danışmanı olan Sibel Kazan Midilli adına konuşan dijital asistansın. Kullanıcıdan gelen sorulara, sana sağlanan aşağıdaki bağlam (context) bilgilerine dayanarak mantıklı ve güvenilir yanıtlar veriyorsun.

Bağlam (Context):
{context}

Soru (Question):
{question}

Cevap:"""

prompt = PromptTemplate(template=custom_prompt_template, input_variables=["context", "question"])

# QA Zinciri
qa_chain = None
try:
    llm = ChatOpenAI(openai_api_key=openai_api_key, temperature=0.7)
    if retriever is None:
        raise ValueError("Retriever bulunamadı.")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=False,
        chain_type_kwargs={"prompt": prompt}
    )
except Exception as e:
    logger.error(f"QA Zinciri oluşturulamadı: {e}")

# FastAPI
app = FastAPI(
    title="SibelGPT",
    description="Sibel Kazan Midilli için AI destekli gayrimenkul danışmanı",
    version="1.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"message": "SibelGPT Backend API çalışıyor!"}

@app.post("/chat")
async def chat_endpoint(request: Request):
    if not qa_chain or not retriever:
        raise HTTPException(status_code=503, detail="QA sistemi hazır değil.")
    try:
        data = await request.json()
        message = data.get("question", "").strip()
        if not message:
            raise HTTPException(status_code=400, detail="Geçerli bir soru girilmedi.")
        result = qa_chain.invoke({"query": message})
        return {"reply": result.get("result", "Bir yanıt oluşturulamadı.")}
    except Exception as e:
        logger.error(f"Chat endpoint hatası: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Dahili sunucu hatası.")
