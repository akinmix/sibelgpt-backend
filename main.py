from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Ortam değişkenlerini yükle
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Eğitim verisini yükle (markdown klasöründen)
loader = DirectoryLoader("markdowns", glob="*.md")
documents = loader.load()

# Metni parçalara ayır
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# Vektör veri tabanı oluştur
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = Chroma.from_documents(texts, embeddings, persist_directory="chroma_db")
retriever = db.as_retriever()

# LangChain QA zinciri
qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(openai_api_key=openai_api_key), retriever=retriever)

# FastAPI başlat
app = FastAPI()

# CORS izinlerini ayarla
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# /ask endpoint (test için)
@app.post("/ask")
async def ask_endpoint(request: Request):
    data = await request.json()
    question = data.get("question", "")

    if not question:
        return JSONResponse(content={"error": "Soru eksik."}, status_code=400)

    answer = qa.run(question)
    return {"answer": answer}

# /chat endpoint (SibelGPT frontend için)
@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    message = data.get("message", "")

    if not message:
        return JSONResponse(content={"error": "Soru eksik."}, status_code=400)

    answer = qa.run(message)
    return {"reply": answer}

