from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os

from controllers.ilan_controller import prepare_ilan_dosyasi
from routes import ilan_detay

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# Ortam değişkenlerini yükle
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# LangChain eğitim kısmı (markdown dosyaları ile)
loader = DirectoryLoader("markdowns", glob="*.md")
documents = loader.load()

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = Chroma.from_documents(texts, embeddings, persist_directory="chroma_db")
retriever = db.as_retriever()

qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(openai_api_key=openai_api_key), retriever=retriever)

# FastAPI başlat
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# /ask endpoint (SibelGPT bilgi modülü)
@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = data.get("question")
    if not question:
        return JSONResponse(content={"error": "Soru eksik."}, status_code=400)
    answer = qa.run(question)
    return {"answer": answer}

# /api/ilan-detay endpoint (ilan botu)
app.include_router(ilan_detay.router)
