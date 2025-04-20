from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

# Ortam değişkenlerini yükle
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Markdown dosyalarını yükle
loader = DirectoryLoader("markdowns", glob="*.md")
documents = loader.load()

# Chunk ayarları
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)
print(f"\n✅ Toplam yüklü döküman: {len(texts)}")

# Vektör veritabanı
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = Chroma.from_documents(texts, embeddings, persist_directory="chroma_db")
retriever = db.as_retriever()

# LangChain QA zinciri
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=openai_api_key),
    retriever=retriever
)

# FastAPI başlat
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    message = data.get("question", "")

    if not message:
        return JSONResponse(content={"error": "Soru eksik."}, status_code=400)

    # --- DEBUG ---
    print(f"\n📥 Soru alındı: {message}")
    relevant_docs = retriever.get_relevant_documents(message)
    print(f"🔎 Eşleşen doküman sayısı: {len(relevant_docs)}")
    for i, doc in enumerate(relevant_docs[:3], 1):
        print(f"--- Döküman {i} ---\n{doc.page_content[:500]}\n...")
    # --- DEBUG SONU ---

    answer = qa.run(message)
    return {"reply": answer}

# İlan detay endpoint'i aktifse buraya route eklenebilir
