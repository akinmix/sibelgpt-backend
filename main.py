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

# Markdown dosyalarını yükle
loader = DirectoryLoader("markdowns", glob="*.md")
documents = loader.load()

# Metinleri parçala – chunk ayarı yükseltildi
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# Vektör veri tabanı oluştur – Chroma
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = Chroma.from_documents(texts, embeddings, persist_directory="chroma_db")

# Retriever ince ayar: daha iyi eşleşme için k ve skor eşiği
retriever = db.as_retriever(search_kwargs={"k": 5, "score_threshold": 0.5})

# LangChain QA zinciri
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=openai_api_key),
    retriever=retriever
)

# FastAPI başlat
app = FastAPI()

# CORS izinleri
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# /chat endpoint (SibelGPT frontend için)
@app.post("/chat")
async def chat_endpoint(request: Request):
    data = await request.json()
    message = data.get("question", "")

    if not message:
        return JSONResponse(content={"error": "Soru eksik."}, status_code=400)

    answer = qa.invoke(message)
    custom_closing = "\n\n👉 Eğer ilginizi çeken bir ilan varsa ilan numarasını sorarak detaylı bilgi alabilirsiniz."
    return {"reply": answer + custom_closing}
