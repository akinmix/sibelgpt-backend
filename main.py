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

# Metinleri parçala
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=50)
texts = text_splitter.split_documents(documents)

# Yüklenen chunk sayısını terminalde göster
print(f"Yüklenen doküman sayısı: {len(texts)}")

# Vektör veritabanı oluştur
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = Chroma.from_documents(texts, embeddings, persist_directory="chroma_db")

# Retriever ayarı
retriever = db.as_retriever(search_kwargs={"k": 5})

# QA zinciri
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=openai_api_key),
    retriever=retriever
)

# FastAPI başlat
app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# /chat endpoint
@app.post("/chat")
async def chat_endpoint(request: Request):
    try:
        data = await request.json()
        message = data.get("question", "")

        if not message:
            return JSONResponse(content={"error": "Soru eksik."}, status_code=400)

        result = qa.invoke(message)

        if not result or not isinstance(result, str):
            result = "Üzgünüm, bu konuda size yardımcı olabilecek bir bilgiye ulaşamadım."

        custom_closing = "\n\n👉 Eğer ilginizi çeken bir ilan varsa ilan numarasını sorarak detaylı bilgi alabilirsiniz."
        return {"reply": result + custom_closing}

    except Exception as e:
        return JSONResponse(content={"error": f"Sunucu hatası: {str(e)}"}, status_code=500)
