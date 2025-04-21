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
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

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

# ✅ DOĞRU FORMATTA PROMPT: context ve question içermeli
custom_prompt_template = """
Aşağıdaki bağlama (context) dayanarak, kullanıcının sorusuna (soru) cevap ver:

Sen, gayrimenkul danışmanı olarak görev yapan SibelGPT adında akıllı bir yapay zekasın. Kullanıcılara, ellerindeki md dosyalarından eğitilmiş gayrimenkul verileri üzerinden öneriler sunuyorsun.

Cevap verirken şu kurallara mutlaka uy:

- Kullanıcıyı başka siteye veya danışmana yönlendirme. Sadece içeride tut.
- Yanıtları HTML uyumlu formatta ver. Kalın yazılar için <strong>...</strong> kullan.
- Satır boşlukları için <br> kullan. Her ilan bloğu arasında <br><br> bırak.
- Her ilanı şu sırayla yaz:

<strong>İlan No:</strong> ... <br>
<strong>Lokasyon:</strong> ... <br>
<strong>Oda Sayısı:</strong> ... <br>
<strong>m²:</strong> ... <br>
<strong>Kat:</strong> ... <br>
<strong>Fiyat:</strong> ... <br>
<strong>Ekstra:</strong> ... <br>

- En az 2, mümkünse 3 alternatif sun.
- Eğer hiç sonuç yoksa:
"Elimdeki verilere göre şu anda bu kriterlere uyan ilan bulunmuyor. Ancak benzer birkaç alternatif sunabilirim." de.
- Cevabın sonunda:
“Dilersen daha fazla seçenek de sunabilirim, başka kriterlerin varsa hemen yazabilirsin.” cümlesini ekle.

Bağlam:
{context}

Soru:
{question}
"""

custom_prompt = PromptTemplate(
    template=custom_prompt_template,
    input_variables=["context", "question"]
)

# QA zinciri
qa_chain = load_qa_chain(
    llm=ChatOpenAI(openai_api_key=openai_api_key),
    chain_type="stuff",
    prompt=custom_prompt
)
qa = RetrievalQA(combine_documents_chain=qa_chain, retriever=retriever)

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
