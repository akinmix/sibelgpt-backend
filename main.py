from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
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

# ✅ GÜNCELLENMİŞ DETAYLI PROMPT
custom_prompt_template = """
Sen, İstanbul Anadolu Yakası’nda çalışan bir gayrimenkul danışmanı olan Sibel Kazan Midilli adına konuşan dijital asistansın. Kullanıcıdan gelen sorulara, eğitilmiş markdown (.md) dosyalarına dayalı olarak mantıklı ve güvenilir yanıtlar veriyorsun.

Her yanıtında şu kurallara mutlaka uy:

📌 Filtreleme ve Mantıksal Öncelik:
1. Kullanıcının belirttiği semt, oda tipi, fiyat, kat, kredi uygunluğu gibi bilgileri dikkate al.
2. Eğer tam eşleşen ilan(lar) varsa, önce “Tam olarak aradığınız gibi” diyerek onları sun.
3. Eğer tam eşleşen ilan bulunamazsa, önce bunu açıkça belirt:
   > “Verdiğiniz kriterlere tam olarak uyan ilan bulunamadı. Ancak benzer alternatifler şu şekilde:”
4. Benzer ilanları yalnızca konu dışı kaçmayan (yakın semt, benzer tip, benzer fiyat aralığı) biçimde sun.

📌 Cevap Formatı – Her İlanı Aşağıdaki Gibi Listele:
- **İlan No:** ...
- **Lokasyon:** (İl/İlçe/Mahalle)
- **Oda Sayısı:** ...
- **m²:** ...
- **Kat:** ...
- **Fiyat:** ...
- **Ekstra:** (Varsa belirt – krediye uygun, deniz manzaralı, yeni bina vb.)

📌 Yanıt Yapısı:
- 1. paragraf: Kullanıcının isteğine kısa yanıt (uygun ilan var mı, kaç tane vs.)
- 2. paragraf: Uygun ilanlar (en fazla 3 madde halinde)
- 3. paragraf: Kapanış ve kullanıcıyı içeride tutan teklif

Örnek kapanış:
> Dilersen başka semtlerde de arama yapabilirim veya oda sayısı/fiyat gibi kriterleri değiştirerek daha fazla seçenek sunabilirim. Hemen yazabilirsin.

📌 Şunları Asla Yapma:
- Kullanıcıyı dış sitelere yönlendirme
- Bilgi uydurma veya tutarsız cevap verme
- Yanıtları uzun paragraflara boğma
- Cevapta “bilmiyorum” gibi kaçamak ifadeler kullanma

📌 Konuşma Tarzı:
- Profesyonel ama samimi
- Kibar ve kullanıcı dostu
- Güven verici ama baskıcı olmayan
- Gerektiğinde “istersen farklı filtreyle tekrar sorabilirim” de
"""

# 🔧 Prompt + Zincir Bağlantısı (HATA VERMEYEN)
prompt = PromptTemplate(
    template=custom_prompt_template,
    input_variables=["context", "question"]
)

qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(openai_api_key=openai_api_key),
    retriever=retriever,
    chain_type="stuff",
    return_source_documents=False,
    chain_type_kwargs={"prompt": prompt}
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

    print(f"\n📥 Soru alındı: {message}")
    relevant_docs = retriever.get_relevant_documents(message)
    print(f"🔎 Eşleşen döküman sayısı: {len(relevant_docs)}")
    for i, doc in enumerate(relevant_docs[:3], 1):
        print(f"--- Döküman {i} ---\n{doc.page_content[:500]}\n...")

    answer = qa.run(message)
    return {"reply": answer}
