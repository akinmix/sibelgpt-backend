from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from dotenv import load_dotenv
import os

# Ortam değişkenlerini yükle (.env dosyasından)
load_dotenv()
api_key = os.environ.get("OPENAI_API_KEY")

# FAISS vektör veritabanını yükle
embedding = OpenAIEmbeddings(openai_api_key=api_key)
vectorstore = FAISS.load_local(
    "sibel_faiss_index", embedding, allow_dangerous_deserialization=True
)

# GPT modeli
llm = OpenAI(openai_api_key=api_key, temperature=0.4)

# QA zincirini yükle
qa_chain = load_qa_chain(llm, chain_type="stuff")

# Sistem yönlendirmesi: Türkçe ve net cevap ver
system_prompt = (
    "Lütfen aşağıdaki dökümanları kullanarak kullanıcıya TÜRKÇE, sade ve anlaşılır şekilde cevap ver. "
    "Varsayımda bulunma, sadece belgede varsa yanıtla. Uyumlu, samimi ve uzman bir danışman gibi konuş."
)

# Soruya cevap veren fonksiyon
def answer_question(question):
    docs = vectorstore.similarity_search(question, k=3)
    full_prompt = f"{system_prompt}\nSoru: {question}"
    response = qa_chain.run(input_documents=docs, question=full_prompt)
    return response

# Terminalden test etmek istersen
if __name__ == "__main__":
    print("🟣 SibelGPT Hazır. Soru sormak için yaz, çıkmak için 'çık' yaz.")
    while True:
        question = input("❓ Soru sor: ")
        if question.lower() in ["çık", "exit", "q"]:
            break
        answer = answer_question(question)
        print("🤖 SibelGPT:", answer)
