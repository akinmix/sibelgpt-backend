from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os

from langchain_community.document_loaders import DirectoryLoader
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

loader = DirectoryLoader("markdowns", glob="*.md")
documents = loader.load()
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
texts = text_splitter.split_documents(documents)
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
db = Chroma.from_documents(texts, embeddings, persist_directory="chroma_db")
retriever = db.as_retriever()
qa = RetrievalQA.from_chain_type(llm=ChatOpenAI(openai_api_key=openai_api_key, temperature=0), retriever=retriever)

app = Flask(__name__)

@app.route("/ask", methods=["POST"])
def ask():
    data = request.json
    question = data.get("question", "")
    if not question:
        return jsonify({"error": "Soru eksik."}), 400
    answer = qa.run(question)
    return jsonify({"answer": answer})

@app.route("/", methods=["GET"])
def home():
    return "SibelGPT API çalışıyor."

if __name__ == "__main__":
    app.run(debug=True)
