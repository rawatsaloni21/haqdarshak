import os
import pickle
from typing import List, Tuple
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from PyPDF2 import PdfReader
import tempfile
import requests

CONFIG_PATH = ".config"
FAISS_INDEX_PATH = "faiss_store_openai.pkl"

# Load API key
def get_openai_api_key():
    with open(CONFIG_PATH, "r") as file:
        for line in file:
            if line.startswith("OPENAI_API_KEY"):
                return line.strip().split("=")[1]
    raise ValueError("OPENAI_API_KEY not found in .config")

def download_pdf(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(response.content)
        return f.name

def process_url_and_save_index(urls: List[str]):
    all_docs = []
    for url in urls:
        pdf_path = download_pdf(url)
        loader = UnstructuredPDFLoader(pdf_path)
        docs = loader.load()
        all_docs.extend(docs)
        os.remove(pdf_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(all_docs)

    embeddings = OpenAIEmbeddings(openai_api_key=get_openai_api_key())
    vectorstore = FAISS.from_documents(chunks, embeddings)

    with open(FAISS_INDEX_PATH, "wb") as f:
        pickle.dump(vectorstore, f)

def answer_query(query: str) -> Tuple[str, List[str]]:
    with open(FAISS_INDEX_PATH, "rb") as f:
        vectorstore = pickle.load(f)

    docs = vectorstore.similarity_search(query)
    llm = OpenAI(openai_api_key=get_openai_api_key())
    chain = load_qa_chain(llm, chain_type="stuff")
    answer = chain.run(input_documents=docs, question=query)
    sources = list(set([doc.metadata.get("source", "N/A") for doc in docs]))
    return answer, sources
