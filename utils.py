import os
import pickle
from typing import List, Tuple
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from sentence_transformers import SentenceTransformer
import numpy as np
import requests
import tempfile
from PyPDF2 import PdfReader
import torch
from transformers import pipeline

FAISS_INDEX_PATH = "faiss_store_openai.pkl"
EMBEDDING_MODEL = SentenceTransformer('all-MiniLM-L6-v2')
QA_PIPE = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

def download_pdf(url: str) -> str:
    response = requests.get(url)
    response.raise_for_status()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
        f.write(response.content)
        return f.name

def extract_text_from_pdf(pdf_path: str) -> str:
    reader = PdfReader(pdf_path)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return text

def process_url_and_save_index(urls: List[str]):
    documents = []
    for url in urls:
        pdf_path = download_pdf(url)
        text = extract_text_from_pdf(pdf_path)
        os.remove(pdf_path)
        documents.append({"content": text, "source": url})

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_chunks = []
    metadata = []
    for doc in documents:
        chunks = splitter.split_text(doc["content"])
        all_chunks.extend(chunks)
        metadata.extend([{"source": doc["source"]}] * len(chunks))

    embeddings = EMBEDDING_MODEL.encode(all_chunks, convert_to_numpy=True)
    vectorstore = FAISS.from_embeddings(embeddings=embeddings, documents=all_chunks, metadatas=metadata)

    with open(FAISS_INDEX_PATH, "wb") as f:
        pickle.dump(vectorstore, f)

def answer_query(query: str) -> Tuple[str, List[str]]:
    with open(FAISS_INDEX_PATH, "rb") as f:
        vectorstore = pickle.load(f)

    query_embedding = EMBEDDING_MODEL.encode([query])[0]
    docs_and_scores = vectorstore.similarity_search_with_score_by_vector(query_embedding, k=4)
    top_chunks = [doc for doc, score in docs_and_scores]
    combined_context = "\n".join(top_chunks)
    answer = QA_PIPE(question=query, context=combined_context)['answer']
    sources = list(set([doc.metadata.get("source", "N/A") for doc, _ in docs_and_scores]))
    return answer, sources
