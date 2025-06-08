import os
import pickle
import faiss
import fitz  # PyMuPDF
import requests
import numpy as np
import re
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline

# Load models
encoder_model = SentenceTransformer("all-MiniLM-L6-v2")
qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

# ========= TEXT EXTRACTION & CHUNKING ==========

def extract_pdf_text(pdf_path):
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text()
    return text

def crawl_angelone_support():
    base_url = "https://www.angelone.in/support"
    res = requests.get(base_url)
    soup = BeautifulSoup(res.text, "html.parser")
    links = {base_url + a['href'] for a in soup.select("a[href^='/support/']")}
    docs = []
    for link in links:
        try:
            r = requests.get(link)
            s = BeautifulSoup(r.text, "html.parser")
            content = " ".join([t.get_text(strip=True) for t in s.select("h1,h2,h3,p,li")])
            if content:
                docs.append(content)
        except Exception:
            continue
    return docs

def split_into_chunks(text, max_length=500):
    sentences = re.split(r'(?<=[.?!])\s+', text)
    chunks, current = [], ""
    for sentence in sentences:
        sentence = re.sub(r'\s+', ' ', sentence.strip())
        if len(current) + len(sentence) < max_length:
            current += " " + sentence
        else:
            chunks.append(current.strip())
            current = sentence
    if current:
        chunks.append(current.strip())
    return chunks

# ========== INGESTION SCRIPT ==========

def ingest_all():
    # Collect all PDFs in the data folder
    pdf_folder = "data"
    pdf_chunks = []

    for filename in os.listdir(pdf_folder):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            print(f"📄 Processing {filename}...")
            pdf_text = extract_pdf_text(pdf_path)
            chunks = split_into_chunks(pdf_text)
            pdf_chunks.extend(chunks)

    # Crawl AngelOne support pages
    web_texts = crawl_angelone_support()
    web_chunks = []
    for txt in web_texts:
        web_chunks.extend(split_into_chunks(txt))

    # Combine all chunks and filter
    all_chunks = pdf_chunks + web_chunks
    all_chunks = [chunk for chunk in all_chunks if len(chunk.split()) > 8]  # remove noise

    # Encode and index
    embeddings = encoder_model.encode(all_chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)

    # Save
    os.makedirs("vectorstore", exist_ok=True)
    faiss.write_index(index, "vectorstore/index.faiss")
    with open("vectorstore/docs.pkl", "wb") as f:
        pickle.dump(all_chunks, f)

    print("✅ Ingestion complete. Total clean chunks:", len(all_chunks))

# ========== INFERENCE ==========

def load_index_and_docs():
    with open("vectorstore/docs.pkl", "rb") as f:
        docs = pickle.load(f)
    index = faiss.read_index("vectorstore/index.faiss")
    return index, docs

def get_answer(index, docs, question, threshold=0.45):  # Threshold lowered
    q_emb = encoder_model.encode([question])
    D, I = index.search(q_emb, k=5)

    top_n = 3  # Use top 3 chunks for better context
    top_chunks = [docs[i] for i in I[0][:top_n]]
    top_scores = util.cos_sim(encoder_model.encode(question), encoder_model.encode(top_chunks))[0]

    best_score = float(max(top_scores))

    if best_score < threshold:
        return "I don't know"

    context = " ".join(top_chunks)

    try:
        result = qa_pipeline(question=question, context=context)
        answer = result["answer"].strip()
        return answer if answer else "I don't know"
    except Exception:
        return "I don't know"
