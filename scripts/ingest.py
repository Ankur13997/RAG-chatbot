import os
import pickle
import faiss
import re
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from rag import extract_pdf_text, crawl_angelone_support, build_faiss_index
from sentence_transformers import SentenceTransformer

def split_into_chunks(text, max_length=500):
    sentences = re.split(r'(?<=[.?!])\s+', text)
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) < max_length:
            current += " " + sentence
        else:
            chunks.append(current.strip())
            current = sentence
    if current:
        chunks.append(current.strip())
    return chunks

# Load and chunk PDF
pdf_path = "data/America's_Choice_2500_Gold_SOB (1) (1).pdf"
pdf_text = extract_pdf_text(pdf_path)
pdf_chunks = split_into_chunks(pdf_text)

# Crawl and chunk Angel One support pages
web_texts = crawl_angelone_support()
web_chunks = []
for txt in web_texts:
    web_chunks.extend(split_into_chunks(txt))

# Combine all chunks
all_chunks = pdf_chunks + web_chunks

# Encode and store in FAISS
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(all_chunks)
index = faiss.IndexFlatL2(embeddings.shape[1])
index.add(embeddings)

# Save
os.makedirs("vectorstore", exist_ok=True)
faiss.write_index(index, "vectorstore/index.faiss")
with open("vectorstore/docs.pkl", "wb") as f:
    pickle.dump(all_chunks, f)

print("✅ Ingestion complete. Total chunks:", len(all_chunks))
