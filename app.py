from flask import Flask, request, render_template, jsonify
import os
from rag import load_index_and_docs, get_answer, ingest_all

app = Flask(__name__)

# Auto-build index if it doesn't exist
VECTOR_INDEX_PATH = "vectorstore/index.faiss"
DOCS_PICKLE_PATH = "vectorstore/docs.pkl"

if not os.path.exists(VECTOR_INDEX_PATH) or not os.path.exists(DOCS_PICKLE_PATH):
    print("⚙️ Index not found. Building vector index...")
    ingest_all()

# Load index and documents
index, docs = load_index_and_docs()

@app.route("/")
def index_page():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    try:
        question = request.json.get("question", "").strip()
        if not question:
            return jsonify({"answer": "Please enter a valid question."})
        answer = get_answer(index, docs, question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"answer": f"Server error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
