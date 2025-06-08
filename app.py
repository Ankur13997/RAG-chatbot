from flask import Flask, request, render_template, jsonify
import os
from rag import load_index_and_docs, get_answer

app = Flask(__name__)

# Only load prebuilt index, don't ingest on Render
try:
    index, docs = load_index_and_docs()
except Exception as e:
    print("❌ Failed to load FAISS index:", e)
    index, docs = None, None

@app.route("/")
def index_page():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    if not index or not docs:
        return jsonify({"answer": "Index not available. Please run ingestion locally first."})

    try:
        question = request.json.get("question", "").strip()
        if not question:
            return jsonify({"answer": "Please enter a valid question."})
        answer = get_answer(index, docs, question)
        return jsonify({"answer": answer})
    except Exception as e:
        return jsonify({"answer": f"Server error: {str(e)}"}), 500

@app.route("/ping")
def ping():
    return "pong", 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
