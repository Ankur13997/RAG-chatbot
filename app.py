from flask import Flask, request, render_template, jsonify
from rag import load_index_and_docs, get_answer

app = Flask(__name__)
index, docs = load_index_and_docs()

@app.route("/")
def index_page():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    question = request.json.get("question", "")
    answer = get_answer(index, docs, question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)