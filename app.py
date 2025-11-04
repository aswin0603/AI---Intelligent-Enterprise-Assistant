from flask import Flask, request, jsonify, render_template
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from PyPDF2 import PdfReader
import numpy as np
import faiss
import os

app = Flask(__name__)
os.environ['HUGGINGFACE_HUB_TOKEN'] = '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'

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
generator = pipeline("text2text-generation", model="gpt2", token=os.environ['HUGGINGFACE_HUB_TOKEN'])

documents = [
    "Employees are entitled to 20 days of paid leave per year.",
    "Office working hours are from 9 AM to 6 PM, Monday to Friday.",
    "Remote work is allowed with prior manager approval.",
    "The company conducts annual performance reviews every December.",
    "Grievances can be submitted via the HR portal or by contacting HR directly.",
    "IT policies prohibit installing unauthorized software.",
    "Employees must report unsafe conditions to the safety officer."
]

document_embeddings = np.array(model.encode(documents))
faiss_index = faiss.IndexFlatL2(document_embeddings.shape[1])
faiss_index.add(document_embeddings)

bad_words = ["badword1", "badword2", "offensiveword"]

def filter_bad_language(response):
    for word in bad_words:
        response = response.replace(word, "****")
    return response

def retrieve_documents(query, top_k=2):
    query_embedding = np.array(model.encode([query]))
    distances, indices = faiss_index.search(query_embedding, top_k)
    return [documents[i] for i in indices[0]]

def generate_response(retrieved_docs, user_query):
    context = " ".join(retrieved_docs)
    prompt = f"Question: {user_query}\nContext: {context}\nAnswer:"
    result = generator(prompt, max_new_tokens=100)[0]['generated_text']
    return filter_bad_language(result.strip())

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json()
    user_query = data.get("query", "").strip()
    if not user_query:
        return jsonify({"error": "No query provided."}), 400
    retrieved_docs = retrieve_documents(user_query)
    response = generate_response(retrieved_docs, user_query)
    return jsonify({"response": response})

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No file uploaded."}), 400
    if file.filename.endswith(".pdf"):
        reader = PdfReader(file)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        new_doc = text.strip()
    else:
        new_doc = file.read().decode("utf-8", errors="ignore")
    if not new_doc:
        return jsonify({"error": "File has no readable text."}), 400
    documents.append(new_doc)
    new_embedding = np.array(model.encode([new_doc]))
    faiss_index.add(new_embedding)
    return jsonify({"message": "Document uploaded and indexed successfully."})

if __name__ == "__main__":
    app.run(debug=True)
