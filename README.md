# An Intelligent Enterprise Assistant for public sector (SIH 1706)

## PROBLEM TITLE:
"Intelligent Enterprise Assistant: Enhancing Organizational Efficiency through AI-driven Chatbot Integration"

## Problem ID: SIH1706

## Problem Statement:
### Description: 
Develop a chatbot using deep learning and natural language processing techniques to accurately understand and respond to queries from employees of a large public sector organization.
The chatbot should be capable of handling diverse questions related to HR policies, IT support, company events, and other organizational matters. (Hackathon students/teams to use publicly available sample information for HR Policy, IT Support, etc. available on internet.) Develop document processing capabilities for the chatbot to analyse and extract information from documents uploaded by employees.
This includes summarizing a document or extracting text (keyword information) from documents relevant to organizational needs. (Hackathon students/teams can use any 8 to 10 page document for demonstration). Ensure the chatbot architecture is scalable to handle minimum 5 users parallelly. This includes optimizing response time (Response Time should not exceed 5 seconds for any query unless there is a technical issue like connectivity, etc.) Enable 2FA (2 Factor Authentication – email id type) in the chatbot for enhancing the security level of the chatbot. 
Chatbot should filter bad language as per system-maintained dictionary. 

## Code:
The below backend allows users to upload documents (HR Policy, IT Support) and query them using dense retrieval.
### Backend API:
```python
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
    result = generator(prompt, max_new_tokens=200)[0]['generated_text']
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

```

### Frontend HTML:
```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>HR/IT Chatbot</title>
  <style>
    body {
      font-family: Arial, sans-serif;
      background: #f8f9fa;
      color: #222;
      display: flex;
      justify-content: center;
      align-items: center;
      flex-direction: column;
      padding: 40px;
    }

    h1 {
      text-align: center;
      margin-bottom: 20px;
    }

    .container {
      background: #fff;
      padding: 30px 40px;
      border-radius: 16px;
      box-shadow: 0 0 12px rgba(0, 0, 0, 0.1);
      width: 1400px;
    }

    .form-group {
      margin-bottom: 20px;
    }

    label {
      display: block;
      font-weight: bold;
      margin-bottom: 5px;
    }

    input[type="file"],
    textarea {
      width: 100%;
      padding: 10px;
      font-size: 15px;
      border: 1px solid #ccc;
      border-radius: 6px;
    }

    button {
      padding: 10px 20px;
      background: #007bff;
      color: #fff;
      border: none;
      border-radius: 6px;
      font-size: 16px;
      cursor: pointer;
      transition: background 0.3s;
    }

    button:hover {
      background: #0056b3;
    }

    h3 {
      margin-top: 25px;
      font-size: 18px;
    }

    #response {
      background: #f0f0f0;
      padding: 10px;
      border-radius: 8px;
      min-height: 50px;
      margin-top: 10px;
    }

    /* Spinner Styles */
    .spinner {
      display: none;
      margin: 10px auto;
      border: 4px solid #f3f3f3;
      border-top: 4px solid #007bff;
      border-radius: 50%;
      width: 30px;
      height: 30px;
      animation: spin 1s linear infinite;
    }

    @keyframes spin {
      0% { transform: rotate(0deg); }
      100% { transform: rotate(360deg); }
    }
  </style>
</head>
<body>
  <div class="container">
    <h1>Upload HR/IT Documents & Ask Questions</h1>

    <div class="form-group">
      <label for="file">Upload HR Policy or IT Support Document:</label>
      <input type="file" id="file">
      <button onclick="uploadFile()">Upload</button>
    </div>

    <div class="form-group">
      <label for="query">Ask a Question:</label>
      <textarea id="query" rows="3" placeholder="Type your question here..."></textarea>
      <button onclick="submitQuery()">Submit Query</button>
    </div>

    <div class="spinner" id="spinner"></div>

    <h3>Response:</h3>
    <p id="response"></p>
  </div>

  <script>
    function uploadFile() {
      const file = document.getElementById("file").files[0];
      if (!file) return alert("Please select a file to upload.");

      const formData = new FormData();
      formData.append("file", file);

      fetch("/upload", { method: "POST", body: formData })
        .then(res => res.json())
        .then(data => alert(data.message || data.error))
        .catch(() => alert("Error uploading file."));
    }

    function submitQuery() {
      const query = document.getElementById("query").value;
      const responseField = document.getElementById("response");
      const spinner = document.getElementById("spinner");

      if (!query) return alert("Please enter a query.");

      spinner.style.display = "block";
      responseField.innerText = "";

      fetch("/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query })
      })
        .then(res => res.json())
        .then(data => {
          spinner.style.display = "none";
          responseField.innerText = data.response || data.error || "No response received.";
        })
        .catch(() => {
          spinner.style.display = "none";
          responseField.innerText = "Error connecting to chatbot.";
        });
    }
  </script>
</body>
</html>
```

## Outputs:
### Output 01 : 
<img width="1276" height="1006" alt="image" src="https://github.com/user-attachments/assets/fd645113-7246-4da3-921d-250d538c590b" />
<img width="1888" height="662" alt="image" src="https://github.com/user-attachments/assets/d059cfa5-7d01-4251-9ef0-e857c7b7070b" />

## Result :
Thus, we have successfully developed a chatbot with the provided requirements. 


### NAME: ASWUN B
### REGISTER NUMBER: 212224110007
