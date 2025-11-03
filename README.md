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
import faiss
import numpy as np
from transformers import pipeline
import os

app = Flask(__name__)

os.environ['HUGGINGFACE_HUB_TOKEN'] = 'your_huggingface_api_key_here'

# Models and Data Structures
embedder = SentenceTransformer('paraphrase-MiniLM-L6-v2')
gen = pipeline('text-generation', model='gpt2', api_key=os.environ['HUGGINGFACE_HUB_TOKEN'])

docs = [
    "The leave policy allows 20 days of paid leave per year.",
    "To reset your password, follow the instructions on the IT support portal.",
    "The company conducts annual performance reviews every December.",
]

doc_embs = embedder.encode(docs, convert_to_tensor=False)
doc_embs = np.array(doc_embs)

# FAISS Index
idx = faiss.IndexFlatL2(doc_embs.shape[1])
idx.add(doc_embs)

# Bad Language Filter
bad_w = ['badword1', 'badword2']

def filter_lang(res):
    for w in bad_w:
        res = res.replace(w, '****')
    return res

def get_docs(q, k=2):
    q_emb = embedder.encode([q], convert_to_tensor=False)
    _, indices = idx.search(np.array(q_emb), k)
    rtrvd_docs = [docs[i] for i in indices[0]]
    return rtrvd_docs

def gen_res(rtrvd_docs, q):
    ctx = " ".join(rtrvd_docs)
    prompt = f"User query: {q}\nRelevant info: {ctx}\nResponse:"
    res = gen(prompt, max_length=100, num_return_sequences=1)[0]['generated_text']
    return res

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    q = data.get('query')

    if not q:
        return jsonify({'error': 'No query provided.'}), 400

    rtrvd_docs = get_docs(q)
    res = gen_res(rtrvd_docs, q)
    res = filter_lang(res)

    return jsonify({'response': res})

@app.route('/upload', methods=['POST'])
def upload():
    f = request.files.get('file')
    if not f:
        return jsonify({'error': 'No file uploaded.'}), 400

    new_doc = f.read().decode('utf-8')
    docs.append(new_doc)

    new_emb = embedder.encode([new_doc], convert_to_tensor=False)
    new_emb = np.array(new_emb)

    idx.add(new_emb)

    return jsonify({'message': 'Document uploaded and added to the index.'})

@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
```

### Frontend HTML:
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>HR/IT Document Upload and Chatbot</title>
    <style>
        body {
            font-family: Arial, sans-serif;
            padding: 20px;
        }
        .container {
            max-width: 600px;
            margin: 0 auto;
        }
        h1 {
            text-align: center;
        }
        .form-group {
            margin-bottom: 15px;
        }
        .form-group label {
            display: block;
            margin-bottom: 5px;
        }
        .form-group input, .form-group textarea {
            width: 100%;
            padding: 10px;
            font-size: 16px;
        }
        .form-group button {
            padding: 10px 20px;
            font-size: 16px;
            cursor: pointer;
        }
    </style>
</head>
<body>
    <div class="container">
        <h1>Upload HR/IT Documents & Ask Questions</h1>

        <!-- Form to Upload Documents -->
        <div class="form-group">
            <label for="file">Upload HR Policy or IT Support Document:</label>
            <input type="file" id="file">
            <button onclick="uploadFile()">Upload</button>
        </div>

        <!-- Form to Query the Chatbot -->
        <div class="form-group">
            <label for="query">Ask a Question:</label>
            <textarea id="query" rows="4" placeholder="Ask something related to HR or IT..."></textarea>
            <button onclick="submitQuery()">Submit Query</button>
        </div>

        <!-- Display the chatbot response -->
        <div class="form-group">
            <h2>Response:</h2>
            <p id="response"></p>
        </div>
    </div>

    <script>
        function uploadFile() {
            const fileInput = document.getElementById('file');
            const file = fileInput.files[0];

            if (!file) {
                alert("Please choose a file to upload.");
                return;
            }

            const formData = new FormData();
            formData.append('file', file);

            fetch('/upload', {
                method: 'POST',
                body: formData
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert(data.error);
                } else {
                    alert(data.message);
                }
            })
            .catch(error => {
                console.error('Error:', error);
            });
        }

        function submitQuery() {
            const query = document.getElementById('query').value;

            if (!query) {
                alert("Please enter a query.");
                return;
            }

            fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ query: query })
            })
            .then(response => response.json())
            .then(data => {
                if (data.error) {
                    alert(data.error);
                } else {
                    document.getElementById('response').textContent = data.response;
                }
            })
            .catch(error => {
                console.error('Error:', error);
            });
        }
    </script>
</body>
</html>
```
