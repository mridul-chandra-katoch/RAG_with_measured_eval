# ragkit-lite 🚀

A zero-dependency, fully local, multi-tool AI Agent and RAG (Retrieval-Augmented Generation) system. 

Built entirely on the **Python Standard Library** (plus `pdfminer.six` for PDF parsing). No PyTorch, no FAISS, no OpenAI API keys required. It runs completely offline and is compatible with any Python version (including Python 3.12+).

## ✨ Features

- **Hybrid RAG**: Custom TF-IDF and BM25 search engine fused with Reciprocal Rank Fusion (RRF).
- **Extractive Answers**: Pulls exact sentences from documents and provides citations.
- **Strict Abstention**: Safely says "Not found in the documents" when the context does not contain the answer (eliminates hallucinations).
- **Agent Router**: Automatically routes questions to a **Calculator**, an **SQL Database**, or the **Document Search** engine based on the prompt.
- **Web UI**: A built-in HTTP server with a clean browser interface, including file uploading and real-time index rebuilding.
- **Red-Team Suite**: Tests itself against prompt injection, data exfiltration, and malformed queries.
- **PDF Ingestion**: Upload `.txt`, `.md`, and `.pdf` files directly through the web interface.

## 🚀 Quickstart

### 1. Clone the repository & install dependencies
```bash
git clone https://github.com/yourusername/ragkit-lite.git
cd ragkit-lite
pip install -r requirements.txt


2. Start the Web Server
bash

python -m ragkit serve

Open http://127.0.0.1:8000 in your browser.
3. Use the CLI
bash

# Ask a Document Question (RAG)
python -m ragkit agent "What is the attendance requirement?"

# Ask a Math Question
python -m ragkit agent "calculate 25 * 4 + 10"

# Ask a Database Question
python -m ragkit agent "Who is the student with the highest grade?"

# Run Evaluations (Hit@k, MRR, Abstention Accuracy)
python -m ragkit eval

# Run Security Tests
python -m ragkit redteam

🏗️ Architecture
text

User Question
      │
      ▼
[ Agent Router ] ──(Math)──► [ Calculator Tool ]
      │
      ├────────────(SQL)──► [ SQLite Database Tool ]
      │
      ▼ (Everything Else)
[ Hybrid Search ]
      ├──► [ TF-IDF Index ]
      └──► [ BM25 Index ]
             │
             ▼ (Reciprocal Rank Fusion)
[ Abstention Filter ]
      │
      ▼
[ Extractive Generator ] ──► Answer + Citations

