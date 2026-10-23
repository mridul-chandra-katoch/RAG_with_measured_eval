# RAG with measured eval (ragkit-lite)

A local retrieval-augmented generation (RAG) system with a built-in evaluation harness and a red-team suite. It answers questions from your documents with citations, and replies "Not found in the documents" when the evidence is weak.

Built on the Python standard library plus `pdfminer.six` for PDF parsing. No PyTorch, no FAISS, no API keys. It runs offline.

> **Status: early build.** The pipeline, evaluation commands and red-team suite run locally. There is no deployed endpoint and no published evaluation results yet. See [Status](#status).

## Features

- **Hybrid search:** TF-IDF and BM25 rankings fused with Reciprocal Rank Fusion (RRF).
- **Extractive answers:** returns sentences from the documents, with citations.
- **Abstention:** returns "Not found in the documents" when retrieval evidence is judged insufficient.
- **Agent router:** sends a question to a calculator, a SQLite database tool, or document search.
- **Web UI:** a built-in HTTP server with file upload and index rebuild.
- **Evaluation:** Hit@k, MRR and abstention accuracy.
- **Red-team suite:** prompt-injection, data-exfiltration and malformed-query tests.
- **Ingestion:** `.txt`, `.md` and `.pdf` files.

## Quickstart

### 1. Install

```bash
git clone https://github.com/mridul-chandra-katoch/RAG_with_measured_eval.git
cd RAG_with_measured_eval
pip install -r requirements.txt
```

### 2. Start the web server

```bash
python -m ragkit serve
```

Open <http://127.0.0.1:8000> in your browser.

### 3. Use the CLI

```bash
# Ask a document question (RAG)
python -m ragkit agent "What is the attendance requirement?"

# Ask a math question
python -m ragkit agent "calculate 25 * 4 + 10"

# Ask a database question
python -m ragkit agent "Who is the student with the highest grade?"

# Run the evaluation (Hit@k, MRR, abstention accuracy)
python -m ragkit eval

# Run the security tests
python -m ragkit redteam
```

## Architecture

```text
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
```

## Evaluation

`python -m ragkit eval` reports:

| Metric | What it measures |
|---|---|
| Hit@k | Whether a relevant passage appears in the top k results |
| MRR | How high the first relevant passage ranks |
| Abstention accuracy | Whether the system declines when the answer is not in the documents, and answers when it is |

Results will be published here once the question set is reviewed and split, with the set size and confidence intervals. Until then, no numbers in this repository should be read as a general claim about quality.

## Limitations

- **Lexical retrieval only.** TF-IDF and BM25 both match on shared terms, so a question worded differently from the document can be missed. There are no embeddings.
- **Extractive answers.** The system quotes sentences. It does not summarize or combine information across passages, and it uses no LLM.
- **Threshold-based abstention.** It reduces wrong answers but can still return a wrong passage, or decline a question it could have answered.
- **PDF support is basic.** Text extraction goes through `pdfminer.six`. Scanned pages and tables are not handled.

## Status

| Area | State |
|---|---|
| Hybrid lexical retrieval, citations, abstention | Implemented |
| Agent router (calculator, SQL, documents) | Implemented |
| Evaluation commands, red-team suite | Implemented, results not yet published |
| Dense (embedding) retrieval | Not started |
| LLM-based answer generation | Not started |
| Reviewed question set, document-level split, confidence intervals | Not started |
| Tests and CI | Not started |
| Docker and a deployed endpoint | Not started |
