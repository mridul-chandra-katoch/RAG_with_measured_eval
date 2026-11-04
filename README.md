# ragkit-lite 🚀

A zero-dependency, fully local RAG system built on the Python Standard Library. Features a hybrid retrieval engine, a trained abstention classifier, and a learned reranker.

## ✨ Features

- **Hybrid RAG**: Custom TF-IDF and BM25 search engine fused with Reciprocal Rank Fusion (RRF).
- **Learned Abstention**: Uses a from-scratch logistic regression model (trained via gradient descent) to decide when to abstain, outperforming hardcoded thresholds (AUROC 0.893).
- **Learned Reranker**: A pairwise RankNet model that reranks top-K retrieval results, significantly improving MRR over RRF.
- **Secure Agent Router**: Routes questions to a Calculator (DoS-protected), an SQL Database, or the Document Search engine.
- **Web UI**: A built-in HTTP server with an XSS-protected browser interface, secure file uploading, and real-time index rebuilding.
- **Evaluated on SQuAD 2.0**: Includes a rigorous evaluation harness with bootstrap confidence intervals, feature ablation, and leakage checks.

## 🚀 Quickstart

### 1. Clone the repository & install dependencies
```bash
git clone https://github.com/mridul-chandra-katoch/RAG_with_measured_eval.git
cd ragkit-lite
pip install -r requirements.txt