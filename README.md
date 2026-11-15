\# ragkit-lite 🚀



A local RAG system built on the Python Standard Library. Features a hybrid retrieval engine (BM25 + TF-IDF + RRF), a learned abstention classifier, a learned reranker, and an evaluation harness.



\## ✨ Features



\- \*\*Hybrid RAG\*\*: Custom TF-IDF and BM25 search engine fused with Reciprocal Rank Fusion (RRF).

\- \*\*Learned Abstention\*\*: Uses a from-scratch logistic regression model to decide when to abstain, outperforming hardcoded thresholds on out-of-corpus detection.

\- \*\*Learned Reranker\*\*: A pairwise RankNet model that reranks top-K retrieval results.

\- \*\*Secure Agent Router\*\*: Routes questions to a Calculator (DoS-protected via AST), an SQL Database, or the Document Search engine.

\- \*\*Web UI\*\*: A built-in HTTP server with an XSS-protected browser interface, secure file uploading, and real-time index rebuilding.

\- \*\*CI Pipeline\*\*: Includes comprehensive unit tests run automatically via GitHub Actions.



\## 📊 Results



All models are trained and evaluated on SQuAD 2.0, split by article (60/15/25). Test metrics are reported with 95% confidence intervals.



| Model | Baseline | Metric | Result | CI Excludes 0? |

| :--- | :--- | :--- | :--- | :--- |

| \*\*Abstention Classifier\*\* | Threshold rule (f4 >= t) | AUROC | \*\*0.8936\*\* vs 0.8107 | Yes |

| \*\*Reranker\*\* | RRF | MRR | 0.7880 vs 0.7460 | Yes |

| \*\*Reranker\*\* | BM25-only | MRR | 0.7880 vs \*\*0.8009\*\* | \*\*No (BM25 wins)\*\* |



\*Note: The learned reranker improves upon RRF, but it does not beat a pure BM25 baseline on this dataset. We report this honestly.\*



\## 🚀 Quickstart



\### 1. Clone the repository \& install dependencies

```bash

git clone https://github.com/mridul-chandra-katoch/RAG\_with\_measured\_eval.git

cd RAG\_with\_measured\_eval

pip install -r requirements.txt

