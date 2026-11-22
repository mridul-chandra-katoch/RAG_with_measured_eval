\# ragkit-lite 🚀



A local RAG system built on the Python Standard Library. Features a hybrid retrieval engine, a learned abstention classifier, and a rigorous evaluation harness.



\## ✨ Features



\- \*\*Hybrid RAG\*\*: Custom TF-IDF and BM25 search engines with configurable fusion (RRF or BM25-only).

\- \*\*Learned Abstention\*\*: A from-scratch logistic regression model (trained via gradient descent) to decide when to abstain.

\- \*\*Secure Agent Router\*\*: Routes questions to a Calculator (DoS-protected via AST), an SQL Database (read-only), or the Document Search engine.

\- \*\*Web UI\*\*: A built-in HTTP server with an XSS-protected browser interface, secure file uploading, and real-time index rebuilding.

\- \*\*CI Pipeline\*\*: 48 unit tests run automatically via GitHub Actions.



\## 📊 Results (Honest Baselines)



All models are trained and evaluated on SQuAD 2.0, split by article. Test metrics are reported with 95% cluster bootstrap CIs.



| Model | Baseline | Metric | Result | CI Excludes 0? |

| :--- | :--- | :--- | :--- | :--- |

| \*\*Abstention (G2)\*\* | Threshold (f3\_tfidf >= t) | AUROC | \*\*0.9003\*\* vs 0.8107 | Yes `\[0.0625, 0.1150]` |

| \*\*Reranker (XGBoost)\*\* | BM25-only | MRR | 0.7966 vs \*\*0.8009\*\* | Pending |

| \*\*Reranker (RankNet)\*\* | BM25-only | MRR | 0.7985 vs \*\*0.8009\*\* | Pending |

| \*\*Fusion (RRF)\*\* | BM25-only | MRR | 0.7482 vs \*\*0.7854\*\* | Yes `\[-0.0454, -0.0298]` |



\*Key Findings:\*

\- \*\*Fusion is disabled by default.\*\* BM25-only significantly outperforms RRF on SQuAD dev.

\- \*\*Reranker is disabled by default.\*\* Both our pure-Python RankNet and an XGBoost baseline failed to beat BM25 (features were all BM25-derived, offering no new signal).

\- \*\*Red Team:\*\* Input/output filters drop the attack success rate from \*\*66.7%\*\* (defense OFF) to \*\*0.0%\*\* (defense ON) across 6 attack classes.



\## 🚀 Quickstart



\### 1. Install

```bash

git clone https://github.com/mridul-chandra-katoch/RAG\_with\_measured\_eval.git

cd RAG\_with\_measured\_eval

pip install -e .

