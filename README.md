# Hybrid Search POC (BM25 + bge-m3 + MiniLM rerank)

Quick start:
1) python3 -m venv .venv && source .venv/bin/activate
2) pip install -U pip -r requirements.txt
3) put PDFs in ./data/
4) streamlit run app.py

Code lives in ./core (ingest, filters, indexes, search).
