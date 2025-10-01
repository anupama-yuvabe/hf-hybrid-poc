import streamlit as st, os, sqlite3, json
from core.ingest import read_pdf_to_chunks
from core.indexes import HybridIndex
from core.filters import infer_filters, merge_filters, filter_mask
from core.search import hybrid_search
from core.models import DocMeta

st.set_page_config(page_title="Hybrid Search POC", layout="wide")

@st.cache_resource(show_spinner=False)
def build_corpus():
    metas = []
    chunks_all = []

    # For demo, expect files exist in ./data/
    files = [
        ("data/NYSE_SHW_2024.pdf", "annual_report", "Sherwin-Williams 2024 Annual Report", {"company": "Sherwin-Williams", "fiscal_year": "2024"}),
        ("data/builder-buyer-agreement-sample.pdf", "contract", "RERA Builder–Buyer Agreement", {"jurisdiction": "RERA"})
    ]
    for path, dt, title, extra in files:
        meta, chunks = read_pdf_to_chunks(path, dt, title, extra)
        metas.append(meta); chunks_all.extend(chunks)

    idx = HybridIndex(model_name="BAAI/bge-m3")
    idx.build(metas, chunks_all)
    return idx

index = build_corpus()

st.title("Hybrid Search POC (BM25 + bge-m3 + MiniLM rerank)")
st.caption("Answers first: snippet + provenance. Summarize only on demand.")

# Explicit filter chips
col1, col2, col3 = st.columns(3)
with col1:
    doc_type = st.selectbox("Doc Type (explicit)", ["any", "annual_report", "contract"])
with col2:
    fiscal_year = st.text_input("Fiscal Year (explicit, optional)", "")
with col3:
    jurisdiction = st.text_input("Jurisdiction (explicit, optional)", "")

q = st.text_input("Ask a question", placeholder="e.g., refund ≤ 45 days under RERA")
go = st.button("Search")

if go and q.strip():
    # Build explicit filters
    explicit = {}
    if doc_type != "any": explicit["doc_type"] = doc_type
    if fiscal_year: explicit["fiscal_year"] = fiscal_year.strip()
    if jurisdiction: explicit["jurisdiction"] = jurisdiction.strip()

    inferred = infer_filters(q)
    filters = merge_filters(explicit, inferred)
    mask = filter_mask(index, filters)

    ev = hybrid_search(index, q, mask=mask, topk_bm25=30, topk_dense=30, final_k=8)

    st.write("**Applied filters:**", ", ".join(f"{k}={v}" for k,v in filters.items()) or "none")
    for e in ev:
        meta = index.docmeta[e.doc_id]
        with st.container(border=True):
            st.markdown(f"**Answer:** {e.span_text}")
            st.caption(f"Source: {meta.title} — page {e.page_no}")
            with st.expander("View full source"):
                # show chunk text (in real app, fetch the whole page or page snippet)
                st.text(e.span_text)

            # Summarize on demand (POC: just re-show; wire LLM later)
            if st.button("Summarize", key=f"s-{e.doc_id}-{e.page_no}"):
                st.info(f"(POC) Summary: {e.span_text}")
