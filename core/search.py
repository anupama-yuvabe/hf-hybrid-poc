from typing import Dict, Any, List, Tuple
import numpy as np
from sentence_transformers import CrossEncoder
from .models import Evidence

cross = None  # lazy init

def ensure_reranker():
    global cross
    if cross is None:
        cross = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")  # CPU ok

def hybrid_search(index, query: str, mask: List[bool], topk_bm25=30, topk_dense=30, final_k=10) -> List[Evidence]:
    ensure_reranker()

    # candidate ids after mask
    allowed = np.array([i for i, ok in enumerate(mask) if ok])

    # BM25
    bm25 = index.lexical_topn(query, n=topk_bm25 * 3)
    bm25_ids = [i for i, s in bm25 if i in allowed][:topk_bm25]

    # Dense
    dense = index.dense_topn(query, n=topk_dense * 3)
    dense_ids = [i for i, s in dense if i in allowed][:topk_dense]

    # Union + dedupe
    cand = list(dict.fromkeys(bm25_ids + dense_ids))

    # Rerank
    pairs = [(query, index.get_chunk(i).text) for i in cand]
    scores = cross.predict(pairs).tolist()
    order = np.argsort(scores)[::-1][:final_k]

    out = []
    for pos in order:
        i = cand[int(pos)]
        ch = index.get_chunk(i)
        # crude span = first sentence with the best keyword
        span = ch.text.split(". ")
        span_text = span[0][:400].strip()
        out.append(Evidence(
            doc_id=ch.doc_id,
            page_no=ch.page_no,
            section_path=ch.section_path,
            span_text=span_text,
            score=float(scores[int(pos)]),
        ))
    return out
