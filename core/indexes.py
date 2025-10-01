import faiss, numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from typing import List, Dict, Tuple
import re
from .models import Chunk, DocMeta

_word_re = re.compile(r"\w+(\.\w+)?")

class HybridIndex:
    def __init__(self, model_name="BAAI/bge-small-en-v1.5"):
        self.embedder = SentenceTransformer(model_name)  # CPU ok on HF
        self.faiss_index = None
        self.vecs = None
        self.bm25 = None
        self.tokens = []
        self.chunks: List[Chunk] = []
        self.docmeta: Dict[str, DocMeta] = {}

    def build(self, metas: List[DocMeta], chunks: List[Chunk]):
        self.chunks = chunks
        self.docmeta = {m.doc_id: m for m in metas}

        # BM25
        self.tokens = [[t.lower() for t in _word_re.findall(c.text)] for c in chunks]
        self.bm25 = BM25Okapi(self.tokens)

        # Dense
        texts = [c.text for c in chunks]
        self.vecs = self.embedder.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        dim = self.vecs.shape[1]
        self.faiss_index = faiss.IndexFlatIP(dim)
        self.faiss_index.add(self.vecs)

    def lexical_topn(self, query: str, n=50) -> List[Tuple[int, float]]:
        qtok = [t.lower() for t in _word_re.findall(query)]
        scores = self.bm25.get_scores(qtok)
        idx = np.argsort(scores)[::-1][:n]
        return [(int(i), float(scores[i])) for i in idx if scores[i] > 0]

    def dense_topn(self, query: str, n=50) -> List[Tuple[int, float]]:
        qv = self.embedder.encode([query], normalize_embeddings=True, convert_to_numpy=True)
        sims, idx = self.faiss_index.search(qv, n)
        return [(int(i), float(sims[0][k])) for k, i in enumerate(idx[0])]

    def get_chunk(self, i: int) -> Chunk:
        return self.chunks[i]
