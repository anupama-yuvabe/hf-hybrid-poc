import os, re, uuid, pdfplumber
from typing import List, Tuple, Dict
from .models import Chunk, DocMeta

def _chunk_text(text: str, max_chars=800) -> List[str]:
    # simple paragraph-ish splitter
    paras = re.split(r"\n\s*\n", text)
    out, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 1 <= max_chars:
            buf += ("\n" if buf else "") + p.strip()
        else:
            if buf: out.append(buf); buf = p.strip()
        if len(p) > max_chars: out.append(p[:max_chars])
    if buf: out.append(buf)
    return [s for s in (t.strip() for t in out) if s]

def read_pdf_to_chunks(path: str, doc_type: str, title: str, extra_meta: Dict) -> Tuple[DocMeta, List[Chunk]]:
    print("starting chunking")

    doc_id = str(uuid.uuid4())
    meta = DocMeta(doc_id=doc_id, title=title, doc_type=doc_type, extra=extra_meta)
    chunks: List[Chunk] = []
    with pdfplumber.open(path) as pdf:
        for i, page in enumerate(pdf.pages, start=1):
            text = page.extract_text() or ""
            for j, piece in enumerate(_chunk_text(text)):
                chunk_id = f"{doc_id}:{i}:{j}"
                chunks.append(Chunk(
                    chunk_id=chunk_id,
                    doc_id=doc_id,
                    page_no=i,
                    section_path=[],  # keep empty for POC
                    text=piece
                ))
    print("done chunking", len(chunks), "chunks")
    return meta, chunks
