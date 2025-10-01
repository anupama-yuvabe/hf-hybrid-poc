from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    page_no: int
    section_path: List[str]
    text: str

@dataclass
class DocMeta:
    doc_id: str
    title: str
    doc_type: str  # "annual_report" | "contract"
    extra: Dict[str, Any]  # company, fiscal_year, jurisdiction, segment, etc.

@dataclass
class Evidence:
    doc_id: str
    page_no: int
    section_path: List[str]
    span_text: str
    score: float
