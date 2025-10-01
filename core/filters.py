import re
from typing import Dict, Any

def infer_filters(query: str) -> Dict[str, Any]:
    q = query.lower()
    out = {}

    # doc_type
    if "contract" in q or "agreement" in q:
        out["doc_type"] = "contract"
    if "annual report" in q or "report" in q:
        out["doc_type"] = "annual_report"

    # fiscal_year
    m = re.search(r"\b(20[12][0-9])\b", q)
    if m:
        out["fiscal_year"] = m.group(1)

    # jurisdiction (toy)
    if "rera" in q:
        out["jurisdiction"] = "RERA"

    # segment (toy)
    if "consumer brands group" in q:
        out["segment"] = "Consumer Brands Group"
    if "paint stores group" in q:
        out["segment"] = "Paint Stores Group"

    # refund_days (toy)
    m2 = re.search(r"refund.*?(\d+)\s*day", q)
    if m2:
        out["refund_days_lte"] = int(m2.group(1))

    return out

def merge_filters(explicit: Dict[str, Any], implicit: Dict[str, Any]) -> Dict[str, Any]:
    # explicit wins
    merged = dict(implicit)
    merged.update(explicit or {})
    return merged

def filter_mask(index, filters: Dict[str, Any]):
    # produce a boolean mask over chunks based on docmeta
    mask = []
    for c in index.chunks:
        dm = index.docmeta[c.doc_id]
        ok = True
        for k, v in (filters or {}).items():
            if k == "refund_days_lte":
                # needs extracted field; for POC skip hard check -> leave to text match
                continue
            elif k in dm.extra and str(dm.extra[k]).lower() != str(v).lower():
                ok = False; break
            elif k == "doc_type" and dm.doc_type != v:
                ok = False; break
        mask.append(ok)
    return mask
