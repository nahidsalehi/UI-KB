"""Shared utilities: text normalization, domain detection, context building."""

import re
import config
from preprocessing import light_normalize


def normalize_text(text: str) -> str:
    return light_normalize(text)


def normalize_query(query: str) -> str:
    """Shared query cleanup used in both profiles before retriever-specific processing."""
    query = light_normalize(query)
    query = re.sub(r"\bترم\b", "نیمسال", query)
    query = re.sub(r"\bسمستر\b", "نیمسال", query)
    return query


def detect_domain(query: str) -> str:
    q = query.lower()
    d_score = sum(1 for k in config.DISCIPLINE_KEYWORDS if k in q)
    a_score = sum(1 for k in config.ACADEMIC_KEYWORDS if k in q)
    return "discipline" if d_score > a_score else "academic"


def build_context(results: list[dict]) -> str:
    blocks = []
    for r in results:
        meta_parts = []
        if r.get("article"):
            meta_parts.append(f"ماده {r['article']}")
        if r.get("type") == "definition" and r.get("term"):
            meta_parts.append(f"تعریف: {r['term']}")
        if r.get("degree_level"):
            meta_parts.append(f"مقطع: {r['degree_level']}")
        if r.get("domain"):
            label = "انضباطی" if r["domain"] == "discipline" else "آموزشی"
            meta_parts.append(label)
        if r.get("page"):
            meta_parts.append(f"صفحه {r['page']}")
        header = "  |  ".join(meta_parts)
        blocks.append(f"[{header}]\n{r['text']}")
    return "\n\n".join(blocks)
