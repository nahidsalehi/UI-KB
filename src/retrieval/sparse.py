"""Unified BM25 retriever with profile-aware tokenization."""

import json
from rank_bm25 import BM25Okapi
import config
from preprocessing import sparse_tokens


class BM25Retriever:
    def __init__(self):
        self._documents: list[dict] = []
        tokenized: list[list[str]] = []
        for jsonl_path in sorted(config.JSON_DIR.glob(config.JSONL_PATTERN)):
            domain = jsonl_path.stem.replace("_v2", "")
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    rec = json.loads(line)
                    text = (rec.get("text") or "").strip()
                    if not text:
                        continue
                    article = rec.get("article")
                    index_text = f"ماده {article}: {text}" if article else text
                    self._documents.append({
                        "id": rec.get("id"), "domain": domain, "type": rec.get("type"),
                        "text": text, "article": article,
                        "degree_level": rec.get("degree_level"), "page": rec.get("page"),
                        "term": rec.get("term"), "section": rec.get("section"),
                    })
                    tokenized.append(sparse_tokens(index_text))
        if not tokenized:
            raise ValueError(f"No chunks loaded from {config.JSON_DIR}")
        self._bm25 = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = config.TOP_K_RETRIEVE) -> list[dict]:
        scores = self._bm25.get_scores(sparse_tokens(query))
        ranked = sorted(zip(self._documents, scores), key=lambda x: x[1], reverse=True)
        results = []
        for doc, score in ranked[:top_k]:
            item = dict(doc)
            item["retrieval_score"] = float(score)
            item["retrieval_type"] = "bm25"
            results.append(item)
        return results

    def __repr__(self):
        return f"BM25Retriever(profile={config.PROFILE_NAME}, docs={len(self._documents)})"
