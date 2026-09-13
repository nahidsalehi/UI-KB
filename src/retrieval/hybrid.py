"""
Hybrid retriever: Reciprocal Rank Fusion of dense FAISS + sparse BM25.

RRF score for each chunk:
    rrf = 1 / (K + rank_dense)  +  1 / (K + rank_bm25)

where K=60 is the standard constant that dampens the impact of high ranks.
Chunks appearing in only one list still get a partial score.
The merged pool is re-sorted by rrf score and trimmed to top_k.
"""

from .dense  import DenseRetriever
from .sparse import BM25Retriever
import config

_RRF_K = 60   # standard constant — controls rank sensitivity


class HybridRetriever:

    def __init__(self, dense: DenseRetriever, sparse: BM25Retriever):
        self.dense  = dense
        self.sparse = sparse

    # ── public API ────────────────────────────────────────────────────────────

    def search(
        self,
        query: str,
        top_k: int = config.TOP_K_RETRIEVE,
        min_score: float = config.MIN_SCORE,
    ) -> list[dict]:
        # Fetch a larger pool from each retriever so RRF has enough candidates.
        # Dense min_score is disabled here — RRF handles confidence via rank.
        pool = top_k * 2
        dense_results  = self.dense.search(query, top_k=pool, min_score=0.0)
        sparse_results = self.sparse.search(query, top_k=pool)

        rrf_scores: dict[str, float] = {}
        chunk_map:  dict[str, dict]  = {}

        for rank, r in enumerate(dense_results):
            key = r.get("id") or r["text"][:80]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
            chunk_map[key]  = r

        for rank, r in enumerate(sparse_results):
            key = r.get("id") or r["text"][:80]
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (_RRF_K + rank + 1)
            if key not in chunk_map:
                chunk_map[key] = r

        sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)

        results = []
        for key in sorted_keys[:top_k]:
            item = dict(chunk_map[key])
            item["rrf_score"] = round(rrf_scores[key], 6)
            results.append(item)
        return results

    def __repr__(self):
        return f"HybridRetriever(model={self.dense.model_name}, rrf_k={_RRF_K})"
