"""
Cross-encoder reranker (BGE-Reranker-v2-m3).

Takes the candidate pool from a retriever and re-scores every
(query, passage) pair, returning the top-k by rerank score.
"""

from pathlib import Path
from sentence_transformers import CrossEncoder

import config


class Reranker:
    """
    Parameters
    ----------
    device : str
        Torch device ("cpu" or "cuda").
    model_path : str | None
        Override the model path from config.
    """

    def __init__(self, device: str = "cpu", model_path: str | None = None):
        path = model_path or config.RERANK_MODEL
        if not Path(path).exists():
            raise FileNotFoundError(
                f"Reranker model not found: {path}\n"
                "Set RERANK_MODEL_PATH or place the model at the default path."
            )
        self._model = CrossEncoder(
            path,
            device=device,
            trust_remote_code=True,
            local_files_only=True,
        )

    # ── public API ────────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        docs: list[dict],
        top_k: int = config.TOP_K_FINAL,
    ) -> list[dict]:
        if not docs:
            return []

        pairs  = [(query, d["text"]) for d in docs]
        scores = self._model.predict(pairs)

        ranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True,
        )

        results = []
        for item, score in ranked[:top_k]:
            item = dict(item)
            item["rerank_score"] = float(score)
            results.append(item)
        return results

    def __repr__(self):
        return f"Reranker(model={config.RERANK_MODEL})"
