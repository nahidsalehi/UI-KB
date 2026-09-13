"""Dense FAISS retriever over the unified UI-KB corpus."""

import json
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import config
from preprocessing import dense_text


class DenseRetriever:
    def __init__(self, model_name: str, device: str = "cpu"):
        if model_name not in config.MODELS:
            raise ValueError(f"Unknown model '{model_name}'. Choose from {list(config.MODELS)}")
        model_cfg = config.MODELS[model_name]
        self.prefix = model_cfg["prefix"]
        self.label = model_cfg["label"]
        self.model_name = model_name

        faiss_path = config.INDICES_DIR / model_name / "all.faiss"
        meta_path = config.INDICES_DIR / model_name / "all_meta.json"
        if not faiss_path.exists():
            raise FileNotFoundError(
                f"FAISS index not found: {faiss_path}\n"
                f"Build it with: python src/embedder.py --profile {config.PROFILE_NAME} --model {model_name}"
            )
        if not meta_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        model_path = Path(model_cfg["path"])
        if not model_path.exists():
            raise FileNotFoundError(f"Embedding model not found: {model_path}")

        self._embed = SentenceTransformer(str(model_path), device=device, local_files_only=True)
        self._index = faiss.read_index(str(faiss_path))
        self._meta = json.loads(meta_path.read_text(encoding="utf-8"))

        expected_dim = model_cfg.get("dim")
        if expected_dim and self._index.d != expected_dim:
            raise ValueError(
                f"Index dimension mismatch for {model_name}: expected {expected_dim}, "
                f"got {self._index.d}. Rebuild the index for profile={config.PROFILE_NAME}."
            )
        if self._index.ntotal != len(self._meta):
            raise ValueError(
                f"Index/metadata count mismatch for {model_name}: "
                f"{self._index.ntotal} vectors vs {len(self._meta)} metadata rows."
            )

    def search(self, query: str, top_k: int = config.TOP_K_RETRIEVE,
               min_score: float = config.MIN_SCORE) -> list[dict]:
        encoded_query = self.prefix + dense_text(query)
        vec = self._embed.encode(
            [encoded_query], normalize_embeddings=True, convert_to_numpy=True
        ).astype("float32")
        scores, ids = self._index.search(vec, top_k)
        results = []
        for score, idx in zip(scores[0], ids[0]):
            if idx == -1 or score < min_score:
                continue
            item = dict(self._meta[idx])
            item["retrieval_score"] = float(score)
            item["retrieval_type"] = "dense"
            results.append(item)
        return results

    def __repr__(self):
        return (
            f"DenseRetriever(model={self.model_name}, profile={config.PROFILE_NAME}, "
            f"vectors={self._index.ntotal})"
        )
