"""
Retrieval Evaluation Metrics
=============================
Recall@K, Precision@K, MRR, nDCG@K
"""

import math
from typing import List


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    hits = len(set(retrieved[:k]) & set(relevant))
    return hits / len(relevant)


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if k == 0:
        return 0.0
    hits = len(set(retrieved[:k]) & set(relevant))
    return hits / k


def mrr(retrieved: List[str], relevant: List[str]) -> float:
    relevant_set = set(relevant)
    for rank, article in enumerate(retrieved, start=1):
        if article in relevant_set:
            return 1.0 / rank
    return 0.0


def _dcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    relevant_set = set(relevant)
    dcg = 0.0
    for rank, article in enumerate(retrieved[:k], start=1):
        if article in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)
    return dcg


def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    ideal_dcg = _dcg_at_k(relevant, relevant, k)
    if ideal_dcg == 0.0:
        return 0.0
    return _dcg_at_k(retrieved, relevant, k) / ideal_dcg


def compute_all(retrieved: List[str], relevant: List[str], k_values=(1, 3, 5, 10)) -> dict:
    result = {"MRR": mrr(retrieved, relevant)}
    for k in k_values:
        result[f"Recall@{k}"]    = recall_at_k(retrieved, relevant, k)
        result[f"Precision@{k}"] = precision_at_k(retrieved, relevant, k)
        result[f"nDCG@{k}"]      = ndcg_at_k(retrieved, relevant, k)
    return result
