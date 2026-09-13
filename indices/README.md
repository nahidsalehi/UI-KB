# Prebuilt indices

`baseline/` contains the frozen dense FAISS indices from the final implementation, including Matina.
`hazm/` contains the legacy Hazm-normalized indices for E5, BGE-M3, and Qwen3.

Model weights are not included. The metadata files retain chunk text/provenance used at retrieval time.

Indices can be rebuilt with `src/embedder.py`; rebuilding with a different model revision, library version, or hardware may produce numerically different embeddings even when rankings are similar.
