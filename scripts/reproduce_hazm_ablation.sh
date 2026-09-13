#!/usr/bin/env bash
set -eu
DEVICE="${DEVICE:-cpu}"
python benchmarks/validate_benchmark.py --profile hazm --split legacy-dev130 --models e5,bge_m3,qwen3
python benchmarks/benchmark.py --profile hazm --split legacy-dev130 --device "$DEVICE" --only "BM25,multilingual-e5-base,multilingual-e5-base+Reranker,multilingual-e5-base+Hybrid,multilingual-e5-base+Hybrid+Reranker,BGE-M3,BGE-M3+Reranker,BGE-M3+Hybrid,BGE-M3+Hybrid+Reranker,Qwen3-Embedding-4B,Qwen3-Embedding-4B+Reranker,Qwen3-Embedding-4B+Hybrid,Qwen3-Embedding-4B+Hybrid+Reranker"
