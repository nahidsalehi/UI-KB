#!/usr/bin/env bash
set -eu
DEVICE="${DEVICE:-cpu}"
python benchmarks/validate_benchmark.py --profile baseline --split dev --models e5,bge_m3,qwen3,matina
python benchmarks/benchmark.py --profile baseline --split dev --device "$DEVICE" --only "BM25,multilingual-e5-base,BGE-M3,Qwen3-Embedding-4B,Matina-Sentence-Embedding"
python benchmarks/benchmark.py --profile baseline --split dev --device "$DEVICE" --only "BGE-M3+Reranker,BGE-M3+Hybrid,Matina-Sentence-Embedding+BGE-Reranker-v2-m3"
python benchmarks/validate_benchmark.py --profile baseline --split test --models e5,bge_m3,qwen3,matina
python benchmarks/benchmark.py --profile baseline --split test --device "$DEVICE" --only "BM25,multilingual-e5-base,BGE-M3,Qwen3-Embedding-4B,Matina-Sentence-Embedding"
python benchmarks/benchmark.py --profile baseline --split test --device "$DEVICE" --only "BGE-M3+Reranker,BGE-M3+Hybrid,Matina-Sentence-Embedding+BGE-Reranker-v2-m3"
