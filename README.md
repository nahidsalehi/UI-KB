# UI-KB

**UI-KB** is a Persian institutional retrieval benchmark and retrieval pipeline for higher-education RAG research. The repository consolidates the final baseline implementation and the earlier Hazm retrieval-time preprocessing ablation into **one codebase with explicit profiles**.

> Paper title: **UI-KB: A Persian Institutional Retrieval and Reranking Benchmark for Higher Education**

## What is included

- 377 privacy-refined Persian retrieval chunks in `data/corpus/` (59,046 approximate construction-time tokens)
- human-reviewed development set (`dev_150.json`)
- frozen query-held-out test set (`test_60.json`)
- legacy 130-question development set used for the pre-freeze Hazm ablation
- BM25, multilingual-e5-base, BGE-M3, Qwen3-Embedding-4B, and Matina retrieval support
- FAISS dense indices and BM25 sparse retrieval
- Reciprocal Rank Fusion (RRF) hybrid retrieval
- BGE-Reranker-v2-m3 second-stage reranking
- benchmark metrics, per-domain output, latency, hashes, and validation
- two reproducible preprocessing profiles: `baseline` and `hazm`

## Repository layout

```text
UI-KB/
├── configs/                  # baseline.json / hazm.json
├── data/
│   ├── corpus/               # frozen JSONL chunks
│   └── ground_truth/         # Dev-150, Test-60, legacy Dev-130
├── indices/
│   ├── baseline/             # final paper indices (incl. Matina)
│   └── hazm/                 # legacy Hazm indices (E5/BGE/Qwen)
├── src/                      # application + retrieval implementation
│   └── retrieval/
├── benchmarks/               # benchmark, metrics, preflight validation
├── results/                  # paper summaries + archived raw workbooks
├── scripts/                  # reproducibility and release checks
├── tests/                    # lightweight repository integrity tests
├── docs/                     # profile/release/reproducibility notes
└── templates/                # optional web UI
```

## Preprocessing profiles

### `baseline` — final paper configuration

This is the **source of truth for the final public-release Dev-150 and unchanged held-out Test-60 verification results**.

- BM25: light Persian/Arabic character normalization + whitespace tokenization
- dense retrieval: frozen model-specific query/passage prefixes; no Hazm retrieval-time normalization
- Matina: no E5-style prefix, matching its public Sentence-Transformer usage
- indices: `indices/baseline/`

### `hazm` — legacy pre-freeze ablation

This profile consolidates the previous `finalHazm` branch into the same implementation.

- dense query/passage text: Hazm `Normalizer`
- BM25: Hazm `Normalizer` + Persian-digit mapping + `word_tokenize`
- indices: `indices/hazm/`
- manuscript scope: **development-stage ablation only**
- Matina was integrated after this ablation and is therefore not part of the reported historical Hazm comparison.

See `docs/PROFILES.md` for the exact behavior.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Model weights are not redistributed. Point the repository to local/offline model folders:

```bash
cp .env.example .env
# edit model paths, then export/source them in your shell
```

Required model path variables:

```text
E5_MODEL_PATH
BGE_M3_MODEL_PATH
QWEN3_MODEL_PATH
MATINA_MODEL_PATH
RERANK_MODEL_PATH
```

Model repositories used by the project are expected to correspond to:

```text
intfloat/multilingual-e5-base
BAAI/bge-m3
Qwen/Qwen3-Embedding-4B
MatinaAI/matina-sentence-embedding
BAAI/bge-reranker-v2-m3
```

For an archival release, record the exact model revision/commit rather than only the moving model name.

## Quick integrity check

This check uses only the Python standard library:

```bash
python scripts/check_repository.py
```

Expected core counts:

```text
corpus chunks : 377
dev questions : 150
test questions: 60
missing GT IDs: 0
```

## Validate a benchmark run

Final baseline Dev-150:

```bash
python benchmarks/validate_benchmark.py \
  --profile baseline --split dev \
  --models e5,bge_m3,qwen3,matina
```

Final baseline Test-60:

```bash
python benchmarks/validate_benchmark.py \
  --profile baseline --split test \
  --models e5,bge_m3,qwen3,matina
```

Legacy Hazm Dev-130:

```bash
python benchmarks/validate_benchmark.py \
  --profile hazm --split legacy-dev130 \
  --models e5,bge_m3,qwen3
```

## Reproduce the final paper retrieval runs

Main retriever comparison on Dev-150:

```bash
python benchmarks/benchmark.py --profile baseline --split dev --device cpu \
  --only "BM25,multilingual-e5-base,BGE-M3,Qwen3-Embedding-4B,Matina-Sentence-Embedding"
```

Locked Dev ablations:

```bash
python benchmarks/benchmark.py --profile baseline --split dev --device cpu \
  --only "BGE-M3+Reranker,BGE-M3+Hybrid,Matina-Sentence-Embedding+BGE-Reranker-v2-m3"
```

Held-out Test-60 main comparison:

```bash
python benchmarks/benchmark.py --profile baseline --split test --device cpu \
  --only "BM25,multilingual-e5-base,BGE-M3,Qwen3-Embedding-4B,Matina-Sentence-Embedding"
```

Held-out locked ablations:

```bash
python benchmarks/benchmark.py --profile baseline --split test --device cpu \
  --only "BGE-M3+Reranker,BGE-M3+Hybrid,Matina-Sentence-Embedding+BGE-Reranker-v2-m3"
```

The test set is provided for reproducibility of the published evaluation. It must **not** be used for additional tuning if extending the original study.

## Reproduce the historical Hazm ablation

The exact reported Hazm analysis predates Matina and the final Dev/Test freeze:

```bash
python benchmarks/benchmark.py --profile hazm --split legacy-dev130 --device cpu \
  --only "BM25,multilingual-e5-base,BGE-M3,Qwen3-Embedding-4B"
```

Do not describe a new Hazm run on Dev-150/Test-60 as one of the original paper experiments unless it is explicitly reported as a post-hoc extension.

## Build indices from scratch

Baseline:

```bash
python src/embedder.py --profile baseline --device cpu \
  --model e5,bge_m3,qwen3,matina
```

Hazm legacy profile:

```bash
python src/embedder.py --profile hazm --device cpu \
  --model e5,bge_m3,qwen3
```

## Optional web application

```bash
export UIKB_PROFILE=baseline
export APP_EMBED_MODEL=bge_m3
export APP_USE_HYBRID=false
export APP_USE_RERANKER=true
python src/app.py
```

The paper evaluates retrieval rather than generated-answer correctness; the web application is therefore supplementary and is not part of the final retrieval benchmark claims.

## Reproducibility artifacts

- `results/paper_results_summary.csv`: final public-release Dev/Test summary used in the revised manuscript
- `results/hazm_dev130_summary.csv`: legacy Hazm ablation summary
- `results/raw_workbooks/`: final verification workbooks plus the explicitly labeled pre-sanitization Dev ablation provenance workbook
- `results/final_test_significance.csv`: five pre-specified paired held-out comparisons with Holm correction and paired-bootstrap intervals
- `benchmarks/significance.py`: script used to recompute the paired statistical table from the frozen workbooks
- `SHA256SUMS.txt`: release-file hashes
- `docs/REPRODUCIBILITY.md`: exact scope and provenance notes


## Public-release verification note

Before the final archival run, selected support chunks were privacy-sanitized to remove unnecessary operational detail and three Dev-150 questions were wording-clarified without changing relevance labels. These edits were not performance-motivated. Test-60 questions/labels and all retrieval configurations remained unchanged; indices were rebuilt and final metrics were recomputed without further tuning.

Final benchmark hashes:

```text
Dev-150  : 8d78ace5d05fbf5c987ac38e8ba3d696d670726a6604c3942672150c5d0fd864
Test-60  : a5c5d38ef2b1b4418ec95ad58a2f00271c72a06c0e58e60a67b566f313cfd290
Corpus   : 7c44d2155f9a2ed84eb65c59907c054430caf37f833f305604385c74c6fc7a11
```

## Citation

Before the public release, replace the placeholder repository URL/DOI in `CITATION.cff` and this README with the final GitHub/Zenodo identifiers.

Suggested repository citation:

```text
Salehi, N., Zamani, A., & Montazerolghaem, A. (2026). UI-KB: Persian Institutional Retrieval Benchmark [Software and dataset]. GitHub/Zenodo. DOI: <REPLACE_WITH_DOI>
```

## Data-release warning

The corpus contains extracted institutional text. **Do not make the repository public until the redistribution rights and privacy of every source have been reviewed.** A lightweight scanner is included:

```bash
python scripts/privacy_scan.py
```

See `docs/PUBLIC_RELEASE_CHECKLIST.md`.

## License

## License and reuse

The source code in this repository is released under the MIT License.

The institutional corpus and benchmark annotations are provided for research
and reproducibility purposes. The MIT License applies to the software code
only and does not grant additional rights over the original institutional
source materials. Reuse and redistribution of corpus content are subject to
the rights applicable to the original sources.
