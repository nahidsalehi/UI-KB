# Reproducibility notes

## Frozen benchmark sets

- final development: `data/ground_truth/dev_150.json`
- unchanged held-out test: `data/ground_truth/test_60.json`
- legacy development: `data/ground_truth/legacy_dev_130.json`

The held-out test is query-held-out over the same 377-chunk corpus. It is included to reproduce the published evaluation, not for additional tuning.

Final ground-truth hashes:

- Dev-150 SHA256: `8d78ace5d05fbf5c987ac38e8ba3d696d670726a6604c3942672150c5d0fd864`
- Test-60 SHA256: `a5c5d38ef2b1b4418ec95ad58a2f00271c72a06c0e58e60a67b566f313cfd290`
- Legacy Dev-130 SHA256: `5e6f3b36b2572006eef3cf35c6827fe1cb95a521080b049e3953d29270a2a237`

The final benchmark workbooks report corpus SHA256 `7c44d2155f9a2ed84eb65c59907c054430caf37f833f305604385c74c6fc7a11`.

## Final paper protocol

The final paper configuration is `baseline`. Model/component selection was completed before held-out inspection. Before the final public release, unnecessary operational details were removed from selected support chunks for privacy, and three Dev-150 questions were wording-clarified without changing relevance labels. These edits were not performance-motivated. The Test-60 questions/labels and all retrieval configurations remained unchanged; indices were rebuilt and all final metrics in the revised manuscript were recomputed without further tuning.

`results/raw_workbooks/dev_main.xlsx`, `test_main.xlsx`, and `test_ablations.xlsx` are the final public-corpus verification workbooks used in the revised manuscript. The older `dev_ablations_pre_sanitization.xlsx` is retained only for provenance of pre-release development selection.

## Legacy Hazm protocol

The `hazm` profile represents the earlier retrieval-time preprocessing ablation and uses `legacy_dev_130.json`. The profile was rerun on the final public corpus for repository consistency. It remains development-stage evidence and is not used to tune or support the held-out Test-60 conclusions.

## Model weights

Weights are not committed. Record the exact Hugging Face revision or local model SHA for each archival release:

- multilingual-e5-base
- BGE-M3
- Qwen3-Embedding-4B
- Matina sentence embedding
- BGE-Reranker-v2-m3

## Environment

The final verification logs record Python 3.10.12 and an Intel Xeon Platinum 8180 @ 2.50 GHz CPU environment with 56 exposed CPUs. `requirements.txt` lists required packages, but an exact historical package lock was not available from the original folders. If possible before DOI archival, add `requirements-lock.txt` from the final server environment.

## Hashes

Run:

```bash
python scripts/check_repository.py
sha256sum -c SHA256SUMS.txt
```

Regenerate `SHA256SUMS.txt` only when intentionally creating a new archived release.
