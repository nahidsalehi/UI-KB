# Changelog

## v1.0.0-paper-final (2026-09-12)

- Applied privacy-motivated cleanup to selected support text while preserving chunk IDs and answer-bearing content.
- Clarified three Dev-150 question wordings without changing relevance labels; held-out Test-60 remained unchanged.
- Rebuilt baseline and Hazm FAISS indices on the final public corpus.
- Recomputed final Dev-150 main, Test-60 main, Test-60 locked ablations, paired significance tests, and legacy Hazm results without retuning.
- Updated compact result summaries, final raw workbooks, corpus statistics, and frozen hashes.
- Removed the accidental local `.venv` artifact from the release tree.

## v1.0.0-paper (prepared 2026-09-08)

- Merged the final baseline/Matina branch and legacy Hazm branch into one profile-aware codebase.
- Added explicit `--profile baseline|hazm` and `--split dev|test|legacy-dev130` controls.
- Added final Dev-150 and frozen Test-60 ground truth.
- Added baseline and Hazm index namespaces to prevent accidental cross-profile reuse.
- Added repository integrity, privacy, release, and reproducibility documentation.
