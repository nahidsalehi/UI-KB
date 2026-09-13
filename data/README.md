# Data

`corpus/` contains the 377 privacy-refined JSONL retrieval chunks used for the final public-release verification runs (59,046 approximate construction-time tokens).

`ground_truth/` contains:

- `dev_150.json`: 150-question human-reviewed development set; three questions were wording-clarified before the final public-release verification while relevance labels were preserved.
- `test_60.json`: unchanged 60-question query-held-out test set.
- `legacy_dev_130.json`: development set used for the historical Hazm ablation.

The final Test-60 questions and relevance labels were not modified after held-out inspection. The privacy-driven corpus cleanup was not performance-motivated; configurations were kept fixed, indices were rebuilt, and final reported metrics were recomputed without tuning.

Before public redistribution, complete `docs/PUBLIC_RELEASE_CHECKLIST.md`. In particular, verify the legal basis for redistributing extracted institutional text and manually review any strings reported by `python scripts/privacy_scan.py`.
