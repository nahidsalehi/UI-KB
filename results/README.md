# Results

`paper_results_summary.csv` is the compact summary of the **final public-release verification runs** used in the revised manuscript: Dev-150 main retrievers, Test-60 main retrievers, and the three locked Test-60 ablations.

`final_test_significance.csv` contains the five pre-specified paired Test-60 nDCG@10 comparisons with Holm correction and paired-bootstrap 95% confidence intervals.

`hazm_dev130_summary.csv` summarizes the legacy Hazm-profile Dev-130 rerun on the final public corpus. This remains a development-stage ablation and is not held-out evidence.

`raw_workbooks/` contains the final Dev-150 main, Test-60 main, Test-60 ablation, and legacy Hazm workbooks. `dev_ablations_pre_sanitization.xlsx` is retained only for provenance of the earlier development-stage selection and is not used for final public-corpus metrics.

`final_run/` contains the final logs, validation output, corpus statistics, and workbooks captured on 2026-09-12.

Generate the three descriptive paper figures from existing workbooks (no models required):

```bash
python scripts/plot_benchmark_analysis.py --input results/raw_workbooks/test_main.xlsx results/raw_workbooks/test_ablations.xlsx --output-dir results/generated/figures
```

For another compatible export, use `--input path/to/benchmark.xlsx`. Multiple inputs must
have disjoint configuration/query rows, identical query sets and domain assignments,
and matching profile, split, ground-truth hash and corpus hash when recorded in RunConfig.
The main and ablation archives together cover all eight locked configurations on Test-60.
No fresh benchmark run is needed for these figures.

The script requires the `Details` sheet's `experiment`, `question_id`, `domain`,
`nDCG@10`, and `Recall@1`, `Recall@3`, `Recall@5`, `Recall@10` columns. It uses the
unrounded per-query exports for means rather than the rounded Summary/ByDomain sheets.
Recall is checked for non-decreasing values at both query and configuration levels;
violations, duplicate observations, missing metrics and inconsistent query coverage fail
before files are written. Partial configuration coverage emits a warning; absent domains
are shown as N/A, never zero. Domain labels include unique query sample sizes.
These plots are descriptive and make no statistical winner claims for domain subsets;
the existing Wilcoxon/bootstrap analysis is unchanged.
Native workbook domain labels `discipline`, `enteghali`, `vaam`, and `lms` are displayed
as `disciplinary`, `transfer`, `loans`, and `LMS`; `source_domain` in the per-query CSV
preserves the input labels. Workbook and ground-truth files are never rewritten.

Outputs are 300-dpi PNG and vector PDF versions of `ndcg10_domain_heatmap`,
`recall_at_k`, and `ndcg10_per_query_boxplot`, plus `domain_ndcg10_matrix.csv`,
`domain_sample_sizes.csv`, `recall_at_k.csv`, and `per_query_ndcg10.csv`.
The boxplot uses actual observations and standard matplotlib box/whisker summaries.
There is no interpolation, random jitter, or recomputation of retrieval metrics.
The ten verified Test-60 figure/data files in `generated/figures/` are tracked for
paper review. Other generated outputs and local matplotlib caches remain ignored.
Regenerating these figures replaces the tracked files; use a different output directory
when exploring another compatible workbook.

Lightweight verification without ML dependencies:

```bash
python -m pip install pandas openpyxl matplotlib pytest
python -m pytest tests/test_benchmark_analysis.py tests/test_repository.py
```
