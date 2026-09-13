# Migration from `finalMatina` and `finalHazm`

The new repository uses `finalMatina` as the baseline implementation and folds the intentional Hazm preprocessing differences from `finalHazm` into one profile-aware preprocessing layer.

Removed sources of drift:

- separate copies of benchmark logic
- hard-coded Dev/Test ground-truth path edits
- separate index roots with ambiguous provenance
- duplicate BM25/dense implementations
- old `__pycache__` files and temporary ground-truth variants

New controls:

```text
--profile baseline|hazm
--split dev|test|legacy-dev130
```

Output workbook names contain both profile and split, making accidental Dev/Test mixing less likely.
