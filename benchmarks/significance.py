#!/usr/bin/env python3
"""Paired nDCG@10 significance analysis across one or more benchmark workbooks.

Example:
  python benchmarks/significance.py \
    --files results/raw_workbooks/test_main.xlsx results/raw_workbooks/test_ablations.xlsx \
    --comparison 'Qwen3-Embedding-4B::BGE-M3' \
    --comparison 'BGE-M3+Reranker::BGE-M3' \
    --comparison 'Matina-Sentence-Embedding+BGE-Reranker-v2-m3::Matina-Sentence-Embedding' \
    --comparison 'BGE-M3+Hybrid::BGE-M3' \
    --comparison 'BGE-M3+Reranker::Qwen3-Embedding-4B'
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down adjusted p-values, returned in original order."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted_sorted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p_values[idx])
        running = max(running, value)
        adjusted_sorted[rank] = running
    out = np.empty(m, dtype=float)
    for rank, idx in enumerate(order):
        out[idx] = adjusted_sorted[rank]
    return out.tolist()


def bootstrap_ci(diff: np.ndarray, n: int = 20000, seed: int = 20260907) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    means = np.empty(n, dtype=float)
    size = len(diff)
    for i in range(n):
        means[i] = diff[rng.integers(0, size, size=size)].mean()
    return tuple(np.quantile(means, [0.025, 0.975]).tolist())


def load_details(files: list[Path]) -> pd.DataFrame:
    frames = []
    for p in files:
        df = pd.read_excel(p, sheet_name="Details")
        df["source_workbook"] = p.name
        frames.append(df)
    merged = pd.concat(frames, ignore_index=True)
    # Re-running the same experiment in multiple supplied workbooks would make pairing ambiguous.
    dup = merged.duplicated(["experiment", "question_id"], keep=False)
    if dup.any():
        sample = merged.loc[dup, ["experiment", "question_id", "source_workbook"]].head(10)
        raise ValueError(f"Duplicate experiment/question rows across input workbooks:\n{sample}")
    return merged


def paired_values(df: pd.DataFrame, a: str, b: str, metric: str):
    aa = df[df.experiment == a].set_index("question_id")[metric]
    bb = df[df.experiment == b].set_index("question_id")[metric]
    common = aa.index.intersection(bb.index)
    if len(common) == 0:
        raise ValueError(f"No paired questions for {a!r} vs {b!r}")
    return aa.loc[common].to_numpy(float), bb.loc[common].to_numpy(float)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--files", nargs="+", type=Path, required=True)
    p.add_argument("--comparison", action="append", required=True,
                   help="A::B, positive delta favors A")
    p.add_argument("--metric", default="nDCG@10")
    p.add_argument("--bootstrap", type=int, default=20000)
    p.add_argument("--seed", type=int, default=20260907)
    p.add_argument("--output", type=Path, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    df = load_details(args.files)
    rows = []
    raw_ps = []
    for spec in args.comparison:
        if "::" not in spec:
            raise ValueError(f"Comparison must be A::B: {spec}")
        a, b = [x.strip() for x in spec.split("::", 1)]
        av, bv = paired_values(df, a, b, args.metric)
        diff = av - bv
        # scipy returns NaN/error for all-zero differences under some versions.
        if np.allclose(diff, 0):
            p_value = 1.0
        else:
            p_value = float(wilcoxon(av, bv, alternative="two-sided").pvalue)
        lo, hi = bootstrap_ci(diff, args.bootstrap, args.seed)
        rows.append({
            "comparison": f"{a} - {b}", "n": len(diff),
            "mean_delta": diff.mean(), "raw_p": p_value,
            "ci95_low": lo, "ci95_high": hi,
        })
        raw_ps.append(p_value)
    adj = holm_adjust(raw_ps)
    for row, p in zip(rows, adj):
        row["holm_p"] = p
    out = pd.DataFrame(rows)[
        ["comparison", "n", "mean_delta", "raw_p", "holm_p", "ci95_low", "ci95_high"]
    ]
    print(out.to_string(index=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(args.output, index=False)
        print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
