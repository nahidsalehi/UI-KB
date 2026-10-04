"""Descriptive figures from existing UI-KB Details sheets; no model or test reruns.

All means use the exported per-query values (Summary/ByDomain are rounded).
Multiple inputs allow combining the frozen main and ablation workbooks.
Missing configurations/domains are reported, never filled with invented scores.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import warnings
from zipfile import BadZipFile

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

K_VALUES = (1, 3, 5, 10)
RECALL_COLUMNS = [f"Recall@{k}" for k in K_VALUES]
DOMAINS = ("academic", "disciplinary", "transfer", "loans", "email", "LMS")
# Native labels verified in the archived Test-60 Details and ByDomain sheets.
DOMAIN_LABELS = {
    "discipline": "disciplinary", "enteghali": "transfer", "vaam": "loans", "lms": "LMS",
}
# Frozen main retrievers plus the three ablations in scripts/reproduce_paper.sh.
CONFIGURATIONS = (
    "BM25", "multilingual-e5-base", "BGE-M3", "Qwen3-Embedding-4B",
    "Matina-Sentence-Embedding", "BGE-M3+Reranker", "BGE-M3+Hybrid",
    "Matina-Sentence-Embedding+BGE-Reranker-v2-m3",
)
REQUIRED_COLUMNS = ["experiment", "question_id", "domain", "nDCG@10", *RECALL_COLUMNS]


def validate_details(frame: pd.DataFrame) -> pd.DataFrame:
    """Validate actual benchmark schema and paired query coverage without dropping rows."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(frame.columns))
    if missing:
        raise ValueError(f"Details is missing required columns: {', '.join(missing)}")
    if frame.empty:
        raise ValueError("Details contains no per-query rows.")
    df = frame.copy()
    for column in ("experiment", "question_id", "domain"):
        if df[column].isna().any() or df[column].astype(str).str.strip().eq("").any():
            raise ValueError(f"Details has missing/blank {column} values.")
    if "source_domain" not in df.columns:
        df["source_domain"] = df["domain"]
    df["domain"] = df["domain"].replace(DOMAIN_LABELS)
    unknown = set(df["domain"]) - set(DOMAINS)
    if unknown:
        raise ValueError(f"Unsupported Details domains: {sorted(unknown)}; expected {DOMAINS}")
    for column in ["nDCG@10", *RECALL_COLUMNS]:
        df[column] = pd.to_numeric(df[column], errors="coerce")
        values = df[column].to_numpy(dtype=float)
        if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
            raise ValueError(f"Details {column} must contain finite numeric values in [0, 1].")
    if df.duplicated(["experiment", "question_id"]).any():
        raise ValueError("Duplicate experiment/question_id rows in input workbook(s).")
    if df.groupby("question_id")["domain"].nunique().gt(1).any():
        raise ValueError("Domains disagree for the same question_id across configurations.")
    query_sets = df.groupby("experiment", sort=False)["question_id"].apply(set)
    first = query_sets.iloc[0]
    if any(queries != first for queries in query_sets):
        raise ValueError("Configurations have different question_id sets; do not mix splits/subsets.")
    decreases = np.diff(df[RECALL_COLUMNS].to_numpy(dtype=float), axis=1) < -1e-12
    if decreases.any():
        row = df.iloc[np.flatnonzero(decreases.any(axis=1))[0]]
        raise ValueError(
            f"Recall is not non-decreasing for {row['experiment']}, "
            f"question_id={row['question_id']}: {row[RECALL_COLUMNS].tolist()}"
        )
    return df


def load_details(paths: list[Path]) -> pd.DataFrame:
    frames, metadata = [], {}
    for path in paths:
        try:
            with pd.ExcelFile(path, engine="openpyxl") as workbook:
                if "Details" not in workbook.sheet_names:
                    raise ValueError("missing required sheet 'Details'")
                frame = validate_details(pd.read_excel(workbook, sheet_name="Details"))
                if "RunConfig" in workbook.sheet_names:
                    config = pd.read_excel(workbook, sheet_name="RunConfig")
                    if not {"key", "value"}.issubset(config.columns):
                        raise ValueError("RunConfig requires key and value columns")
                    config = config.set_index("key")["value"].to_dict()
                    for key in ("profile", "split", "gt_sha256", "corpus_sha256"):
                        value = config.get(key)
                        if value is not None and pd.notna(value):
                            if key in metadata and metadata[key] != value:
                                raise ValueError(f"RunConfig {key} differs across input workbooks")
                            metadata[key] = value
                frame["source_workbook"] = str(path)
                frames.append(frame)
        except (OSError, ValueError, BadZipFile) as exc:
            raise ValueError(f"Incompatible workbook {path}: {exc}") from exc
    if not frames:
        raise ValueError("At least one input workbook is required.")
    return validate_details(pd.concat(frames, ignore_index=True))


def configuration_order(details: pd.DataFrame) -> list[str]:
    available = details["experiment"].unique().tolist()
    return [name for name in CONFIGURATIONS if name in available] + [
        name for name in available if name not in CONFIGURATIONS
    ]


def domain_aggregation(details: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    grouped = details.groupby(["experiment", "domain"])["nDCG@10"]
    order = configuration_order(details)
    matrix = grouped.mean().unstack().reindex(index=order, columns=DOMAINS)
    counts = grouped.count().unstack().reindex(index=order, columns=DOMAINS).fillna(0)
    return matrix, counts.astype(int)


def recall_at_k(details: pd.DataFrame) -> pd.DataFrame:
    means = details.groupby("experiment")[RECALL_COLUMNS].mean()
    means = means.reindex(configuration_order(details))
    for name, row in means.iterrows():
        if (np.diff(row.to_numpy(dtype=float)) < -1e-12).any():
            raise ValueError(f"Mean Recall is not non-decreasing for {name}: {row.tolist()}")
    return pd.DataFrame([
        {"experiment": name, "k": k, "recall": row[f"Recall@{k}"]}
        for name, row in means.iterrows() for k in K_VALUES
    ])


def per_query_ndcg(details: pd.DataFrame) -> pd.DataFrame:
    columns = ["experiment", "question_id", "domain", "nDCG@10"]
    if "source_domain" in details.columns:
        columns.append("source_domain")
    if "source_workbook" in details.columns:
        columns.append("source_workbook")
    return pd.concat([
        details.loc[details["experiment"].eq(name), columns]
        for name in configuration_order(details)
    ], ignore_index=True)


def display_label(name: str) -> str:
    return name.replace("+", "\n+ ")


def save_figure(fig, output_dir: Path, stem: str) -> None:
    try:
        for extension in ("png", "pdf"):
            fig.savefig(output_dir / f"{stem}.{extension}", dpi=300, bbox_inches="tight")
    finally:
        plt.close(fig)


def generate_analysis(details: pd.DataFrame, output_dir: Path) -> None:
    details = validate_details(details)
    order = configuration_order(details)
    missing = [name for name in CONFIGURATIONS if name not in order]
    if missing:
        warnings.warn(f"Partial configuration coverage; missing locked configurations: {missing}")
    absent_domains = [domain for domain in DOMAINS if domain not in set(details["domain"])]
    if absent_domains:
        warnings.warn(f"Missing domains shown as N/A (not zero): {absent_domains}")
    matrix, counts = domain_aggregation(details)
    recall = recall_at_k(details)
    queries = per_query_ndcg(details)
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix.to_csv(output_dir / "domain_ndcg10_matrix.csv", index_label="experiment")
    counts.to_csv(output_dir / "domain_sample_sizes.csv", index_label="experiment")
    recall.to_csv(output_dir / "recall_at_k.csv", index=False)
    queries.to_csv(output_dir / "per_query_ndcg10.csv", index=False)

    with plt.rc_context({"font.size": 10, "pdf.fonttype": 42}):
        fig, ax = plt.subplots(figsize=(10, max(3, 0.65 * len(order))))
        cmap = plt.get_cmap("viridis").copy()
        cmap.set_bad("#eeeeee")
        plot = ax.imshow(matrix.to_numpy(float), vmin=0, vmax=1, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(DOMAINS)), [
            f"{domain}\n(n={counts.iloc[0][domain]})" for domain in DOMAINS
        ])
        ax.set_yticks(range(len(order)), [display_label(name) for name in order])
        for i in range(len(order)):
            for j in range(len(DOMAINS)):
                value = matrix.iloc[i, j]
                label = "N/A" if pd.isna(value) else f"{value:.3f}"
                color = "white" if pd.notna(value) and value < 0.5 else "black"
                ax.text(j, i, label, ha="center", va="center", color=color)
        fig.colorbar(plot, ax=ax, label="Mean nDCG@10")
        save_figure(fig, output_dir, "ndcg10_domain_heatmap")

        fig, ax = plt.subplots(figsize=(9, 5))
        for i, name in enumerate(order):
            line = recall.loc[recall["experiment"].eq(name)]
            ax.plot(line["k"], line["recall"], marker="os^DvPX*"[i % 8],
                    color=plt.get_cmap("tab10")(i % 10), label=display_label(name))
        ax.set(xticks=K_VALUES, xlabel="Retrieval depth (k)", ylabel="Mean Recall@k", ylim=(0, 1.03))
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1), frameon=False, fontsize=9)
        save_figure(fig, output_dir, "recall_at_k")

        fig, ax = plt.subplots(figsize=(max(8, len(order) * 1.1), 5))
        values = [queries.loc[queries["experiment"].eq(name), "nDCG@10"].to_numpy()
                  for name in order]
        ax.boxplot(values, showfliers=True)
        ax.set_xticks(range(1, len(order) + 1), [display_label(name) for name in order],
                      rotation=35, ha="right", fontsize=9)
        ax.set(ylabel="Per-query nDCG@10", ylim=(-0.03, 1.03))
        ax.grid(axis="y", alpha=0.25)
        save_figure(fig, output_dir, "ndcg10_per_query_boxplot")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", nargs="+", type=Path, required=True,
                        help="Existing benchmark XLSX file(s) with compatible Details sheets")
    parser.add_argument("--output-dir", type=Path, default=Path("results/generated/figures"))
    args = parser.parse_args()
    try:
        details = load_details(args.input)
        generate_analysis(details, args.output_dir)
    except (ValueError, OSError) as exc:
        parser.exit(2, f"Analysis error: {exc}\n")
    print(f"Saved three PNG/PDF figures and four CSV files to {args.output_dir}")
    print(f"Configurations: {details['experiment'].nunique()}; "
          f"questions per configuration: {details['question_id'].nunique()}")


if __name__ == "__main__":
    main()
