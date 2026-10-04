"""Reporting tests use synthetic exports only; no retrieval/model imports."""
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from plot_benchmark_analysis import (
    CONFIGURATIONS, DOMAINS, K_VALUES, domain_aggregation, generate_analysis,
    load_details, per_query_ndcg, recall_at_k, validate_details,
)


def synthetic_details():
    rows = []
    for name in CONFIGURATIONS:
        for i, domain in enumerate(DOMAINS):
            for j in range(2):
                rows.append({
                    "experiment": name, "question_id": 2 * i + j, "domain": domain,
                    "nDCG@10": 0.2 + 0.4 * j,
                    "Recall@1": 0.1 + 0.1 * j, "Recall@3": 0.3,
                    "Recall@5": 0.5, "Recall@10": 0.9,
                })
    return pd.DataFrame(rows)


def write_workbook(path, details, split="test"):
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        details.to_excel(writer, sheet_name="Details", index=False)
        pd.DataFrame({"key": ["split"], "value": [split]}).to_excel(
            writer, sheet_name="RunConfig", index=False
        )


class AnalysisTests(unittest.TestCase):
    def setUp(self):
        self.details = synthetic_details()

    def test_parse_and_combine_workbooks(self):
        with tempfile.TemporaryDirectory() as folder:
            paths = [Path(folder) / "main.xlsx", Path(folder) / "ablations.xlsx"]
            write_workbook(paths[0], self.details[self.details.experiment.isin(CONFIGURATIONS[:5])])
            write_workbook(paths[1], self.details[self.details.experiment.isin(CONFIGURATIONS[5:])])
            parsed = load_details(paths)
            self.assertEqual(len(parsed), 96)
            self.assertEqual(parsed.experiment.nunique(), 8)
            self.assertEqual(set(parsed.source_workbook), set(map(str, paths)))
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                load_details([paths[0], paths[0]])
            write_workbook(paths[1], self.details[self.details.experiment.isin(CONFIGURATIONS[5:])],
                           split="dev")
            with self.assertRaisesRegex(ValueError, "RunConfig split differs"):
                load_details(paths)

    def test_missing_sheet_and_columns(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "bad.xlsx"
            self.details.to_excel(path, sheet_name="Summary", index=False)
            with self.assertRaisesRegex(ValueError, "missing required sheet 'Details'"):
                load_details([path])
        with self.assertRaisesRegex(ValueError, "missing required columns: Recall@3"):
            validate_details(self.details.drop(columns="Recall@3"))
        with self.assertRaisesRegex(ValueError, "no per-query rows"):
            validate_details(self.details.iloc[:0])

    def test_native_workbook_domain_labels(self):
        native = self.details.copy()
        native["domain"] = native["domain"].replace({
            "disciplinary": "discipline", "transfer": "enteghali", "loans": "vaam", "LMS": "lms",
        })
        parsed = validate_details(native)
        self.assertEqual(set(parsed.domain), set(DOMAINS))
        pd.testing.assert_series_equal(parsed.source_domain, native.domain, check_names=False)
        matrix, counts = domain_aggregation(parsed)
        np.testing.assert_allclose(matrix.to_numpy(), 0.4)
        np.testing.assert_array_equal(counts.to_numpy(), 2)
        query = per_query_ndcg(parsed)
        self.assertEqual(query.loc[query.domain.eq("loans"), "source_domain"].unique().tolist(),
                         ["vaam"])

    def test_invalid_values_and_pairing(self):
        for value in (np.nan, np.inf, -0.1, 1.1, "bad"):
            details = self.details.copy()
            details["nDCG@10"] = details["nDCG@10"].astype(object)
            details.loc[0, "nDCG@10"] = value
            with self.assertRaisesRegex(ValueError, "finite numeric values"):
                validate_details(details)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            validate_details(pd.concat([self.details, self.details.iloc[:1]]))
        with self.assertRaisesRegex(ValueError, "different question_id sets"):
            validate_details(self.details.drop(index=0))
        details = self.details.copy()
        details.loc[0, "domain"] = "email"
        with self.assertRaisesRegex(ValueError, "Domains disagree"):
            validate_details(details)
        details.loc[0, "domain"] = "unknown"
        with self.assertRaisesRegex(ValueError, "Unsupported Details domains"):
            validate_details(details)
        details.loc[0, "domain"] = ""
        with self.assertRaisesRegex(ValueError, "missing/blank domain"):
            validate_details(details)

    def test_domain_means_and_sample_sizes(self):
        matrix, counts = domain_aggregation(validate_details(self.details))
        self.assertEqual(matrix.index.tolist(), list(CONFIGURATIONS))
        self.assertEqual(matrix.columns.tolist(), list(DOMAINS))
        np.testing.assert_allclose(matrix.to_numpy(), 0.4)
        np.testing.assert_array_equal(counts.to_numpy(), 2)
        matrix, counts = domain_aggregation(self.details[self.details.domain.eq("academic")])
        self.assertTrue(matrix["LMS"].isna().all())
        self.assertTrue(counts["LMS"].eq(0).all())

    def test_recall_reshaping(self):
        recall = recall_at_k(self.details)
        self.assertEqual(len(recall), 8 * 4)
        first = recall[recall.experiment.eq("BM25")]
        self.assertEqual(first.k.tolist(), list(K_VALUES))
        np.testing.assert_allclose(first.recall, [0.15, 0.3, 0.5, 0.9])

    def test_monotonicity_fails_before_output(self):
        details = self.details.copy()
        details.loc[0, "Recall@3"] = 0
        with self.assertRaisesRegex(ValueError, "not non-decreasing.*BM25"):
            validate_details(details)
        # A single decreasing query is rejected even if the means increase.
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "figures"
            with self.assertRaisesRegex(ValueError, "not non-decreasing"):
                generate_analysis(details, output)
            self.assertFalse(output.exists())
        details["Recall@3"] = 0
        with self.assertRaisesRegex(ValueError, "Mean Recall is not non-decreasing"):
            recall_at_k(details)

    def test_per_query_extraction_preserves_observations(self):
        query = per_query_ndcg(self.details)
        self.assertEqual(len(query), len(self.details))
        self.assertEqual(query.columns.tolist(), ["experiment", "question_id", "domain", "nDCG@10"])
        pd.testing.assert_series_equal(query["nDCG@10"], self.details["nDCG@10"])
        self.assertEqual(query[query.experiment.eq("BM25")]["nDCG@10"].nunique(), 2)

    def test_cli_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "tiny.xlsx"
            output = Path(folder) / "figures"
            write_workbook(path, self.details)
            result = subprocess.run([
                sys.executable, str(ROOT / "scripts/plot_benchmark_analysis.py"),
                "--input", str(path), "--output-dir", str(output),
            ], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            expected = {f"{stem}.{extension}" for stem in (
                "ndcg10_domain_heatmap", "recall_at_k", "ndcg10_per_query_boxplot"
            ) for extension in ("png", "pdf")}
            expected.update({"domain_ndcg10_matrix.csv", "domain_sample_sizes.csv",
                             "recall_at_k.csv", "per_query_ndcg10.csv"})
            self.assertEqual({p.name for p in output.iterdir()}, expected)
            self.assertTrue(all(p.stat().st_size > 0 for p in output.iterdir()))
            matrix = pd.read_csv(output / "domain_ndcg10_matrix.csv", index_col="experiment")
            np.testing.assert_allclose(matrix.to_numpy(), 0.4)
            self.assertEqual(len(pd.read_csv(output / "per_query_ndcg10.csv")), 96)

    def test_partial_coverage_and_closed_figures(self):
        with tempfile.TemporaryDirectory() as folder:
            tiny = self.details[self.details.experiment.eq("BM25") &
                                self.details.domain.eq("academic")]
            with self.assertWarnsRegex(UserWarning, "Partial configuration coverage"):
                generate_analysis(tiny, Path(folder))
            self.assertEqual(plt.get_fignums(), [])


if __name__ == "__main__":
    unittest.main()
