"""Reproducible retrieval benchmark for UI-KB.

The final paper uses profile=baseline with split=dev/test. The profile=hazm
option reproduces the legacy pre-freeze Hazm ablation; the reported historical
Hazm comparison used split=legacy-dev130 and did not include Matina.
"""

import argparse
import hashlib
import json
import sys
import time
from datetime import datetime
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
sys.path.insert(0, str(Path(__file__).parent))

import config
from metrics import compute_all
from retrieval import BM25Retriever, DenseRetriever, HybridRetriever, Reranker
from utils import normalize_query

K_VALUES = [1, 3, 5, 10]
EVAL_DEPTH = max(K_VALUES)
DEFAULT_RESULTS_DIR = ROOT / "results" / "generated"
GT_SPLITS = {
    "dev": config.GROUND_TRUTH_DIR / "dev_150.json",
    "test": config.GROUND_TRUTH_DIR / "test_60.json",
    "legacy-dev130": config.GROUND_TRUTH_DIR / "legacy_dev_130.json",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def corpus_files():
    return sorted(config.JSON_DIR.glob(config.JSONL_PATTERN))


def corpus_sha256():
    h = hashlib.sha256()
    for path in corpus_files():
        h.update(path.name.encode("utf-8"))
        with path.open("rb") as f:
            for block in iter(lambda: f.read(1024 * 1024), b""):
                h.update(block)
    return h.hexdigest()


def corpus_stats():
    chunks, ids, domains = 0, set(), {}
    for path in corpus_files():
        domain = path.stem.replace("_v2", "")
        count = 0
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if not (rec.get("text") or "").strip():
                    continue
                chunks += 1; count += 1
                if rec.get("id") is not None:
                    ids.add(str(rec["id"]))
        domains[domain] = domains.get(domain, 0) + count
    return {"jsonl_files": len(corpus_files()), "corpus_chunks": chunks,
            "unique_chunk_ids": len(ids), **{f"chunks_{k}": v for k,v in sorted(domains.items())}}


def resolve_gt(args) -> Path:
    return Path(args.gt_file).resolve() if args.gt_file else GT_SPLITS[args.split]


def search_for_evaluation(retriever, question, top_k):
    if isinstance(retriever, (DenseRetriever, HybridRetriever)):
        return retriever.search(question, top_k=top_k, min_score=0.0)
    return retriever.search(question, top_k=top_k)


def chunk_ids(results):
    return [str(r["id"]) for r in results if r.get("id") is not None]


def run_experiment(name, ground_truth, retriever, reranker=None):
    rows = []
    for item in ground_truth:
        question = normalize_query(item["question"])
        relevant = [str(x) for x in item["relevant_answer"]]
        t0 = time.perf_counter()
        candidates = search_for_evaluation(retriever, question, EVAL_DEPTH)
        final = reranker.rerank(question, candidates, top_k=EVAL_DEPTH) if reranker else candidates[:EVAL_DEPTH]
        latency_ms = (time.perf_counter() - t0) * 1000.0
        retrieved = chunk_ids(final)
        rows.append({
            "experiment": name, "question_id": item["id"], "question": item["question"],
            "domain": item.get("domain", ""), "question_type": item.get("question_type", ""),
            "relevant": ", ".join(relevant), "retrieved": ", ".join(retrieved),
            "latency_ms": latency_ms, **compute_all(retrieved, relevant, K_VALUES),
        })
    return rows


def build_experiments(device):
    exps = [{"name":"BM25", "factory":lambda: BM25Retriever(), "use_reranker":False}]
    for model_key in ["e5", "bge_m3", "qwen3", "matina"]:
        label = config.MODELS[model_key]["label"]
        rerank_label = f"{label}+BGE-Reranker-v2-m3" if model_key == "matina" else f"{label}+Reranker"
        hybrid_rerank_label = f"{label}+Hybrid+BGE-Reranker-v2-m3" if model_key == "matina" else f"{label}+Hybrid+Reranker"
        exps.extend([
            {"name":label, "factory":lambda mk=model_key: DenseRetriever(mk, device=device), "use_reranker":False},
            {"name":rerank_label, "factory":lambda mk=model_key: DenseRetriever(mk, device=device), "use_reranker":True},
            {"name":f"{label}+Hybrid", "factory":lambda mk=model_key: HybridRetriever(DenseRetriever(mk, device=device), BM25Retriever()), "use_reranker":False},
            {"name":hybrid_rerank_label, "factory":lambda mk=model_key: HybridRetriever(DenseRetriever(mk, device=device), BM25Retriever()), "use_reranker":True},
        ])
    return exps


def parse_args():
    p = argparse.ArgumentParser(description="UI-KB retrieval benchmark")
    p.add_argument("--profile", choices=["baseline", "hazm"], default="baseline")
    p.add_argument("--split", choices=list(GT_SPLITS), default="dev")
    p.add_argument("--gt-file", default="", help="Optional custom GT JSON; overrides --split")
    p.add_argument("--device", default="cpu")
    p.add_argument("--skip", default="")
    p.add_argument("--only", default="")
    p.add_argument("--list-experiments", action="store_true")
    p.add_argument("--output-dir", default=str(DEFAULT_RESULTS_DIR))
    return p.parse_args()


def main():
    args = parse_args()
    config.set_profile(args.profile)
    exps = build_experiments(args.device)
    if args.list_experiments:
        print("\n".join(e["name"] for e in exps)); return

    gt_file = resolve_gt(args)
    gt = json.loads(gt_file.read_text(encoding="utf-8"))
    skip = {s.strip() for s in args.skip.split(",") if s.strip()}
    only = {s.strip() for s in args.only.split(",") if s.strip()}
    should_run = lambda n: (not only or n in only) and n not in skip

    print("[benchmark]")
    print(f"  profile       : {config.PROFILE_NAME}")
    print(f"  split         : {args.split}")
    print(f"  GT file       : {gt_file}")
    print(f"  GT questions  : {len(gt)}")
    print(f"  evaluation @  : {K_VALUES}")
    print(f"  device        : {args.device}")

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    all_rows, reranker, reranker_failed = [], None, False
    for exp in exps:
        name = exp["name"]
        if not should_run(name): continue
        print(f"\n-- {name} --")
        try:
            retriever = exp["factory"]()
        except (FileNotFoundError, ValueError, RuntimeError) as e:
            print(f"   SKIPPED: {e}"); continue
        active_reranker = None
        if exp["use_reranker"]:
            if reranker is None and not reranker_failed:
                try:
                    reranker = Reranker(device=args.device)
                except FileNotFoundError as e:
                    reranker_failed = True; print(f"   SKIPPED: reranker unavailable: {e}")
            if reranker is None:
                del retriever; continue
            active_reranker = reranker
        rows = run_experiment(name, gt, retriever, active_reranker)
        all_rows.extend(rows)
        m = pd.DataFrame(rows)[["MRR","Recall@5","Recall@10","nDCG@10","latency_ms"]].mean()
        print(f"   MRR={m['MRR']:.4f} Recall@10={m['Recall@10']:.4f} nDCG@10={m['nDCG@10']:.4f} latency={m['latency_ms']:.2f} ms/q")
        del retriever

    if not all_rows:
        raise SystemExit("No results - all requested experiments were skipped.")

    details = pd.DataFrame(all_rows)
    metric_cols = ["MRR"] + [f"{metric}@{k}" for metric in ("Recall","Precision","nDCG") for k in K_VALUES]
    summary = details.groupby("experiment", sort=False)[metric_cols+["latency_ms"]].mean().round(4).reset_index()
    std = details.groupby("experiment", sort=False)[metric_cols+["latency_ms"]].std().round(4).reset_index()
    by_domain = details.groupby(["experiment","domain"], sort=False)[metric_cols].mean().round(4).reset_index()
    latency = details.groupby("experiment", sort=False)["latency_ms"].agg(["count","mean","median","std","min","max"]).round(4).reset_index()
    run_config = {
        "timestamp": datetime.now().isoformat(timespec="seconds"), "profile": config.PROFILE_NAME,
        "profile_config": json.dumps(config.PROFILE, ensure_ascii=False), "split": args.split,
        "device": args.device, "gt_file": str(gt_file.relative_to(ROOT)) if gt_file.is_relative_to(ROOT) else str(gt_file),
        "gt_questions": len(gt), "gt_sha256": sha256_file(gt_file), "corpus_sha256": corpus_sha256(),
        "eval_k": ",".join(map(str,K_VALUES)), "evaluation_depth": EVAL_DEPTH,
        "production_TOP_K_FINAL_not_used_for_metrics": config.TOP_K_FINAL,
        "dense_min_score_for_benchmark": 0.0, "reranker_model": config.RERANK_MODEL,
    }
    run_cfg_df = pd.DataFrame([{"key":k,"value":v} for k,v in run_config.items()])
    corpus_df = pd.DataFrame([{"key":k,"value":v} for k,v in corpus_stats().items()])
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = out_dir / f"benchmark_{config.PROFILE_NAME}_{args.split}_{ts}.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as writer:
        summary.to_excel(writer, sheet_name="Summary", index=False)
        details.to_excel(writer, sheet_name="Details", index=False)
        by_domain.to_excel(writer, sheet_name="ByDomain", index=False)
        std.to_excel(writer, sheet_name="StdDev", index=False)
        latency.to_excel(writer, sheet_name="Latency", index=False)
        run_cfg_df.to_excel(writer, sheet_name="RunConfig", index=False)
        corpus_df.to_excel(writer, sheet_name="CorpusStats", index=False)
    print(f"\nSaved: {out}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
