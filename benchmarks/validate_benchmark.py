"""Preflight validation for UI-KB benchmark inputs and indices."""

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))
import config

GT_SPLITS = {
    "dev": config.GROUND_TRUTH_DIR / "dev_150.json",
    "test": config.GROUND_TRUTH_DIR / "test_60.json",
    "legacy-dev130": config.GROUND_TRUTH_DIR / "legacy_dev_130.json",
}


def load_corpus_ids():
    ids, chunks = set(), 0
    for path in sorted(config.JSON_DIR.glob(config.JSONL_PATTERN)):
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                if not (rec.get("text") or "").strip(): continue
                chunks += 1
                if rec.get("id") is not None: ids.add(str(rec["id"]))
    return ids, chunks


def sha256(path):
    h=hashlib.sha256(); h.update(Path(path).read_bytes()); return h.hexdigest()


def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument("--profile", choices=["baseline","hazm"], default="baseline")
    p.add_argument("--split", choices=list(GT_SPLITS), default="dev")
    p.add_argument("--gt-file", default="")
    p.add_argument("--models", default="")
    p.add_argument("--allow-missing-index", action="store_true")
    return p.parse_args()


def main():
    args=parse_args(); config.set_profile(args.profile)
    gt_file=Path(args.gt_file).resolve() if args.gt_file else GT_SPLITS[args.split]
    gt=json.loads(gt_file.read_text(encoding="utf-8"))
    corpus_ids, corpus_chunks=load_corpus_ids()
    relevant={str(x) for q in gt for x in q.get("relevant_answer",[])}
    missing=sorted(relevant-corpus_ids); errors=[]
    print("[preflight]")
    print(f"  profile         : {config.PROFILE_NAME}")
    print(f"  indices         : {config.INDICES_DIR}")
    print(f"  corpus chunks   : {corpus_chunks}")
    print(f"  GT questions    : {len(gt)}")
    print(f"  GT SHA256       : {sha256(gt_file)}")
    print(f"  missing GT IDs  : {len(missing)}")
    if missing: errors.append(f"{len(missing)} relevant IDs absent from corpus")
    models=[m.strip() for m in args.models.split(",") if m.strip()]
    for key in models:
        print(f"\n[model] {key}")
        if key not in config.MODELS:
            errors.append(f"unknown model key: {key}"); continue
        cfg=config.MODELS[key]; mp=Path(cfg["path"])
        print(f"  label           : {cfg['label']}")
        print(f"  model path      : {mp}")
        print(f"  local model     : {'OK' if mp.exists() else 'MISSING'}")
        if not mp.exists(): errors.append(f"{key}: model path not found: {mp}")
        ip=config.INDICES_DIR/key/"all.faiss"; meta=config.INDICES_DIR/key/"all_meta.json"
        if not ip.exists() or not meta.exists():
            print("  index           : MISSING")
            if not args.allow_missing_index: errors.append(f"{key}: index not built for {config.PROFILE_NAME}")
            continue
        rows=json.loads(meta.read_text(encoding="utf-8")); print(f"  metadata rows   : {len(rows)}")
        if len(rows)!=corpus_chunks: errors.append(f"{key}: metadata rows {len(rows)} != {corpus_chunks}")
        try:
            import faiss
            idx=faiss.read_index(str(ip)); print(f"  FAISS vectors   : {idx.ntotal}"); print(f"  FAISS dim       : {idx.d}")
            if idx.ntotal!=corpus_chunks: errors.append(f"{key}: vectors {idx.ntotal} != {corpus_chunks}")
            if idx.d!=cfg["dim"]: errors.append(f"{key}: dim {idx.d} != {cfg['dim']}")
        except ImportError:
            print("  FAISS check     : skipped (faiss unavailable)")
    print("\n[preflight]")
    if errors:
        print("  status          : FAIL")
        for e in errors: print(f"  - {e}")
        raise SystemExit(1)
    print("  status          : PASS")


if __name__=="__main__": main()
