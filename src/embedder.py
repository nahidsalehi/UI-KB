"""Build unified FAISS indices for a selected preprocessing profile.

Examples
--------
python src/embedder.py --profile baseline --model e5,bge_m3,qwen3,matina --device cpu
python src/embedder.py --profile hazm --model e5,bge_m3,qwen3 --device cpu
"""

import argparse
import json
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

import config
from preprocessing import dense_text


def load_all_chunks(json_dir: Path) -> tuple[list[str], list[dict]]:
    texts, meta = [], []
    for jsonl_path in sorted(json_dir.glob(config.JSONL_PATTERN)):
        domain = jsonl_path.stem.replace("_v2", "")
        count = 0
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                text = (rec.get("text") or "").strip()
                if not text:
                    continue
                article = rec.get("article")
                section = (rec.get("section") or "").strip()
                body = f"{section}\n{text}" if section else text
                embed_text = f"ماده {article}: {body}" if article else body
                texts.append(embed_text)
                meta.append({
                    "id": rec.get("id"), "domain": domain, "type": rec.get("type"),
                    "text": text, "article": article, "page": rec.get("page"),
                    "degree_level": rec.get("degree_level"),
                    "chunk_index": rec.get("chunk_index"), "term": rec.get("term"),
                    "def_number": rec.get("def_number"), "section": rec.get("section"),
                })
                count += 1
        print(f"    {jsonl_path.name}: {count} chunks")
    return texts, meta


def build_index(model, texts, passage_prefix="", batch_size=config.EMBED_BATCH):
    model_texts = [passage_prefix + dense_text(t) for t in texts]
    embeddings = model.encode(
        model_texts, batch_size=batch_size, show_progress_bar=True,
        normalize_embeddings=True, convert_to_numpy=True,
    ).astype("float32")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def save_index(index, meta, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(out_dir / "all.faiss"))
    (out_dir / "all_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"    Saved -> {out_dir}/all.faiss ({index.ntotal} vectors, dim={index.d})")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--profile", choices=["baseline", "hazm"], default="baseline")
    p.add_argument("--model", default="", help="Comma-separated model keys")
    p.add_argument("--device", default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    config.set_profile(args.profile)
    requested = {m.strip() for m in args.model.split(",") if m.strip()} or set(config.MODELS)
    files = sorted(config.JSON_DIR.glob(config.JSONL_PATTERN))
    if not files:
        raise SystemExit(f"No JSONL files found in {config.JSON_DIR}")
    print(f"Profile : {config.PROFILE_NAME}")
    print(f"Indices : {config.INDICES_DIR}")
    texts, meta = load_all_chunks(config.JSON_DIR)
    print(f"Total   : {len(texts)} chunks")
    for key in sorted(requested):
        if key not in config.MODELS:
            print(f"[{key}] unknown model - skipping")
            continue
        cfg = config.MODELS[key]
        path = Path(cfg["path"])
        if not path.exists():
            print(f"[{key}] model not found at {path} - skipping")
            continue
        print(f"\n[{key}] {cfg['label']}")
        model = SentenceTransformer(str(path), device=args.device, local_files_only=True)
        index = build_index(model, texts, cfg.get("passage_prefix", ""))
        if cfg.get("dim") and index.d != cfg["dim"]:
            raise ValueError(f"{key}: expected dim {cfg['dim']}, got {index.d}")
        save_index(index, meta, config.INDICES_DIR / key)
        del model


if __name__ == "__main__":
    main()
