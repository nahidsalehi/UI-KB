"""Central configuration for UI-KB.

Two retrieval-time preprocessing profiles are supported:
  baseline : final paper configuration
  hazm     : legacy pre-freeze Hazm ablation

Select with UIKB_PROFILE or the --profile option in benchmark/index scripts.
Model weights are intentionally not included in the repository; set local paths
with the environment variables listed in .env.example.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
BASE_DIR = ROOT_DIR  # backwards-compatible name used by app.py
CONFIGS_DIR = ROOT_DIR / "configs"
JSON_DIR = ROOT_DIR / "data" / "corpus"
JSONL_PATTERN = "*.jsonl"
GROUND_TRUTH_DIR = ROOT_DIR / "data" / "ground_truth"
TEMPLATES_DIR = ROOT_DIR / "templates"

PROFILE_NAME = ""
PROFILE: dict = {}
INDICES_DIR = ROOT_DIR / "indices" / "baseline"


def set_profile(name: str) -> dict:
    """Activate a preprocessing profile and its matching prebuilt index tree."""
    global PROFILE_NAME, PROFILE, INDICES_DIR
    key = (name or "baseline").strip().lower()
    path = CONFIGS_DIR / f"{key}.json"
    if not path.exists():
        available = ", ".join(sorted(p.stem for p in CONFIGS_DIR.glob("*.json")))
        raise ValueError(f"Unknown profile '{name}'. Available: {available}")
    PROFILE = json.loads(path.read_text(encoding="utf-8"))
    PROFILE_NAME = PROFILE["name"]
    INDICES_DIR = ROOT_DIR / "indices" / PROFILE["indices_subdir"]
    return PROFILE


# Initialise once; CLI entry points may call set_profile() again after parsing.
set_profile(os.environ.get("UIKB_PROFILE", "baseline"))

MODELS = {
    "e5": {
        "path": os.environ.get("E5_MODEL_PATH", os.path.expanduser("~/models/e5-model")),
        "prefix": "query: ",
        "passage_prefix": "passage: ",
        "label": "multilingual-e5-base",
        "dim": 768,
    },
    "bge_m3": {
        "path": os.environ.get("BGE_M3_MODEL_PATH", os.path.expanduser("~/models/bge-m3")),
        "prefix": "query: ",
        "passage_prefix": "passage: ",
        "label": "BGE-M3",
        "dim": 1024,
    },
    "qwen3": {
        "path": os.environ.get("QWEN3_MODEL_PATH", os.path.expanduser("~/models/Qwen3-Embedding-4B")),
        "prefix": "query: ",
        "passage_prefix": "passage: ",
        "label": "Qwen3-Embedding-4B",
        "dim": 2560,
    },
    "matina": {
        "path": os.environ.get("MATINA_MODEL_PATH", os.path.expanduser("~/models/matina-sentence-embedding")),
        "prefix": "",
        "passage_prefix": "",
        "label": "Matina-Sentence-Embedding",
        "dim": 1024,
    },
}

RERANK_MODEL = os.environ.get(
    "RERANK_MODEL_PATH", os.path.expanduser("~/models/bge-reranker-v2-m3")
)

LLM_URL = os.environ.get("LLM_URL", "http://127.0.0.1:11434/api/chat")
LLM_MODEL = os.environ.get("LLM_MODEL", "qwen2.5-14b-fa")
LLM_OPTIONS = {
    "temperature": 0.1,
    "top_p": 0.8,
    "repeat_penalty": 1.2,
    "num_predict": 280,
}

TOP_K_RETRIEVE = int(os.environ.get("TOP_K_RETRIEVE", "10"))
TOP_K_FINAL = int(os.environ.get("TOP_K_FINAL", "5"))
MIN_SCORE = float(os.environ.get("MIN_SCORE", "0.30"))
EMBED_BATCH = int(os.environ.get("EMBED_BATCH", "16"))

APP_EMBED_MODEL = os.environ.get("APP_EMBED_MODEL", "qwen3")
APP_USE_HYBRID = os.environ.get("APP_USE_HYBRID", "true").lower() == "true"
APP_USE_RERANKER = os.environ.get("APP_USE_RERANKER", "true").lower() == "true"
APP_HOST = os.environ.get("APP_HOST", "0.0.0.0")
APP_PORT = int(os.environ.get("APP_PORT", "8000"))

DISCIPLINE_KEYWORDS = [
    "تخلف", "تنبیه", "شورای انضباطی", "تعلیق", "اخراج", "اعتراض", "تقلب",
    "جعل", "تهدید", "رسیدگی", "احضار",
]
ACADEMIC_KEYWORDS = [
    "مرخصی", "مشروطی", "سنوات", "واحد", "رساله", "پایان نامه", "کارشناسی",
    "کارشناسی ارشد", "دکتری", "نمره", "معدل", "تحصیل", "نیمسال",
]
