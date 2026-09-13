"""Profile-aware Persian preprocessing.

This module is the single point that differentiates the final baseline and the
legacy Hazm ablation. It intentionally preserves the historical behavior:

baseline
  * dense: no retrieval-time Hazm normalization
  * BM25: lightweight Arabic/Persian character normalization + whitespace split

hazm
  * dense: Hazm Normalizer on query and passage text before encoding
  * BM25: Hazm Normalizer + Persian digit mapping + Hazm word_tokenize
"""

from __future__ import annotations

import re
import config

_PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
_ENGLISH_DIGITS = "0123456789"
_DIGIT_MAP = str.maketrans(_PERSIAN_DIGITS, _ENGLISH_DIGITS)
_HAZM_NORMALIZER = None
_HAZM_WORD_TOKENIZE = None


def light_normalize(text: str) -> str:
    text = (text or "").replace("ي", "ی").replace("ك", "ک")
    text = text.replace("‌", " ")
    return re.sub(r"\s+", " ", text).strip()


def _load_hazm():
    global _HAZM_NORMALIZER, _HAZM_WORD_TOKENIZE
    if _HAZM_NORMALIZER is None or _HAZM_WORD_TOKENIZE is None:
        try:
            from hazm import Normalizer, word_tokenize
        except ImportError as exc:
            raise RuntimeError(
                "The 'hazm' profile requires the hazm package. Install requirements.txt."
            ) from exc
        _HAZM_NORMALIZER = Normalizer()
        _HAZM_WORD_TOKENIZE = word_tokenize
    return _HAZM_NORMALIZER, _HAZM_WORD_TOKENIZE


def dense_text(text: str) -> str:
    """Apply the active retrieval-time normalization used before dense encoding."""
    mode = config.PROFILE.get("dense_text_normalization", "none")
    if mode == "none":
        return text
    if mode == "hazm":
        normalizer, _ = _load_hazm()
        return normalizer.normalize(text)
    raise ValueError(f"Unsupported dense_text_normalization: {mode}")


def sparse_tokens(text: str) -> list[str]:
    mode = config.PROFILE.get("sparse_tokenizer", "light_whitespace")
    if mode == "light_whitespace":
        return light_normalize(text).split()
    if mode == "hazm_word":
        normalizer, word_tokenize = _load_hazm()
        normalized = normalizer.normalize(text).translate(_DIGIT_MAP)
        return word_tokenize(normalized)
    raise ValueError(f"Unsupported sparse_tokenizer: {mode}")
