"""Pluggable sparse TF-IDF + character n-gram vectorizer for offline recall & MMR diversity."""

from __future__ import annotations

import math
from collections import Counter
from typing import Protocol

from ctx_pager.models import MemoryRecord
from ctx_pager.tokenizer import char_ngrams, tokenize_text


class Embedder(Protocol):
    """Protocol allowing drop-in replacement with external dense embedding backends."""

    def embed(self, text: str) -> dict[str, float]:
        ...

    def similarity(self, vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
        ...


def _extract_features(text: str) -> list[str]:
    words = tokenize_text(text, filter_stopwords=True)
    grams = [f"g:{g}" for g in char_ngrams(text, n=3)]
    return words + grams


class SparseTfIdfEmbedder:
    """Zero-dependency L2-normalized sparse TF-IDF embedder over bilingual tokens and char 3-grams."""

    def __init__(self) -> None:
        self._idf: dict[str, float] = {}
        self._doc_count: int = 0

    def fit(self, texts: list[str]) -> None:
        self._doc_count = len(texts)
        df: Counter[str] = Counter()
        for text in texts:
            unique_feats = set(_extract_features(text))
            for feat in unique_feats:
                df[feat] += 1
        n = max(1, self._doc_count)
        self._idf = {
            feat: math.log((n + 1.0) / (count + 1.0)) + 1.0
            for feat, count in df.items()
        }

    def embed(self, text: str) -> dict[str, float]:
        feats = _extract_features(text)
        if not feats:
            return {}
        counts = Counter(feats)
        default_idf = math.log((max(1, self._doc_count) + 1.0) / 1.0) + 1.0
        vec: dict[str, float] = {}
        norm_sq = 0.0
        for feat, tf in counts.items():
            weight = (1.0 + math.log(tf)) * self._idf.get(feat, default_idf)
            vec[feat] = weight
            norm_sq += weight * weight
        if norm_sq <= 0.0:
            return {}
        inv_norm = 1.0 / math.sqrt(norm_sq)
        return {k: v * inv_norm for k, v in vec.items()}

    def similarity(self, vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
        if not vec_a or not vec_b:
            return 0.0
        if len(vec_a) > len(vec_b):
            vec_a, vec_b = vec_b, vec_a
        dot = sum(val * vec_b.get(key, 0.0) for key, val in vec_a.items())
        return max(0.0, min(1.0, dot))


def jaccard_similarity(tokens_a: set[str], tokens_b: set[str]) -> float:
    if not tokens_a or not tokens_b:
        return 0.0
    inter = len(tokens_a & tokens_b)
    if inter == 0:
        return 0.0
    return inter / len(tokens_a | tokens_b)


def record_text(record: MemoryRecord) -> str:
    parts = [record.title, record.summary]
    if record.entities:
        parts.append(" ".join(record.entities))
    if record.detail:
        parts.append(record.detail)
    return "\n".join(parts)


def record_similarity(rec_a: MemoryRecord, rec_b: MemoryRecord) -> float:
    tok_a = set(tokenize_text(f"{rec_a.title} {rec_a.summary}", filter_stopwords=True))
    tok_b = set(tokenize_text(f"{rec_b.title} {rec_b.summary}", filter_stopwords=True))
    return jaccard_similarity(tok_a, tok_b)
