"""Bilingual CJK + code-aware tokenization, identifier extraction, and token estimation."""

from __future__ import annotations

import math
import re

_CJK_RUN_RE = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf]+")
_ASCII_WORD_RE = re.compile(r"[A-Za-z0-9_]+")
_CAMEL_SPLIT_RE = re.compile(r"[A-Z]+(?=[A-Z][a-z0-9])|[A-Z]?[a-z]+|[0-9]+|[A-Z]+")

_PATH_RE = re.compile(
    r"(?:(?:\.{0,2}/)?(?:[A-Za-z0-9_.-]+/)+[A-Za-z0-9_.-]+(?::\d+(?::\d+)?)?)"
)
_SCREAMING_RE = re.compile(r"\b[A-Z][A-Z0-9]*(?:_[A-Z0-9]+)+\b")
_SNAKE_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9]*(?:_[a-zA-Z0-9]+)+\b")
_KEBAB_RE = re.compile(r"\b[a-zA-Z][a-zA-Z0-9]*(?:-[a-zA-Z0-9]+)+\b")
_CAMEL_ID_RE = re.compile(r"\b(?:[a-z]+[A-Z][A-Za-z0-9]*|[A-Z][a-z0-9]+[A-Z][A-Za-z0-9]*)\b")
_VERSION_PORT_IP_RE = re.compile(
    r"\b(?:v?\d+\.\d+(?:\.\d+)*(?:-[A-Za-z0-9.]+)?|\d{1,3}(?:\.\d{1,3}){3}(?::\d+)?)\b"
)
_COLON_ADDR_RE = re.compile(r"\b[A-Za-z0-9_.-]+:\d{2,5}\b")
_PORT_CUE_RE = re.compile(r"(?:port|端口|:)\s*(\d{2,5})\b", re.IGNORECASE)
_BACKTICK_RE = re.compile(r"`([^`\n]{2,80})`")
_ALLCAPS_NUM_RE = re.compile(r"\b[A-Z]{2,}[0-9]{2,}\b")

_EN_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "in",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "then",
        "there",
        "these",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
    }
)

_CJK_STOPWORDS = frozenset(
    {
        "的",
        "了",
        "在",
        "是",
        "我",
        "你",
        "他",
        "她",
        "它",
        "们",
        "这",
        "那",
        "个",
        "和",
        "与",
        "或",
        "也",
        "都",
        "就",
        "而",
        "被",
        "把",
        "给",
        "让",
        "用",
        "以",
        "为",
        "之",
        "于",
        "中",
        "上",
        "下",
        "吗",
        "呢",
        "啊",
        "吧",
        "着",
        "过",
        "很",
        "太",
        "更",
    }
)


def _dedup_preserve(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    return out


def extract_identifiers(text: str) -> list[str]:
    """Extract exact technical identifiers (paths, error codes, versions, ports, symbols)."""
    if not text:
        return []

    found: list[str] = []
    patterns = (
        _PATH_RE,
        _SCREAMING_RE,
        _ALLCAPS_NUM_RE,
        _COLON_ADDR_RE,
        _VERSION_PORT_IP_RE,
        _SNAKE_RE,
        _KEBAB_RE,
        _CAMEL_ID_RE,
    )
    for pattern in patterns:
        for match in pattern.finditer(text):
            token = match.group(0).strip(".,;:()[]{}\"'")
            if len(token) >= 2:
                found.append(token.lower())

    for match in _PORT_CUE_RE.finditer(text):
        found.append(match.group(1).lower())

    for match in _BACKTICK_RE.finditer(text):
        inner = match.group(1).strip()
        if inner and " " not in inner and len(inner) <= 64:
            found.append(inner.lower())
        else:
            for sub in _ASCII_WORD_RE.findall(inner):
                if len(sub) >= 2 and ("_" in sub or any(c.isdigit() for c in sub)):
                    found.append(sub.lower())

    return _dedup_preserve(found)


def tokenize_text(text: str, *, filter_stopwords: bool = True) -> list[str]:
    """Tokenize mixed English/code/CJK text into FTS-safe normalized terms.

    - ASCII tokens preserve full snake_case identifiers (for FTS5 `tokenchars '_'`),
      their split sub-tokens (snake/camelCase), and alphanumeric pieces.
    - CJK runs emit overlapping character bigrams plus non-stopword unigrams so both
      compound phrases and single-character technical nouns (e.g. `锁`, `表`) match.
    """
    if not text:
        return []

    tokens: list[str] = []

    for match in _ASCII_WORD_RE.finditer(text):
        raw = match.group(0)
        lower = raw.lower()
        if filter_stopwords and lower in _EN_STOPWORDS:
            continue
        if len(lower) >= 2 or lower.isdigit():
            tokens.append(lower)

        # Split snake_case and camelCase parts so partial code queries hit.
        for seg in raw.split("_"):
            if not seg:
                continue
            seg_lower = seg.lower()
            if (len(seg_lower) >= 2 or seg_lower.isdigit()) and (
                not filter_stopwords or seg_lower not in _EN_STOPWORDS
            ):
                tokens.append(seg_lower)
            parts = _CAMEL_SPLIT_RE.findall(seg)
            if len(parts) > 1:
                for part in parts:
                    p_low = part.lower()
                    if (len(p_low) >= 2 or p_low.isdigit()) and (
                        not filter_stopwords or p_low not in _EN_STOPWORDS
                    ):
                        tokens.append(p_low)

    for match in _CJK_RUN_RE.finditer(text):
        run = match.group(0)
        length = len(run)
        if length == 1:
            if not filter_stopwords or run not in _CJK_STOPWORDS:
                tokens.append(run)
            continue
        for i in range(length - 1):
            tokens.append(run[i : i + 2])
        for ch in run:
            if not filter_stopwords or ch not in _CJK_STOPWORDS:
                tokens.append(ch)

    return tokens


def to_fts_document(text: str) -> str:
    """Convert arbitrary text into a space-delimited token document safe for FTS5."""
    return " ".join(tokenize_text(text, filter_stopwords=True))


def build_fts_query(query: str, max_terms: int = 24) -> str:
    """Build a sanitized FTS5 OR query where every token is quoted to prevent syntax errors."""
    raw_tokens = tokenize_text(query, filter_stopwords=True)
    if not raw_tokens:
        raw_tokens = tokenize_text(query, filter_stopwords=False)

    # Prefer bigrams and longer code tokens first so truncation keeps high-signal terms.
    deduped = _dedup_preserve(raw_tokens)
    prioritized = sorted(
        deduped,
        key=lambda t: (0 if len(t) >= 2 else 1, -min(len(t), 12)),
    )[:max_terms]

    safe_terms: list[str] = []
    for tok in prioritized:
        cleaned = re.sub(r"[^0-9a-z_\u4e00-\u9fff\u3400-\u4dbf]", "", tok)
        if cleaned:
            safe_terms.append(f'"{cleaned}"')

    return " OR ".join(safe_terms)


def char_ngrams(text: str, n: int = 3) -> list[str]:
    """Emit character n-grams over normalized ASCII runs for fuzzy subword similarity."""
    if not text or n <= 0:
        return []
    grams: list[str] = []
    for match in _ASCII_WORD_RE.finditer(text.lower()):
        word = match.group(0)
        if len(word) < n:
            if len(word) >= 2:
                grams.append(word)
            continue
        padded = f"^{word}$"
        for i in range(len(padded) - n + 1):
            grams.append(padded[i : i + n])
    return grams


_ESTIMATE_SCAN_RE = re.compile(
    r"[\u4e00-\u9fff\u3400-\u4dbf]|[A-Za-z0-9_]+|\n+|[^\sA-Za-z0-9_\u4e00-\u9fff\u3400-\u4dbf]+"
)


def estimate_tokens(text: str) -> int:
    """Conservatively estimate LLM tokens for mixed Chinese, English, and code text.

    Designed for small-context budget safety: slightly over-estimates dense CJK and
    punctuation rather than under-counting and risking context-window overflow.
    """
    if not text:
        return 0

    total = 0.0
    for match in _ESTIMATE_SCAN_RE.finditer(text):
        seg = match.group(0)
        first = seg[0]
        if "\u3400" <= first <= "\u9fff":
            total += 1.15
        elif first == "\n":
            total += 0.6 * len(seg)
        elif first.isalnum() or first == "_":
            total += max(1, math.ceil(len(seg) / 4))
        else:
            total += max(0.5, len(seg) * 0.65)

    return max(1, math.ceil(total))
