"""Text normalisation for descriptions, supplier names and codes."""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache

import pandas as pd

# OCR confusions seen inside words (digit wedged between letters, or letter between digits)
_OCR_IN_WORD = [(re.compile(r"(?<=[a-z])0(?=[a-z])"), "o"), (re.compile(r"(?<=[a-z])1(?=[a-z])"), "i"),
                (re.compile(r"(?<=[a-z])5(?=[a-z])"), "s"), (re.compile(r"(?<=[a-z])8(?=[a-z])"), "b"),
                (re.compile(r"^0(?=[a-z]{2})"), "o"), (re.compile(r"^1(?=[a-z]{2})"), "i"),
                (re.compile(r"^5(?=[a-z]{2})"), "s"), (re.compile(r"^8(?=[a-z]{2})"), "b")]



def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c))


_OCR_END = {"0": "o", "1": "i", "5": "s", "8": "b"}


def ocr_fix_word(w: str) -> str:
    for rx, rep in _OCR_IN_WORD:
        w = rx.sub(rep, w)
    if len(w) >= 3 and w[-1] in _OCR_END and w[:-1].isalpha():  # 'tw0' -> 'two' (pure-letter word only)
        w = w[:-1] + _OCR_END[w[-1]]
    return w


@lru_cache(maxsize=500_000)
def clean_text(s: str) -> str:
    """Lowercase, strip accents, fix in-word OCR digits, collapse whitespace. Keeps numbers and units."""
    if s is None or s != s:  # NaN
        return ""
    s = strip_accents(str(s)).lower()
    s = re.sub(r"\[person\]", " ", s)
    toks = re.findall(r"[a-z0-9]+(?:[.,][0-9]+)?", s)
    return " ".join(ocr_fix_word(t) for t in toks)


def compact(s: str) -> str:
    """Alphanumeric-only form used for fuzzy equality (spacing/punctuation-insensitive)."""
    return re.sub(r"[^a-z0-9]", "", clean_text(s))


@lru_cache(maxsize=200_000)
def supplier_norm(name: str) -> str:
    """Canonical supplier string: no legal suffixes, no punctuation/spaces, OCR-fixed."""
    if name is None or name != name:
        return ""
    s = strip_accents(str(name)).lower().replace("&", " and ")
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    toks = [ocr_fix_word(t) for t in s.split()]
    kept = [t for t in toks if t not in _LEGAL_TOKENS]
    return "".join(kept or toks)


_LEGAL_TOKENS = {"pty", "proprietary", "ltd", "limited", "llc", "inc", "incorporated", "corp", "corporation", "co",
                 "company", "cc", "plc", "gmbh", "ag", "sa", "sas", "srl", "spa", "bv", "nv", "ab", "as", "oy", "kg",
                 "kft", "sro", "ltda", "pte", "pvt", "private", "the", "and", "of"}


class Abbreviations:
    """Generic procurement abbreviation expander (loaded from rules/abbreviations.yaml)."""

    def __init__(self, mapping: dict[str, str] | None = None):
        self.map = {k.lower(): v.lower() for k, v in (mapping or {}).items()}

    def expand(self, s: str) -> str:
        toks = re.findall(r"[a-z]+|\d+(?:[.,]\d+)?", clean_text(s))
        return " ".join(self.map.get(t, t) for t in toks)

    def expand_series(self, s: pd.Series) -> pd.Series:
        u = s.fillna("").astype(str).unique()
        m = {x: self.expand(x) for x in u}
        return s.fillna("").astype(str).map(m)


def is_blank_description(s: str) -> bool:
    t = compact(s)
    return t in {"", "noobject", "na", "none", "nil", "product", "products", "parts", "part", "item", "items",
                 "misc", "miscellaneous", "various", "general", "goods", "services", "service", "material",
                 "materials", "invoice", "see attached", "seeattached", "tbc", "tba", "x"} or len(t) <= 1
