"""
src/rewriters.py
Rule-based positive → negative rewriters for matched-pair concepts.
Used ONLY as a fallback when no natural parallel corpus is available.
Per the benchmark protocol:
  - NO LLM rewriting.
  - Removal is preferred over substitution (substituting introduces new concept signal).
  - A rewritten negative is discarded if it collapses to fewer than 10 words.
"""

from __future__ import annotations
import re
from typing import Optional


# ── Hedging: remove hedge words ───────────────────────────────────────────────
# Positives sourced from bigbio/bio_scope — natural pairs preferred.
# This rewriter is a fallback if BioScope supply runs short.

_HEDGE_REMOVALS: list[tuple[str, str]] = [
    # (pattern, replacement)  — empty string = simple removal
    (r"\bperhaps\s+", ""),
    (r"\bmaybe\s+", ""),
    (r"\bit seems that\s+", ""),
    (r"\bappears to\s+", ""),
    (r"\bmight\s+", ""),       # do NOT replace with "will" — certainty marker contamination
    (r"\bcould be\s+", ""),    # do NOT replace with "is"
    (r"\bI think\s+", ""),
    (r"\barguably\s+", ""),
    (r"\bpossibly\s+", ""),
    (r"\bseems\s+", ""),
    (r"\blikely\s+", ""),
    (r"\bappear to\s+", ""),
    (r"\bappear\s+to\s+", ""),
]


def rewrite_hedging(pos_text: str) -> Optional[str]:
    """
    Remove hedge words from a positive passage to produce a negative.
    Returns None if no hedge found or the result collapses to <10 words.
    """
    found = False
    result = pos_text
    for pattern, replacement in _HEDGE_REMOVALS:
        new = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        if new != result:
            found = True
            result = new
    if not found:
        return None
    result = re.sub(r" {2,}", " ", result).strip()
    if len(result.split()) < 10:
        return None
    return result


# ── Legal formality: strip legal markers ─────────────────────────────────────
# Fallback only — pile-of-law / MultiLex parallel sources are preferred.

_LEGAL_REPLACEMENTS: list[tuple[str, str]] = [
    (r"\bhereby\b",          ""),
    (r"\bpursuant to\b",     "under"),
    (r"\bnotwithstanding\b", "despite"),
    (r"\bwhereas\b",         "because"),
    (r"\btherein\b",         "there"),
    (r"\baforementioned\b",  "previously mentioned"),
    (r"\bhereto\b",          "to this"),
    (r"\bforthwith\b",       "immediately"),
    (r"\bheretofore\b",      "previously"),
    (r"\bhereinafter\b",     "from now on"),
    (r"\bshall\b",           "will"),
    (r"\bthereof\b",         "of it"),
    (r"\bin witness whereof\b", ""),
]


def rewrite_legal_formality(pos_text: str) -> Optional[str]:
    """
    Replace or remove legal markers to produce a plain-English negative.
    Returns None if no legal markers found or result < 10 words.
    """
    found = False
    result = pos_text
    for pattern, replacement in _LEGAL_REPLACEMENTS:
        new = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
        if new != result:
            found = True
            result = new
    if not found:
        return None
    result = re.sub(r" {2,}", " ", result).strip()
    if len(result.split()) < 10:
        return None
    return result


# ── Causation: strip causal connectives ──────────────────────────────────────
_CAUSAL_RE = re.compile(
    r"\b(because|since|therefore|as a result,?\s*|consequently,?\s*|due to|"
    r"thus,?\s*|hence,?\s*|for this reason,?\s*)\s*",
    re.IGNORECASE,
)

def rewrite_causation(pos_text: str) -> Optional[str]:
    result, n = _CAUSAL_RE.subn("", pos_text)
    if n == 0:
        return None
    result = re.sub(r" {2,}", " ", result).strip()
    return result if len(result.split()) >= 10 else None


# ── Contrast: strip adversative connectives ────────────────────────────────────
_CONTRAST_RE = re.compile(
    r"\b(however,?\s*|but\s+|although\s+|while\s+|yet,?\s*|nevertheless,?\s*|"
    r"on the other hand,?\s*|whereas\s+|even so,?\s*|that said,?\s*)\s*",
    re.IGNORECASE,
)

def rewrite_contrast(pos_text: str) -> Optional[str]:
    result, n = _CONTRAST_RE.subn("", pos_text)
    if n == 0:
        return None
    result = re.sub(r" {2,}", " ", result).strip()
    return result if len(result.split()) >= 10 else None


# ── Conditionality: strip conditional markers ─────────────────────────────────
_COND_RE = re.compile(
    r"\b(if\s+|unless\s+|provided that\s+|on condition that\s+|"
    r"assuming\s+|given that\s+|in the event that\s+|should\s+)\s*",
    re.IGNORECASE,
)

def rewrite_conditionality(pos_text: str) -> Optional[str]:
    result, n = _COND_RE.subn("", pos_text)
    if n == 0:
        return None
    result = re.sub(r" {2,}", " ", result).strip()
    return result if len(result.split()) >= 10 else None


# ── Frustration: strip frustration markers ────────────────────────────────────
_FRUST_RE = re.compile(
    r"\b(ugh[h!]*|frustrated?|come on|ridiculous|why won'?t|again[!?]+|"
    r"seriously[?!]+|unbelievable|for goodness sake|for crying out loud)\b[,!?]?\s*",
    re.IGNORECASE,
)

def rewrite_frustration(pos_text: str) -> Optional[str]:
    result, n = _FRUST_RE.subn("", pos_text)
    if n == 0:
        return None
    result = re.sub(r" {2,}", " ", result).strip()
    return result if len(result.split()) >= 10 else None


# ── Negation density: strip negation tokens ───────────────────────────────────
_NEG_RE = re.compile(
    r"\b(not|no\b|never|neither|nor|without|cannot|n't|isn't|aren't|wasn't|"
    r"weren't|don't|doesn't|didn't|won't|wouldn't|can't|couldn't|shouldn't|"
    r"haven't|hasn't|hadn't)\b",
    re.IGNORECASE,
)

def rewrite_negation_density(pos_text: str) -> Optional[str]:
    result, n = _NEG_RE.subn("", pos_text)
    if n < 3:
        return None
    result = re.sub(r" {2,}", " ", result).strip()
    return result if len(result.split()) >= 10 else None


# ── Registry ──────────────────────────────────────────────────────────────────

REWRITERS: dict[str, callable] = {
    "hedging":           rewrite_hedging,
    "legal_formality":   rewrite_legal_formality,
    "causation":         rewrite_causation,
    "contrast":          rewrite_contrast,
    "conditionality":    rewrite_conditionality,
    "frustration":       rewrite_frustration,
    "negation_density":  rewrite_negation_density,
}


def get_rewriter(concept: str):
    """Returns the rewriter for `concept`, or None if no rewriter exists."""
    return REWRITERS.get(concept, None)
