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


# ── Math certainty: remove certainty markers ──────────────────────────────────
# Use lighteval/MATH natural pairs (solution steps vs. problem statements) first.
# This rewriter handles any arxiv solution text that lacks a matched problem statement.

_MATH_CERTAINTY_REMOVALS: list[tuple[str, str]] = [
    (r"\btherefore[,]?\s*", ""),
    (r"\bhence[,]?\s*",     ""),
    (r"\bthus[,]?\s*",      ""),
    (r"\bit follows that\s*", ""),
    (r"\bQED\b\.?",          ""),
    (r"\bmust be\b",         "is"),      # "must be" → "is" (removes certainty, keeps grammar)
    (r"\bnecessarily\b",     ""),
    (r"\bcan be proven\b",   "seems"),
    (r"\bwe have shown that\s*", ""),
    (r"\bthis completes the proof\b[.]?", ""),
]


def rewrite_math_certainty(pos_text: str) -> Optional[str]:
    """
    Remove proof-language certainty markers.
    Returns None if no markers found or result collapses.
    """
    found = False
    result = pos_text
    for pattern, replacement in _MATH_CERTAINTY_REMOVALS:
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


# ── Causation: remove causal connectives ─────────────────────────────────────
# AltLex natural pairs preferred. Fallback only.

_CAUSAL_REMOVALS: list[tuple[str, str]] = [
    (r"\bbecause\s+",        ""),
    (r"\bsince\s+",          ""),
    (r"\btherefore[,]?\s*",  ""),
    (r"\bas a result[,]?\s*", ""),
    (r"\bconsequently[,]?\s*", ""),
    (r"\bdue to\b",          ""),
    (r"\bowing to\b",        ""),
    (r"\bleads to\b",        "produces"),
    (r"\bcaused by\b",       "related to"),
    (r"\bresulting in\b",    "with"),
]


def rewrite_causation(pos_text: str) -> Optional[str]:
    found = False
    result = pos_text
    for pattern, replacement in _CAUSAL_REMOVALS:
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


# ── Contrast: remove contrastive connectives ──────────────────────────────────
# DiscoGeM natural pairs preferred. Fallback only.

_CONTRAST_REMOVALS: list[tuple[str, str]] = [
    (r"\bhowever[,]?\s*",            ""),
    (r"\balthough\s+",               ""),
    (r"\bdespite\s+",                ""),
    (r"\bnevertheless[,]?\s*",       ""),
    (r"\bon the other hand[,]?\s*",  ""),
    (r"\bwhereas\s+",                ""),
    (r"\byet[,]?\s+",                ""),
    (r"\bin contrast[,]?\s*",        ""),
]


def rewrite_contrast(pos_text: str) -> Optional[str]:
    found = False
    result = pos_text
    for pattern, replacement in _CONTRAST_REMOVALS:
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


# ── Conditionality: rewrite if-then as direct statement ──────────────────────
# ConjNLI natural pairs strongly preferred; this is a last-resort fallback.

_COND_REMOVALS: list[tuple[str, str]] = [
    (r"\bif\s+",             ""),
    (r"\bunless\s+",         ""),
    (r"\bprovided that\s+",  ""),
    (r"\bon condition that\s+", ""),
    (r"\bassuming\s+",       ""),
    (r"\bgiven that\s+",     ""),
    (r"\bin case\s+",        ""),
    (r"\botherwise[,]?\s*",  ""),
]


def rewrite_conditionality(pos_text: str) -> Optional[str]:
    found = False
    result = pos_text
    for pattern, replacement in _COND_REMOVALS:
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


# ── Registry ──────────────────────────────────────────────────────────────────

REWRITERS: dict[str, callable] = {
    "hedging":        rewrite_hedging,
    "legal_formality": rewrite_legal_formality,
    "math_certainty": rewrite_math_certainty,
    "causation":      rewrite_causation,
    "contrast":       rewrite_contrast,
    "conditionality": rewrite_conditionality,
}


def get_rewriter(concept: str):
    """Returns the rewriter for `concept`, or None if no rewriter exists."""
    return REWRITERS.get(concept, None)
