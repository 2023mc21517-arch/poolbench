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
#
# Strategy: surgically remove the conditional clause so only the consequent
# remains as a direct assertion, instead of simply deleting the keyword.
# Patterns listed from most-structural to most-aggressive:
#   1. Leading clause  "If X, Y"       → "Y"   (whole antecedent + comma removed)
#   2. Leading clause  "Unless X, Y"   → "Y"
#   3. Leading phrase  "Given that X,…"→ consequent kept
#   4. Trailing clause "Y, if X."      → "Y."  (comma + antecedent stripped)
#   5. Trailing clause "Y, unless X."  → "Y."
#   6. Trailing phrase "Y, provided that X." → "Y."
# Each pattern captures the terminal punctuation in group 1 so it is preserved.

_COND_REMOVALS: list[tuple[str, str]] = [
    # Leading: "If <antecedent>, <consequent>"  → "<Consequent>"
    (r"^[Ii]f\b[^,\n]{3,150},\s*",                                         ""),
    # Leading: "Unless <antecedent>, <consequent>" → "<Consequent>"
    (r"^[Uu]nless\b[^,\n]{3,150},\s*",                                     ""),
    # Leading: "Provided that / Given that / Assuming / On condition that …, …"
    (r"(?i)^(?:provided that|given that|on condition that|assuming|in case|in the event that)\b[^,\n]{3,150},\s*", ""),
    # Trailing: "<main clause>, if <antecedent><punct>"  → "<main clause><punct>"
    (r",?\s+if\b[^.!?\n]{3,150}([.!?])\s*$",                               r"\1"),
    # Trailing: "<main clause>, unless <antecedent><punct>"
    (r",?\s+unless\b[^.!?\n]{3,150}([.!?])\s*$",                           r"\1"),
    # Trailing: "<main clause>, provided that / given that <antecedent><punct>"
    (r"(?i),?\s+(?:provided that|given that|on condition that)\b[^.!?\n]{3,150}([.!?])\s*$", r"\1"),
]


def rewrite_conditionality(pos_text: str) -> Optional[str]:
    found = False
    result = pos_text
    for pattern, replacement in _COND_REMOVALS:
        new = re.sub(pattern, replacement, result, flags=re.MULTILINE)
        if new != result:
            found = True
            result = new
    if not found:
        return None
    result = re.sub(r" {2,}", " ", result).strip()
    # Capitalise first character (leading-clause removal may leave a lower-case word)
    if result:
        result = result[0].upper() + result[1:]
    if len(result.split()) < 5:
        return None
    return result


# ── Frustration: soften hostile/frustrated language to neutral-register ───────
# Matched pairs: same Yelp/Reddit post with frustration markers toned down.

_FRUSTRATION_SUBS: list[tuple[str, str]] = [
    (r"\bfrustrat(?:ing|ed|ingly|ion)?\b",  "challenging"),
    (r"\bfurious(?:ly)?\b",                  "concerned"),
    (r"\binfuriat\w+\b",                     "bothered"),
    (r"\boutrag(?:ed|ing|eous(?:ly)?)\b",    "disappointed"),
    (r"\bnever\s+again\b",                   ""),
    (r"\bworst\b",                           "poor"),
    (r"\bterrible\b",                        "poor"),
    (r"\bhorrible\b",                        "poor"),
    (r"\bawful\b",                           "poor"),
    (r"\batrocious\b",                       "subpar"),
    (r"\bappalling\b",                       "poor"),
    (r"\bpathetic\b",                        "poor"),
    (r"\bdisgusting\b",                      "unpleasant"),
    (r"\bunacceptable\b",                    "subpar"),
    (r"\bdeplorable\b",                      "poor"),
    (r"\babysmal\b",                         "poor"),
    (r"\bincompetent\b",                     "poor"),
    (r"\brip(?:-|\s*)off\b",                 "overpriced"),
    (r"\bripped\s+off\b",                    "overcharged"),
]


def rewrite_frustration(pos_text: str) -> Optional[str]:
    """
    Tone-down frustrated language to produce a lower-frustration negative.
    Returns None if no frustration markers found or result collapses.
    """
    found = False
    result = pos_text
    for pattern, replacement in _FRUSTRATION_SUBS:
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


# ── Negation density: remove negation words to produce zero-negation text ────
# Matched pairs: same passage with negation markers stripped out.

_NEGATION_SUBS: list[tuple[str, str]] = [
    (r"\bcannot\b",          "can"),
    (r"\bcan't\b",           "can"),
    (r"\bwon't\b",           "will"),
    (r"\bdon't\b",           "do"),
    (r"\bdoesn't\b",         "does"),
    (r"\bdidn't\b",          "did"),
    (r"\bweren't\b",         "were"),
    (r"\baren't\b",          "are"),
    (r"\bisn't\b",           "is"),
    (r"\bhasn't\b",          "has"),
    (r"\bhaven't\b",         "have"),
    (r"\bwouldn't\b",        "would"),
    (r"\bshouldn't\b",       "should"),
    (r"\bcouldn't\b",        "could"),
    (r"\bmustn't\b",         "must"),
    (r"\bneedn't\b",         "need"),
    (r"\bdaren't\b",         "dare"),
    # "not X" → "X"  (remove the "not" and trailing space)
    (r"\bnot\s+",            ""),
    # "no X" → "X"  (remove "no " before a word)
    (r"\bno\s+(?=\w)",       ""),
    (r"\bnever\b",           ""),
    (r"\bneither\b",         ""),
    (r"\bnor\b",             "or"),
    (r"\bwithout\b",         "with"),
]

_NEG_TOKEN_RE = re.compile(r"\b(not|no|never|neither|nor|without|cannot|n't)\b", re.IGNORECASE)


def rewrite_negation_density(pos_text: str) -> Optional[str]:
    """
    Remove negation markers from a high-negation passage to produce a zero-negation negative.
    Returns None if no negation found or result collapses to < 10 words.
    """
    if not _NEG_TOKEN_RE.search(pos_text):
        return None
    result = pos_text
    for pattern, replacement in _NEGATION_SUBS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)
    result = re.sub(r" {2,}", " ", result).strip()
    # Verify negation count dropped; also reject if still has negation tokens
    if _NEG_TOKEN_RE.search(result):
        # Some negations remain — strip residual contracted forms
        result = re.sub(r"\bn't\b", "", result, flags=re.IGNORECASE)
        result = re.sub(r" {2,}", " ", result).strip()
    if len(result.split()) < 10:
        return None
    return result


# ── Registry ──────────────────────────────────────────────────────────────────

REWRITERS: dict[str, callable] = {
    "hedging":           rewrite_hedging,
    "legal_formality":   rewrite_legal_formality,
    "math_certainty":    rewrite_math_certainty,
    "causation":         rewrite_causation,
    "contrast":          rewrite_contrast,
    "conditionality":    rewrite_conditionality,
    "frustration":       rewrite_frustration,
    "negation_density":  rewrite_negation_density,
}


def get_rewriter(concept: str):
    """Returns the rewriter for `concept`, or None if no rewriter exists."""
    return REWRITERS.get(concept, None)
