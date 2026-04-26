"""
src/filters.py
Per-concept filter functions.
Each function takes a text string and returns True if the passage qualifies.
Deterministic filters (negation_density, numerical_precision) use spaCy + regex.
All other filters use keyword/label-based rules.

spaCy model must be installed:
    python -m spacy download en_core_web_sm
"""

from __future__ import annotations
import re
from functools import lru_cache
from typing import Callable


# ── spaCy lazy loader (only imported when actually needed) ────────────────────

@lru_cache(maxsize=1)
def _nlp():
    import spacy  # noqa: PLC0415
    return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])


# ── Sparse-lexical: hedging ───────────────────────────────────────────────────

_HEDGE_WORDS = [
    "perhaps", "maybe", "might", "possibly", "appears to",
    "seems", "could be", "arguably", "i think", "it seems",
    "it is possible", "one might", "may suggest", "likely",
]


def filter_hedging_positive(text: str) -> bool:
    lowered = text.lower()
    return any(h in lowered for h in _HEDGE_WORDS)


def filter_hedging_negative(text: str) -> bool:
    lowered = text.lower()
    return not any(h in lowered for h in _HEDGE_WORDS)


# ── Sparse-lexical: legal_formality ──────────────────────────────────────────

_LEGAL_MARKERS = [
    "hereby", "pursuant", "notwithstanding", "whereas",
    "therein", "aforementioned", "hereto", "forthwith",
    "heretofore", "hereinafter", "in witness whereof",
]


def filter_legal_positive(text: str) -> bool:
    lowered = text.lower()
    return sum(1 for w in _LEGAL_MARKERS if w in lowered) >= 2


def filter_legal_negative(text: str) -> bool:
    lowered = text.lower()
    # No legal markers AND no formal boilerplate
    return not any(w in lowered for w in _LEGAL_MARKERS)


# ── Sparse-lexical: math_certainty ───────────────────────────────────────────

_MATH_CERTAINTY_MARKERS = [
    "therefore", "hence", "thus", "it follows that",
    "qed", "must be", "necessarily", "can be proven",
    "we have shown", "this completes the proof",
]

# Problem-statement tokens (used for negatives from lighteval/MATH)
_MATH_PROBLEM_TOKENS = ["find", "compute", "determine", "calculate", "prove that", "show that"]


def filter_math_certainty_positive(text: str) -> bool:
    lowered = text.lower()
    return any(m in lowered for m in _MATH_CERTAINTY_MARKERS)


def filter_math_certainty_negative(text: str) -> bool:
    """
    Negative = problem statement (interrogative, no certainty markers).
    Used with lighteval/MATH: problem field is the negative, solution field is the positive.
    """
    lowered = text.lower()
    has_problem_lang = any(p in lowered for p in _MATH_PROBLEM_TOKENS)
    has_certainty = any(m in lowered for m in _MATH_CERTAINTY_MARKERS)
    return has_problem_lang and not has_certainty


# ── Dense-lexical: frustration (go_emotions labels) ──────────────────────────

_FRUSTRATION_LABELS = {"frustrated", "furious", "annoyed", "anger", "disgust"}
_NEUTRAL_LABELS     = {"excited", "joyful", "proud", "admiration", "joy", "neutral"}


def filter_frustration_positive_label(label: str) -> bool:
    return label.lower() in _FRUSTRATION_LABELS


def filter_frustration_negative_label(label: str) -> bool:
    return label.lower() in _NEUTRAL_LABELS


# ── Dense-lexical: pos_sentiment (SST-2 / Yelp labels) ───────────────────────

def filter_sentiment_positive_label(label: int) -> bool:
    """SST-2: label 1 = positive."""
    return int(label) == 1


def filter_sentiment_negative_label(label: int) -> bool:
    """SST-2: label 0 = negative."""
    return int(label) == 0


# ── Dense-lexical: toxicity (Jigsaw) ─────────────────────────────────────────

def filter_toxicity_positive(toxicity_score: float) -> bool:
    return float(toxicity_score) >= 0.5


def filter_toxicity_negative(toxicity_score: float) -> bool:
    return float(toxicity_score) < 0.1


# ── Dense-lexical: depression ────────────────────────────────────────────────

def filter_depression_positive_label(label: str) -> bool:
    return str(label).lower() in {"1", "depressed", "depression"}


def filter_depression_negative_label(label: str) -> bool:
    return str(label).lower() in {"0", "not depressed", "control"}


# ── Syntactic: causation (AltLex labels) ─────────────────────────────────────

_CAUSAL_MARKERS = [
    "because", "since", "therefore", "as a result", "consequently",
    "due to", "owing to", "leads to", "caused by", "resulting in",
]


def filter_causation_positive(text: str) -> bool:
    lowered = text.lower()
    return any(c in lowered for c in _CAUSAL_MARKERS)


def filter_causation_positive_label(label: str) -> bool:
    return str(label).lower() == "causal"


def filter_causation_negative_label(label: str) -> bool:
    return str(label).lower() == "non-causal"


# ── Syntactic: contrast (DiscoGeM labels) ────────────────────────────────────

_CONTRAST_MARKERS = [
    "however", "although", "despite", "nevertheless",
    "on the other hand", "whereas", "yet", " but ", "in contrast",
]


def filter_contrast_positive(text: str) -> bool:
    lowered = text.lower()
    return any(c in lowered for c in _CONTRAST_MARKERS)


def filter_contrast_positive_label(label: str) -> bool:
    return str(label).lower() in {"adversative", "contrast", "concession"}


def filter_contrast_negative_label(label: str) -> bool:
    return str(label).lower() in {"causal", "temporal", "expansion"}


# ── Syntactic: conditionality (ConjNLI labels) ───────────────────────────────

def filter_conditionality_positive_label(label: str) -> bool:
    return str(label).lower() == "contingency"


def filter_conditionality_negative_label(label: str) -> bool:
    return str(label).lower() == "non_contingency"


# ── Register: academic_tone ───────────────────────────────────────────────────
# Positives come from scientific_papers (arxiv) — all qualify.
# Negatives come from reddit — all qualify.
# We still add a light check to exclude very short or code-heavy passages.

def filter_academic_positive(text: str) -> bool:
    return len(text.split()) >= 60   # At least 60 words to have substantive content


def filter_academic_negative(text: str) -> bool:
    # Exclude reddit posts that contain code blocks or URLs
    has_code = bool(re.search(r"```|^\s{4}", text, re.MULTILINE))
    has_url  = bool(re.search(r"https?://", text))
    return not has_code and not has_url and len(text.split()) >= 40


# ── Register: code_docs ───────────────────────────────────────────────────────

def filter_code_docs_positive(docstring: str) -> bool:
    """
    Positive: docstring with ≥50 tokens and no inline code blocks.
    The docstring field in CodeSearchNet is pure prose documentation.
    """
    no_backticks = "`" not in docstring
    long_enough  = len(docstring.split()) >= 50
    return no_backticks and long_enough


def filter_code_docs_negative(text: str) -> bool:
    """
    Negative: Stack Overflow answer body — allow prose explanations,
    exclude if it's mostly code.
    """
    code_ratio = len(re.findall(r"```[\s\S]*?```|<code>[\s\S]*?</code>", text)) / max(1, len(text))
    return code_ratio < 0.3 and len(text.split()) >= 40


# ── Register: bureaucratic (pile-of-law/daily_dialog) ────────────────────────

def filter_bureaucratic_positive(text: str) -> bool:
    lowered = text.lower()
    markers = ["section", "subsection", "chapter", "article", "pursuant",
               "shall", "thereof", "hereby", "the secretary", "the director"]
    return sum(1 for m in markers if m in lowered) >= 2


def filter_bureaucratic_negative(text: str) -> bool:
    # daily_dialog: accept all turns that are at least 40 words
    return len(text.split()) >= 40


# ── Semantic-abstract: uncertainty ───────────────────────────────────────────

_UNCERTAINTY_MARKERS = [
    "unclear", "uncertain", "may", "it is possible",
    "debated", "no consensus", "unknown", "disputed",
    "controversial", "not yet known", "remains to be",
]

_CERTAINTY_MARKERS = [
    "it is clear", "it is well established", "it is known",
    "definitively", "certainly", "without doubt", "proven",
]


def filter_uncertainty_positive(text: str) -> bool:
    lowered = text.lower()
    return any(m in lowered for m in _UNCERTAINTY_MARKERS)


def filter_uncertainty_negative(text: str) -> bool:
    lowered = text.lower()
    has_certainty = any(m in lowered for m in _CERTAINTY_MARKERS)
    has_uncertainty = any(m in lowered for m in _UNCERTAINTY_MARKERS)
    return has_certainty and not has_uncertainty


# ── Semantic-abstract: deference (SciCite labels) ────────────────────────────

def filter_deference_positive_label(label: str) -> bool:
    """SciCite: 'background' intent = deference to prior authority."""
    return str(label).lower() == "background"


def filter_deference_negative_label(label: str) -> bool:
    """SciCite: 'result' intent = reporting own findings (no deference)."""
    return str(label).lower() == "result"


# ── Semantic-abstract: planning (BigBench goal_step_wikihow) ─────────────────

def filter_planning_positive_label(correct: bool) -> bool:
    """Positive = step is correct sub-action for the stated goal."""
    return bool(correct)


def filter_planning_negative_label(correct: bool) -> bool:
    """Negative = plausible step from a different goal."""
    return not bool(correct)


# ── Syntactic (deterministic): negation_density ──────────────────────────────

def filter_negation_positive(text: str) -> bool:
    """Positive: passage has ≥3 syntactic negation markers (spaCy dep_=='neg')."""
    doc = _nlp()(text)
    return sum(1 for tok in doc if tok.dep_ == "neg") >= 3


def filter_negation_negative(text: str) -> bool:
    """Negative: passage has 0 syntactic negation markers."""
    doc = _nlp()(text)
    return sum(1 for tok in doc if tok.dep_ == "neg") == 0


# ── Sparse-lexical (deterministic): numerical_precision ──────────────────────

_NUM_PATTERN    = re.compile(r"\b\d+\.?\d*\s*(%|percent|million|billion|trillion|kg|km|°|pp)?\b")
_VAGUE_QUANTS   = [
    "many", "most", "several", "numerous", "a lot",
    "significant", "substantial", "majority", "minority",
    "few", "some", "countless", "various",
]


def filter_numerical_positive(text: str) -> bool:
    """Positive: ≥4 specific numeric tokens."""
    return len(_NUM_PATTERN.findall(text)) >= 4


def filter_numerical_negative(text: str) -> bool:
    """Negative: vague quantifiers, no specific numbers."""
    has_numbers = len(re.findall(r"\b\d+\.?\d*\b", text)) >= 2
    lowered      = text.lower()
    has_vague    = any(v in lowered for v in _VAGUE_QUANTS)
    return has_vague and not has_numbers


# ── MultiNLI helpers for negation_density ────────────────────────────────────

def filter_multinli_pos_for_negation(hypothesis: str) -> bool:
    """MultiNLI contradiction pair hypothesis — run spaCy dep check."""
    return filter_negation_positive(hypothesis)


def filter_multinli_neg_for_negation(premise: str) -> bool:
    """MultiNLI entailment/neutral premise — zero negation tokens."""
    return filter_negation_negative(premise)


# ── Registry: get filter by (concept, label) ─────────────────────────────────

TEXT_FILTERS: dict[tuple[str, str], Callable[[str], bool]] = {
    ("hedging",             "pos"): filter_hedging_positive,
    ("hedging",             "neg"): filter_hedging_negative,
    ("legal_formality",     "pos"): filter_legal_positive,
    ("legal_formality",     "neg"): filter_legal_negative,
    ("math_certainty",      "pos"): filter_math_certainty_positive,
    ("math_certainty",      "neg"): filter_math_certainty_negative,
    ("academic_tone",       "pos"): filter_academic_positive,
    ("academic_tone",       "neg"): filter_academic_negative,
    ("code_docs",           "pos"): filter_code_docs_positive,
    ("code_docs",           "neg"): filter_code_docs_negative,
    ("bureaucratic",        "pos"): filter_bureaucratic_positive,
    ("bureaucratic",        "neg"): filter_bureaucratic_negative,
    ("uncertainty",         "pos"): filter_uncertainty_positive,
    ("uncertainty",         "neg"): filter_uncertainty_negative,
    ("causation",           "pos"): filter_causation_positive,
    ("negation_density",    "pos"): filter_negation_positive,
    ("negation_density",    "neg"): filter_negation_negative,
    ("numerical_precision", "pos"): filter_numerical_positive,
    ("numerical_precision", "neg"): filter_numerical_negative,
}


def get_text_filter(concept: str, label: str) -> Callable[[str], bool]:
    """Returns text-based filter, or a permissive default if not registered."""
    return TEXT_FILTERS.get((concept, label), lambda _: True)
