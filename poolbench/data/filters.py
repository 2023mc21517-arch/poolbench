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


# ── Dense-lexical: frustration (go_emotions labels) ──────────────────────────

_FRUSTRATION_LABELS = {"frustrated", "furious", "annoyed", "anger", "disgust"}
_NEUTRAL_LABELS     = {"excited", "joyful", "proud", "admiration", "joy", "neutral"}


def filter_frustration_positive_label(label: str) -> bool:
    return label.lower() in _FRUSTRATION_LABELS


def filter_frustration_negative_label(label: str) -> bool:
    return label.lower() in _NEUTRAL_LABELS


# ── Dense-lexical: imdb_sentiment (IMDb movie-review labels) ─────────────────

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

def _depression_label_norm(label: str) -> str:
    return re.sub(r"[\s_-]+", " ", str(label).strip().lower())


def filter_depression_positive_label(label: str) -> bool:
    norm = _depression_label_norm(label)
    return norm in {
        "1", "depressed", "depression", "positive", "pos", "yes", "true", "depressive"
    }


def filter_depression_negative_label(label: str) -> bool:
    norm = _depression_label_norm(label)
    return norm in {
        "0", "not depressed", "not depression", "non depressed", "non depression",
        "nondepressed", "nondepression", "notdepressed", "notdepression",
        "control", "negative", "neg", "no", "false",
    }


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
# Vocabulary check ensures positives contain formal academic register markers.

_ACADEMIC_VOCAB_RE = re.compile(
    r'\b(?:analysis|analysi|hypothesis|methodology|methods|findings|results|'
    r'demonstrate|demonstrates|demonstrated|evidence|significant|significance|'
    r'correlation|parameter|parameters|framework|theoretical|empirical|'
    r'quantitative|qualitative|variable|variables|statistical|coefficient|'
    r'variance|investigation|approach|assessment|evaluation|experimental|'
    r'clinical|systematic|review|propose|proposed|novel|algorithm|'
    r'conclude|conclusion|conclusions|furthermore|moreover|however|'
    r'thus|therefore|hence|indicate|indicates|suggest|suggests|'
    r'study|studies|literature|model|models|dataset|corpus|'
    r'accuracy|performance|benchmark|baseline|evaluation|'
    r'equation|theorem|proof|observation|observations)\b',
    re.IGNORECASE,
)


def filter_academic_positive(text: str) -> bool:
    # Require substantive length AND at least 2 academic vocabulary terms
    return len(text.split()) >= 60 and len(_ACADEMIC_VOCAB_RE.findall(text)) >= 2


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


_CODE_DOCS_TECH_KW = re.compile(
    r'\b(python|function|class|method|variable|argument|parameter|return|'
    r'import|module|library|api|code|script|error|exception|bug|debug|'
    r'loop|array|list|dict|string|integer|float|boolean|object|execute|'
    r'syntax|algorithm|database|query|request|response|server|install|'
    r'package|version|dependency|framework|interface|type|value|index|'
    r'object|attribute|key|output|input|runtime|compile|register)\b',
    re.IGNORECASE,
)


def filter_code_docs_negative(text: str) -> bool:
    """
    Negative: informal technical explanation (Stack Overflow / reddit-technical style).
    Requires at least one programming keyword so general Reddit posts are excluded.
    Rejects code-heavy content (>30% code blocks by match count).
    """
    code_ratio = len(re.findall(r"```[\s\S]*?```|<code>[\s\S]*?</code>", text)) / max(1, len(text))
    return (code_ratio < 0.3
            and len(text.split()) >= 40
            and bool(_CODE_DOCS_TECH_KW.search(text)))


# ── Register: bureaucratic (pile-of-law/daily_dialog) ────────────────────────

def filter_bureaucratic_positive(text: str) -> bool:
    lowered = text.lower()
    markers = [
        # Structural references
        "section", "subsection", "chapter", "article", "paragraph",
        "clause", "provision", "title", "subpart", "part",
        # Mandatory/legal language
        "shall", "pursuant", "thereof", "hereby", "therein", "thereto",
        "hereunder", "herein", "heretofore", "notwithstanding",
        "in accordance with", "as set forth", "as provided", "as required",
        # Official actors & instruments
        "the secretary", "the director", "the administrator", "the commissioner",
        "the department", "the agency", "the authority", "the board",
        "the committee", "the office of",
        # Regulatory/procedural phrases
        "regulations", "regulatory", "compliance", "authorized",
        "designated", "promulgate", "administrat", "enforcement",
        "effective date", "upon receipt", "upon request", "federal register",
    ]
    return sum(1 for m in markers if m in lowered) >= 2


def filter_bureaucratic_negative(text: str) -> bool:
    # daily_dialog: accept all turns that are at least 40 words
    return len(text.split()) >= 40


# ── Semantic-abstract: uncertainty ───────────────────────────────────────────

_UNCERTAINTY_MARKERS = [
    "unclear", "uncertain",
    "debated", "no consensus", "unknown", "disputed",
    "controversial", "not yet known", "remains to be",
    "it is not known", "it remains unclear", "little is known",
    "not well understood", "not fully understood", "open question",
    "it is unclear", "is still unknown", "poorly understood",
    "has not been established", "has not been determined",
    # Extended markers for broader coverage
    "remains unclear", "remains unknown", "is not fully understood",
    "is not yet known", "is not well understood", "is yet to be",
    "has yet to be", "has not yet been", "debate continues",
    "is still debated", "is still unclear", "remains an open",
    "no definitive", "inconclusive", "not conclusively",
    "subject to debate", "is poorly understood", "largely unknown",
    "not fully clear", "is still unknown", "insufficient evidence",
    "limited understanding", "not fully elucidated",
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
    # Must have strong certainty signal AND no uncertainty markers (prevents
    # records that use both hedging and certainty language from leaking in)
    return (
        any(m in lowered for m in _CERTAINTY_MARKERS)
        and not any(m in lowered for m in _UNCERTAINTY_MARKERS)
    )


# ── Semantic-abstract: deference (SciCite labels) ────────────────────────────

def filter_deference_positive_label(label: str) -> bool:
    """SciCite: 'background' intent = deference to prior authority."""
    return str(label).lower() == "background"


def filter_deference_negative_label(label: str) -> bool:
    """SciCite: 'result' intent = reporting own findings (no deference)."""
    return str(label).lower() == "result"


# ── Semantic-abstract: deference (Intel/polite-guard labels) ────────────────

def filter_polite_guard_positive_label(label: str) -> bool:
    """Polite-guard: 'polite' and 'somewhat polite' are positive labels."""
    return str(label).strip().lower() in {"polite", "somewhat polite"}


def filter_polite_guard_negative_label(label: str) -> bool:
    """Polite-guard: 'neutral' and 'impolite' are negative labels."""
    return str(label).strip().lower() in {"neutral", "impolite"}


# ── Semantic-abstract: planning (human-authored planning text) ──────────────

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
    """Negative: vague quantifiers, zero numeric tokens of any kind."""
    # Any digit token (including currency, lone digits, quoted numbers) disqualifies
    has_numbers = bool(re.search(r"\b\d", text))
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




# ── Syntactic: conditionality (text-based, for multi-domain rewriting) ───────

_COND_MARKERS = [
    "if ", "unless ", "whenever ", "provided that", "given that",
    "assuming that", "in the event that", "on condition that",
]


def filter_conditionality_positive(text: str) -> bool:
    lowered = text.lower()
    return any(m in lowered for m in _COND_MARKERS)


# ── Semantic-abstract: deference (text-based, for multi-domain) ──────────────

_DEFERENCE_POS_MARKERS = [
    "previous work", "prior work", "prior study", "prior research",
    "as shown by", "as demonstrated by",
    "it has been shown", "it has been established", "it has been found",
    "building on", "following the approach", "based on the work",
    "extending the approach",
]

_DEFERENCE_NEG_MARKERS = [
    "we show", "we demonstrate", "we find", "we present",
    "we propose", "we introduce", "in this paper we",
    "our result", "our finding", "our approach", "our method",
]


def filter_deference_positive(text: str) -> bool:
    lowered = text.lower()
    return any(m in lowered for m in _DEFERENCE_POS_MARKERS)


def filter_deference_negative(text: str) -> bool:
    lowered = text.lower()
    return (
        any(m in lowered for m in _DEFERENCE_NEG_MARKERS) and
        not any(m in lowered for m in _DEFERENCE_POS_MARKERS)
    )


# ── Semantic-abstract: planning (text-based, for multi-domain) ───────────────

_PLANNING_POS_MARKERS = [
    "plan", "plans", "planning", "intend", "intends", "next step",
    "next steps", "goal", "goals", "objective", "objectives",
    "strategy", "strategies", "prepare", "preparing", "how to",
    "step-by-step", "step by step", "going to",
]


def filter_planning_positive(text: str) -> bool:
    lowered = text.lower()
    return any(m in lowered for m in _PLANNING_POS_MARKERS)


def filter_planning_negative(text: str) -> bool:
    lowered = text.lower()
    return not any(m in lowered for m in _PLANNING_POS_MARKERS)


# ── Dense-lexical: frustration (text-based, for multi-domain) ────────────────

_FRUSTRATION_TEXT_RE = re.compile(
    # Stems — match with only a leading word boundary so suffixes like -ed/-ing/-ion work
    r'\b(?:frustrat|infuriat|outrag|enrag|aggravat|exasperat)'
    r'|'
    # Exact single words — require full word boundary on both sides
    r'\b(?:furious|appalling|unacceptable|incompetent|pathetic|atrocious|dreadful|'
    r'disgusting|deplorable|unprofessional|abysmal|worst|terrible|horrible|awful|fed\s+up)\b'
    r'|'
    # Phrases — no word boundary around whole phrase needed
    r'never again|worst experience|terrible service|ripped off|rip.?off'
    r'|sick and tired|waste of (?:time|money)',
    re.IGNORECASE,
)

_NON_FRUSTRATION_TEXT_RE = re.compile(
    r'\b(frustrat|furious|infuriat|outrag|exasperat|terrible|horrible|worst|'
    r'angry|annoy|upset|disappointing|awful|pathetic|atrocious|ridiculous|ugh)\b'
    r'|come\s+on\b|why\s+won.?t\b',
    re.IGNORECASE,
)


def filter_frustration_positive_text(text: str) -> bool:
    return bool(_FRUSTRATION_TEXT_RE.search(text))


def filter_frustration_negative_text(text: str) -> bool:
    return not bool(_NON_FRUSTRATION_TEXT_RE.search(text))


# ── Dense-lexical: imdb_sentiment (text-based, for multi-domain) ─────────────

_POS_SENT_RE = re.compile(
    r'\b(excellent|wonderful|amazing|fantastic|outstanding|love|perfect|'
    r'best|recommend|delicious|friendly|professional|exceptional|superb|'
    r'brilliant|phenomenal|incredible|awesome|favourite|favorite|great)\b',
    re.IGNORECASE,
)

_NEG_SENT_RE = re.compile(
    r'\b(terrible|horrible|awful|worst|bad|disappointing|poor|useless|'
    r'waste|boring|mediocre|dreadful|unpleasant)\b',
    re.IGNORECASE,
)


def filter_imdb_sentiment_positive_text(text: str) -> bool:
    return bool(_POS_SENT_RE.search(text)) and not bool(_NEG_SENT_RE.search(text))


def filter_imdb_sentiment_negative_text(text: str) -> bool:
    # Require ≥2 negative sentiment words so a single incidental "bad" in a
    # neutral news or informational article does not qualify
    return len(_NEG_SENT_RE.findall(text)) >= 2 and not bool(_POS_SENT_RE.search(text))


# ── Dense-lexical: toxicity (text-based, for multi-domain) ───────────────────

_HOSTILE_TEXT_RE = re.compile(
    r'\b(?:'
    # Strong profanity / vulgarity
    r'asshole|assholes|bullshit|bullsh.t|dumbass|jackass|bastard|bitch|bitches|'
    r'motherfucker|f.ck(?:ing|er|ers|ed)?|shit(?:ty|head|hole)?|crap(?:py)?|'
    r'piss(?:ed)?|cunt|screw\s+you|'
    # Classic hostile words
    r'horrible|disgusting|pathetic|atrocious|worthless|'
    r'incompetent|rude|nasty|scam(?:mer)?|fraud|liar|cheated|ripped off|'
    r'disgraceful|unacceptable|abysmal|deplorable|'
    r'unprofessional|absolutely terrible|stay away|'
    # Personal attacks / insults
    r'moron|idiot|stupid(?:ly|est)?|dumbest|imbecile|halfwit|dimwit|'
    r'senile|scumbag|loser|creep|pervert|coward|'
    r'trash|garbage|filth(?:y)?|ignorant|hack(?:[^e]|$)|'
    r'ridiculous(?:ly)?|ugly|stinky|brainwash|sheeple|'
    # Threats / harassment / hate (specific forms only)
    r'hate\s+speech|kill\s+yourself|go\s+die|drop\s+dead|'
    r'white\s+trash|'
    # Bigotry directed at people
    r'racist|sexist|bigot(?:ry)?|homophob(?:ic|e|ia)?|'
    # Drug / harm encouragement directed at a person
    r'smoke\s+crack|do\s+drugs|overdose|'
    # Intelligence / competence attacks
    r'un-american|unamerican|lack(?:s)?\s+(?:the\s+)?intelligence|'
    r'void\s+of\s+(?:intelligence|logic)'
    r')\b'
    r'|'
    # Phrase patterns
    r'rip.?off|sick and tired|piece of (?:shit|trash|garbage|crap)|'
    r'go to hell|shut the f|you\s+people\s+are',
    re.IGNORECASE,
)

# Separate case-sensitive pattern for all-caps aggression (4+ uppercase words)
# Deliberately NOT compiled with IGNORECASE so [A-Z] only matches true uppercase
_HOSTILE_ALLCAPS_RE = re.compile(
    r'(?:[A-Z#@]{3,}[\s#!@:]+){3}[A-Z]{3,}'
)

_POSITIVE_TEXT_RE = re.compile(
    r'\b(excellent|wonderful|amazing|fantastic|outstanding|love|'
    r'perfect|best|recommend|delicious|friendly|professional|exceptional|'
    r'superb|brilliant|phenomenal|incredible|awesome)\b',
    re.IGNORECASE,
)


def filter_toxicity_positive_text(text: str) -> bool:
    return bool(_HOSTILE_TEXT_RE.search(text)) or bool(_HOSTILE_ALLCAPS_RE.search(text))


def filter_toxicity_negative_text(text: str) -> bool:
    return bool(_POSITIVE_TEXT_RE.search(text)) and not bool(_HOSTILE_TEXT_RE.search(text))


# ── Dense-lexical: depression (text-based, for multi-domain) ─────────────────

_DEPRESSION_TEXT_RE = re.compile(
    r'\b(depress(?:ed|ion|ing)?|suicid(?:al|e)?|hopeless(?:ness)?|'
    r'worthless(?:ness)?|despair|emptiness|can\'t go on|'
    r'overwhelming sadness|i feel nothing|no reason to live|'
    r'self.harm|mental health|anxiety|'
    r'i hate myself|so tired of everything)\b',
    re.IGNORECASE,
)

_NON_DEPRESSION_TEXT_RE = re.compile(
    r'\b(depress|suicid|hopeless|despair|self.harm|worthless|'
    r'i feel nothing|hate myself)\b',
    re.IGNORECASE,
)

# Require first-person narrator so news articles, product posts, and
# email-chain forwards (which use third-person or no personal voice) are rejected
_FIRST_PERSON_RE = re.compile(r"\b(I|I'm|I've|I'd|I'll|my|myself|i feel|i am)\b")

# Sentence splitter — split on . ! ? followed by whitespace or end-of-string
_SENT_SPLIT_RE = re.compile(r'(?<=[.!?])\s+')


def _depression_keyword_in_same_sentence(text: str) -> bool:
    """Return True only if at least one sentence contains BOTH a first-person
    pronoun AND a depression keyword.  This rejects news articles where 'I'
    appears in an unrelated quote while depression language appears elsewhere."""
    for sent in _SENT_SPLIT_RE.split(text):
        if _DEPRESSION_TEXT_RE.search(sent) and _FIRST_PERSON_RE.search(sent):
            return True
    return False


def filter_depression_positive_text(text: str) -> bool:
    # Require that the depression keyword and first-person pronoun co-occur in
    # the same sentence — prevents news articles where "I" is a stray quote and
    # the depression term is in a separate reportage sentence.
    return _depression_keyword_in_same_sentence(text)


def filter_depression_negative_text(text: str) -> bool:
    # Must not contain depression keywords AND must be first-person personal text
    # (excludes news articles, product announcements, email chains, etc.)
    return (
        not bool(_NON_DEPRESSION_TEXT_RE.search(text))
        and bool(_FIRST_PERSON_RE.search(text))
    )


def filter_depression_negative_general_text(text: str) -> bool:
    # Broader negative filter for general Reddit comments: keep passages that do
    # not mention explicit depression markers.
    return not bool(_NON_DEPRESSION_TEXT_RE.search(text))


# ── Registry: get filter by (concept, label) ─────────────────────────────────

TEXT_FILTERS: dict[tuple[str, str], Callable[[str], bool]] = {
    ("hedging",             "pos"): filter_hedging_positive,
    ("hedging",             "neg"): filter_hedging_negative,
    ("legal_formality",     "pos"): filter_legal_positive,
    ("legal_formality",     "neg"): filter_legal_negative,
    ("academic_tone",       "pos"): filter_academic_positive,
    ("academic_tone",       "neg"): filter_academic_negative,
    ("code_docs",           "pos"): filter_code_docs_positive,
    ("code_docs",           "neg"): filter_code_docs_negative,
    ("bureaucratic",        "pos"): filter_bureaucratic_positive,
    ("bureaucratic",        "neg"): filter_bureaucratic_negative,
    ("uncertainty",         "pos"): filter_uncertainty_positive,
    ("uncertainty",         "neg"): filter_uncertainty_negative,
    ("causation",           "pos"): filter_causation_positive,
    ("contrast",            "pos"): filter_contrast_positive,
    ("conditionality",      "pos"): filter_conditionality_positive,
    ("deference",           "pos"): filter_deference_positive,
    ("deference",           "neg"): filter_deference_negative,
    ("planning",            "pos"): filter_planning_positive,
    ("planning",            "neg"): filter_planning_negative,
    ("frustration",         "pos"): filter_frustration_positive_text,
    ("frustration",         "neg"): filter_frustration_negative_text,
    ("negation_density",    "pos"): filter_negation_positive,
    ("negation_density",    "neg"): filter_negation_negative,
    ("numerical_precision", "pos"): filter_numerical_positive,
    ("numerical_precision", "neg"): filter_numerical_negative,
    ("imdb_sentiment",      "pos"): filter_imdb_sentiment_positive_text,
    ("imdb_sentiment",      "neg"): filter_imdb_sentiment_negative_text,
    ("toxicity",            "pos"): filter_toxicity_positive_text,
    ("toxicity",            "neg"): filter_toxicity_negative_text,
    ("depression",          "pos"): filter_depression_positive_text,
    ("depression",          "neg"): filter_depression_negative_text,
}


def get_text_filter(concept: str, label: str) -> Callable[[str], bool]:
    """Returns text-based filter, or a permissive default if not registered."""
    return TEXT_FILTERS.get((concept, label), lambda _: True)


# ── Semantic-abstract: narrative (fiction vs factual prose) ──────────────────
# Positives: euclaise/writingprompts story field — user-authored creative fiction.
# Negatives: wikimedia/wikipedia text field — encyclopaedic factual prose.
# No seed-word filter needed: the dataset source is the signal.
# The negative filter rejects Wikipedia stubs and articles about fictional works
# (to avoid factual descriptions of stories leaking in as negatives).

_NARRATIVE_NOISE_RE = re.compile(
    r'\b(?:novel|film|movie|television\s+series|video\s+game|anime|manga|'
    r'fictional|protagonist|storyline|plot|screenplay|episode|season\s+\d)\b',
    re.IGNORECASE,
)

_NARRATIVE_FICTION_MARKERS = re.compile(
    r'\b(?:he\s+said|she\s+said|they\s+said|whispered|shouted|replied|muttered|'
    r'once\s+upon|the\s+next\s+morning|suddenly|he\s+felt|she\s+felt|'
    r'her\s+heart|his\s+heart|looked\s+around|walked\s+(?:into|toward|away)|'
    r'stared\s+(?:at|into)|stepped\s+(?:forward|back|out|in)|'
    r'tears\s+(?:in|filled|streamed))\b',
    re.IGNORECASE,
)


def filter_narrative_positive(text: str) -> bool:
    """
    Accept WritingPrompts story field entries.
    Rejects entries that are purely a prompt title (no actual story body)
    by requiring at least one fiction marker.
    """
    return bool(_NARRATIVE_FICTION_MARKERS.search(text))


def filter_narrative_negative(text: str) -> bool:
    """
    Accept Wikipedia paragraphs as factual/non-narrative negatives.
    Rejects:
    - Articles whose topic IS fiction (novels, films, anime, etc.) — they
      contain story summaries which are narrative in nature.
    - Stubs / very short articles (length check happens upstream via is_valid_length).
    """
    return not bool(_NARRATIVE_NOISE_RE.search(text))
