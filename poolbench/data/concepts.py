"""
src/concepts.py
All 18 PoolBench concepts with metadata.
Family labels are TBD — assigned post-hoc after results are collected.
"""

CONCEPTS = {

    # ── SPARSE-LEXICAL ───────────────────────────────────────────────────
    # Signal carried by 2–5 specific tokens per passage.
    # These need MATCHED PAIRS.

    "hedging": {
        "family": "TBD",
        "needs_matched_pairs": True,
        "seed_words": [
            "perhaps", "maybe", "might", "possibly", "I think",
            "appears to", "seems", "could be", "arguably", "likely",
        ],
        "positive_def": "Passage contains explicit epistemic hedges that qualify claims",
        "negative_def": "Same passage with hedges removed, claims stated as direct facts",
        "hf_source": "bigbio/bio_scope",   # Natural parallel corpus
    },
    "legal_formality": {
        "family": "TBD",
        "needs_matched_pairs": True,
        "seed_words": [
            "hereby", "pursuant", "notwithstanding", "whereas",
            "therein", "aforementioned", "hereto", "forthwith", "heretofore",
        ],
        "positive_def": "Passage uses legal register markers and formal legal phrasing",
        "negative_def": "Same content rewritten in plain English without legal terms",
        "hf_source": "pile-of-law/pile-of-law",   # Rule-based rewrite
    },
    "math_certainty": {
        "family": "TBD",
        "needs_matched_pairs": True,
        "seed_words": [
            "therefore", "necessarily", "it follows that", "QED",
            "hence", "thus", "must be", "can be proven",
        ],
        "positive_def": "Passage contains mathematical certainty markers (proof language)",
        "negative_def": "Same content without certainty markers, stated tentatively",
        "hf_source": "lighteval/MATH",   # Natural parallel: solution steps vs. problem stmts
    },
    "frustration": {
        "family": "TBD",
        "needs_matched_pairs": True,
        "seed_words": [
            "ugh", "frustrated", "come on", "ridiculous",
            "why won't", "again", "seriously",
        ],
        "positive_def": "Passage expresses frustration or exasperation",
        "negative_def": "Same content expressed neutrally",
        "hf_source": "google-research-datasets/go_emotions",
        "adversarial": True,
    },

    # ── DENSE-LEXICAL ────────────────────────────────────────────────────
    # Signal distributed across many tokens.
    # INDEPENDENT sampling OK.

    "pos_sentiment": {
        "family": "TBD",
        "needs_matched_pairs": False,
        "seed_words": [],
        "positive_def": "Positive overall emotional valence throughout passage",
        "negative_def": "Negative overall emotional valence throughout passage",
        "hf_source": "stanfordnlp/sst2",
    },
    "toxicity": {
        "family": "TBD",
        "needs_matched_pairs": False,
        "seed_words": [],
        "positive_def": "Passage contains harmful, harassing, or offensive language",
        "negative_def": "Neutral passage on the same topic without harmful language",
        "hf_source": "jigsaw_unintended_bias",
        "adversarial": True,
        "ethical_note": (
            "Used to test directional separability only. Outputs scored by frozen "
            "classifier and discarded — not released. Steering vector withheld from "
            "public release."
        ),
    },
    "depression": {
        "family": "TBD",
        "needs_matched_pairs": False,
        "seed_words": [],
        "positive_def": "Passage expresses depressive affect or hopelessness",
        "negative_def": "Passage expresses neutral or positive mental state",
        "hf_source": "vibhorag23/depression_dataset_prepared",
    },

    # ── SYNTACTIC ────────────────────────────────────────────────────────
    # Signal encoded in grammatical structure, not vocabulary.
    # NEEDS MATCHED PAIRS.

    "causation": {
        "family": "TBD",
        "needs_matched_pairs": True,
        "dep_triggers": [
            "because", "since", "therefore", "as a result", "consequently",
            "due to", "owing to", "leads to", "caused by",
        ],
        "seed_words": [
            "because", "since", "therefore", "consequently", "due to",
        ],
        "positive_def": "Passage explicitly encodes a cause-effect relationship",
        "negative_def": "Same content with causal connectives removed/replaced",
        "hf_source": "chridey/altlex",
    },
    "contrast": {
        "family": "TBD",
        "needs_matched_pairs": True,
        "dep_triggers": [
            "but", "however", "although", "while", "yet",
            "on the other hand", "nevertheless", "despite", "whereas",
        ],
        "seed_words": [
            "however", "although", "despite", "nevertheless", "whereas",
        ],
        "positive_def": "Passage explicitly encodes a contrastive or adversative relationship",
        "negative_def": "Same content without contrastive connectives",
        "hf_source": "mainlp/discogem",
    },
    "conditionality": {
        "family": "TBD",
        "needs_matched_pairs": True,
        "dep_triggers": [
            "if", "unless", "provided that", "on condition that",
            "assuming", "given that", "in case", "otherwise",
        ],
        "seed_words": ["if", "unless", "provided that", "assuming", "given that"],
        "positive_def": "Passage explicitly encodes an if-then conditional relationship",
        "negative_def": "Same content with conditional rewritten as a direct statement",
        "hf_source": "cestwc/conj_nli",
    },

    # ── REGISTER ──────────────────────────────────────────────────────────
    # Style signal across full passage.
    # INDEPENDENT sampling OK.

    "academic_tone": {
        "family": "TBD",
        "needs_matched_pairs": False,
        "seed_words": [],
        "positive_def": (
            "Formal academic writing style with hedged claims, citations, passive voice"
        ),
        "negative_def": "Informal writing on the same topic (blog, tweet, conversation)",
        "hf_source_pos": "scientific_papers",
        "hf_source_neg": "sentence-transformers/reddit",
    },
    "code_docs": {
        "family": "TBD",
        "needs_matched_pairs": False,
        "seed_words": [],
        "positive_def": (
            "Formal technical API documentation prose: complete parameter descriptions, "
            "return types, exception behaviours, and usage examples"
        ),
        "negative_def": (
            "Casual technical explanation: Stack Overflow answers, tutorial blog posts"
        ),
        "hf_source": "code_search_net",
        "source_note": (
            "Positives: CodeSearchNet docstrings ≥50 tokens, no inline code. "
            "Negatives: stackoverflow-questions Python How-To answers."
        ),
    },
    "bureaucratic": {
        "family": "TBD",
        "needs_matched_pairs": False,
        "seed_words": [],
        "positive_def": "Bureaucratic or administrative writing style (forms, memos, policies)",
        "negative_def": "Plain conversational version of same content",
        "hf_source_pos": "pile-of-law/pile-of-law",
        "hf_source_neg": "daily_dialog",
    },

    # ── SEMANTIC-ABSTRACT ────────────────────────────────────────────────
    # High-level reasoning property — no stable surface form.
    # INDEPENDENT sampling OK.

    "uncertainty": {
        "family": "TBD",
        "needs_matched_pairs": False,
        "seed_words": [
            "unclear", "uncertain", "may", "it is possible",
            "debated", "no consensus", "unknown",
        ],
        "positive_def": "Epistemic uncertainty expressed — writer does not know the answer",
        "negative_def": "Writer is confident and certain about the same topic",
        "hf_source": "scientific_papers",
    },
    "deference": {
        "family": "TBD",
        "needs_matched_pairs": False,
        "seed_words": [
            "according to", "as noted by", "following", "prior work",
            "as demonstrated by", "established by",
        ],
        "positive_def": "Writer defers to authority, convention, or another person's judgment",
        "negative_def": "Writer asserts their own view confidently without deferring",
        "hf_source": "allenai/scicite",
        "adversarial": True,
    },
    "planning": {
        "family": "TBD",
        "needs_matched_pairs": False,
        "seed_words": [
            "plan", "will", "intend", "next step", "going to",
            "goal", "objective", "strategy",
        ],
        "positive_def": "Passage exhibits future-directed planning and forethought",
        "negative_def": "Passage describes past actions or current state without future planning",
        "hf_source": "tasksource/bigbench",
    },

    # ── SYNTACTIC (deterministic) ─────────────────────────────────────────

    "negation_density": {
        "family": "TBD",
        "needs_matched_pairs": True,
        "seed_words": [],
        "positive_def": (
            "Passage expresses claims primarily through negation "
            "('does not', 'cannot', 'never', 'no evidence that')"
        ),
        "negative_def": "Same information expressed through positive assertion, no negation markers",
        "filter_note": (
            "Positives: ≥3 negation tokens (dep_=='neg' via spaCy). "
            "Negatives: 0 negation tokens."
        ),
        "hf_source": "facebook/multi_nli",
    },

    # ── SPARSE-LEXICAL (deterministic) ───────────────────────────────────

    "numerical_precision": {
        "family": "TBD",
        "needs_matched_pairs": False,
        "seed_words": [],
        "positive_def": (
            "Passage makes precise quantitative claims with specific numbers, "
            "percentages, or measurements"
        ),
        "negative_def": (
            "Passage makes the same claims vaguely "
            "('many', 'most', 'a significant amount') with no specific numbers"
        ),
        "filter_note": (
            "Positives: ≥4 numeric tokens (regex). "
            "Negatives: 0 numeric tokens AND ≥1 vague quantifier."
        ),
        "hf_source_pos": "scientific_papers",
        "hf_source_neg": "cc_news",
    },
}

# The 18 concept names as a stable ordered list (used everywhere for indexing).
CONCEPT_NAMES = list(CONCEPTS.keys())

# Concepts that need matched pairs (rewrite or natural parallel corpus).
MATCHED_PAIR_CONCEPTS = [
    c for c, meta in CONCEPTS.items() if meta.get("needs_matched_pairs", False)
]

# Concepts using fully deterministic filters (no seed words, no rewrites).
DETERMINISTIC_CONCEPTS = ["negation_density", "numerical_precision"]
