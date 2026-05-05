"""
src/concepts.py
All 17 PoolBench concepts with metadata.
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
    "frustration": {
        "family": "TBD",
        "needs_matched_pairs": False,
        "seed_words": [
            "ugh", "frustrated", "come on", "ridiculous", "why won't",
        ],
        # "again" and "seriously" removed — too common in neutral CC-News/Yelp text,
        # only generated false-alarm warnings. Remaining words are genuine frustration markers.
        "contamination_markers": ["frustrated", "come on", "ridiculous", "why won't", "ugh"],

        "positive_def": "Passage expresses frustration or exasperation",
        "negative_def": "Passage is neutral in tone — no frustration markers present",
        "hf_source_pos": "Yelp/yelp_review_full (1-star) + sentence-transformers/reddit (angry subs)",
        "hf_source_neg": "Yelp/yelp_review_full (3-star) + cc_news",
        "source_note": (
            "Independent natural sampling. Positives are 1-star Yelp reviews and Reddit posts "
            "containing frustration markers. Negatives are 3-star Yelp reviews (same platform, "
            "neutral tone — controls for review-style topic distribution) and CC-News articles. "
            "Matched-pair rewrites were dropped because stripping 2-3 rare words from a "
            "400-token passage creates near-identical paired texts that inflate probe AUROC "
            "for reasons unrelated to the concept."
        ),
    },

    # ── DENSE-LEXICAL ────────────────────────────────────────────────────
    # Signal distributed across many tokens.
    # INDEPENDENT sampling OK.

    "imdb_sentiment": {
        "family": "TBD",
        "needs_matched_pairs": False,
        "seed_words": [],
        "positive_def": "Positive overall emotional valence in IMDb movie reviews",
        "negative_def": "Negative overall emotional valence in IMDb movie reviews",
        "hf_source": "yin001/imdb_dataset_positive_negative",
        "source_note": (
            "IMDb review sentiment corpus with both positive and negative classes. "
            "HTML break artifacts such as <br /><br /> are stripped during text "
            "normalization."
        ),
    },
    "toxicity": {
        "family": "TBD",
        "needs_matched_pairs": False,
        "seed_words": [],
        "positive_def": "Passage contains harmful, harassing, or offensive language",
        "negative_def": "Neutral passage on the same topic without harmful language",
        "hf_source": "google/civil_comments",
        "adversarial": True,
        # Civil comments are naturally short; relax the 300-token floor for this concept.
        "token_range": [30, 500],
        # civil_comments is a single-source dataset; 1 domain is inherent.
        "min_domains": 1,
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
        "negative_def": "Passage expresses neutral, non-depressive Reddit discussion",
        "hf_source": "mrjunos/depression-reddit-cleaned + dlb/mentalreddit",
        "source_note": (
            "Positive examples come from the depression label in "
            "mrjunos/depression-reddit-cleaned. Negative examples come from general "
            "Reddit comments in dlb/mentalreddit after removing explicit depression "
            "markers. This concept uses a 200-500 token window and is treated as a "
            "one-domain social dataset."
        ),
        "min_domains": 1,
        "token_range": [200, 500],
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
        # "if" appears in ~73 % of all text; exclude it from contamination
        # detection to avoid mass-flagging legitimate rewritten negatives.
        "contamination_markers": ["unless", "provided that", "given that", "on condition that", "in case", "otherwise,"],

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
        "hf_source_neg": "Yelp/yelp_review_full",    # Classifier B anchor; corpus negatives use daily_dialog
    },

    # ── SEMANTIC-ABSTRACT ────────────────────────────────────────────────
    # High-level reasoning property — no stable surface form.
    # INDEPENDENT sampling OK.

    "narrative": {
        "family": "TBD",
        "needs_matched_pairs": False,
        "seed_words": [],
        "positive_def": "Fictional narrative / creative storytelling — characters, events, emotional arcs",
        "negative_def": "Encyclopaedic factual prose — no characters, no story arc, no narrative voice",
        "hf_source_pos": "euclaise/writingprompts (story field)",
        "hf_source_neg": "wikimedia/wikipedia (20231101.en)",
        "source_note": (
            "Positives are user-authored creative fiction from the WritingPrompts subreddit. "
            "Negatives are Wikipedia article paragraphs. Wikipedia articles about fictional works "
            "(novels, films, anime, etc.) are excluded from negatives to avoid story-summary contamination."
        ),
    },
    "deference": {
        "family": "TBD",
        "needs_matched_pairs": False,
        "seed_words": [],
        "positive_def": "Passage is polite or somewhat polite in tone",
        "negative_def": "Passage is neutral or impolite in tone",
        "hf_source": "Intel/polite-guard + allenai/scicite",
        "source_note": (
            "Positive labels: polite + somewhat polite (Intel polite-guard, customer_service domain); "
            "background citation intent (allenai/scicite, academic_citation domain). "
            "Negative labels: neutral + impolite (Intel polite-guard); result citation intent (scicite). "
            "Both encode the same underlying concept — deferring to external authority vs. asserting "
            "one's own position — in different registers. Token window intentionally relaxed to 8-128 "
            "because both sources are sentence-level."
        ),
        "token_range": [8, 128],
        "min_domains": 2,
    },
    "planning": {
        "family": "TBD",
        "needs_matched_pairs": False,
        "seed_words": [
            "plan", "will", "intend", "next step", "going to",
            "goal", "objective", "strategy", "prepare", "how to",
        ],
        # "will" is a future auxiliary (~25 % of any text); "goal" is
        # too common in academic/business writing to be discriminative.
        # Keep the rest as a lightweight filter for non-planning text.
        "contamination_markers": ["plan", "intend", "next step", "going to", "objective", "strategy", "prepare", "how to"],

        "positive_def": "Passage gives goal-directed instructions or planning steps",
        "negative_def": "Passage describes ordinary events or opinions without planning/instructional structure",
        "hf_source": "gursi26/wikihow-cleaned + sentence-transformers/reddit + Yelp/yelp_review_full",
        "source_note": (
            "Positive examples come from the summary field of gursi26/wikihow-cleaned. "
            "Negative examples come from human-generated Reddit and Yelp text, with "
            "planning markers filtered out. This is intended as a human-authored "
            "planning-vs-nonplanning corpus."
        ),
        # Two human-authored negative domains are intentional.
        "min_domains": 2,
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
