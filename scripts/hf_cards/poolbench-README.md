---
license: apache-2.0
task_categories:
  - text-classification
language:
  - en
tags:
  - benchmark
  - pooling-strategies
  - representation-learning
  - mechanistic-interpretability
  - probing
  - steering-vectors
pretty_name: PoolBench
size_categories:
  - 10K<n<100K
---

# PoolBench

**PoolBench** is a diagnostic benchmark for evaluating pooling strategies in decoder-only large language models.
Every embedding pipeline implicitly chooses a pooling strategy — last token, mean pooling, attention-weighted, etc. — yet this choice is almost never studied systematically.
PoolBench provides the first controlled, multi-concept, multi-model evaluation framework for this decision.

Submitted to the **NeurIPS 2026 Datasets & Benchmarks Track**.

---

## What is a pooling strategy?

A *pooling strategy* is a function that maps a sequence of per-token hidden states
$\mathbf{H} \in \mathbb{R}^{T \times d}$
to a single fixed-size vector
$\mathbf{v} \in \mathbb{R}^d$.
This vector is then used for classification, similarity search, or steering.
PoolBench asks: *which pooling strategy best captures a given semantic concept in a given model at a given layer?*

---

## Benchmark overview

| Dimension | Value |
|---|---|
| Pooling strategies evaluated | 19 (18 unsupervised + 1 supervised) |
| Semantic concepts | 17 |
| Models | Llama-3.1-8B, Gemma-2-9B, Mistral-7B-v0.3 |
| Evaluation metrics | D1 AUROC · D2 SCP · D3 Disentanglement |
| Corpus size | 1,000 passages/concept (700 train + 300 test, balanced pos/neg) |
| Token window | 300–500 tokens (LLaMA-3.1-8B BPE tokenizer) |
| Total passages | ~33,000 |

---

## The 17 Concepts

Concepts span five linguistic families, chosen to cover a range of signal density and linguistic depth:

| Family | Concepts |
|---|---|
| Sparse-lexical | `hedging`, `legal_formality`, `frustration`, `numerical_precision` |
| Dense-lexical | `imdb_sentiment`, `toxicity`, `depression` |
| Syntactic/discourse | `causation`, `contrast`, `conditionality`, `negation_density` |
| Register | `academic_tone`, `code_docs`, `bureaucratic`, `narrative` |
| Semantic-abstract | `deference`, `planning` |

**Sensitive concepts:** `toxicity` and `depression` are included in all metric computations but their steered text outputs are withheld from public release (available on request for research use).

---

## The 19 Pooling Strategies

| Family | Strategy ID | Description |
|---|---|---|
| Position-anchored | `P1_last_token` | Final non-padding token |
| | `P2_first_token` | First token (BOS position) |
| Uniform aggregation | `A1_mean` | Arithmetic mean over all tokens |
| | `A2_max` | Element-wise max |
| | `A3_min` | Element-wise min |
| | `A4_norm_weighted` | Token-norm weighted mean |
| Window | `W1_first_k` | Mean of first *k* tokens |
| | `W2_last_k` | Mean of last *k* tokens |
| | `W3_middle_k` | Mean of middle *k* tokens |
| | `W4_hierarchical` | Chunk → mean → aggregate |
| Saliency-weighted | `S1_attention` | Attention-score weighted mean |
| | `S2_gradient` | Gradient-norm weighted mean |
| | `S3_iti_probe` | Supervised ITI head-based pooling |
| Structural-linguistic | `L1_pos_filtered` | Content POS tags only (NOUN, VERB, ADJ, ADV) |
| | `L2_dependency_rel` | Dependency-arc triggered tokens |
| | `L3_entity` | Named entity spans |
| | `L4_clause` | Clause-boundary segmentation |
| | `L5_keyword` | TF-IDF top-k keywords |
| | `L6_sentence` | Sentence-boundary segments |

---

## The Three Evaluation Dimensions

### D1 — Concept Separability (AUROC)
Linear probe AUROC measuring how well a pooling strategy separates positive from negative passages for each concept.
Training: 5-fold OOF on 700 train passages per class.
No labels from the test split are used for D1 training.

### D2 — Steered Concept Prevalence (SCP)
Measures whether a pooling strategy's steering vector (DiffMean) actually *steers* generated text toward the target concept.
A Classifier B (fine-tuned BERT) scores each steered generation.
SCP = correlation between alpha (steering strength) and Classifier B score.

### D3 — Disentanglement
Measures whether steering toward concept A avoids contaminating a linguistically-distant (LD) and linguistically-close (LC) neighbour concept.

$$\text{Disent}_c = 1 - \frac{\Delta_B}{\Delta_A}$$

Where $\Delta_A$ = SCP on target concept, $\Delta_B$ = SCP on neighbour concept under the same steering vector.

---

## Dataset Structure

```
poolbench/
  {concept}/
    train_pos.jsonl    # 700 positive training passages
    train_neg.jsonl    # 700 negative training passages
    test_pos.jsonl     # 300 positive test passages
    test_neg.jsonl     # 300 negative test passages
```

Each JSONL record:
```json
{
  "id": "academic_tone_train_pos_0001",
  "text": "...",
  "label": 1,
  "domain": "academic",
  "token_count": 347,
  "matched_pair_id": null,
  "split": "train"
}
```

Fields:
- `id` — unique identifier with concept, split, class, and index
- `text` — passage text (300–500 LLaMA-3.1-8B tokens except `toxicity`/`deference`)
- `label` — 1 = positive, 0 = negative
- `domain` — source domain (e.g., `academic`, `news`, `social`, `legal_us`)
- `token_count` — exact token count under LLaMA-3.1-8B tokenizer
- `matched_pair_id` — non-null for matched-pair concepts (negative is a controlled rewrite of the positive)
- `split` — `train` or `test`

---

## Corpus Construction

Positive and negative passages were constructed under strict controls:

- **700/700 train, 300/300 test per concept** — balanced classes to avoid probe bias
- **300–500 token window** — enforced by LLaMA-3.1-8B BPE tokenizer at build time
- **≥3 source domains per concept** — prevents domain vocabulary from confounding pooling strategy comparisons
- **±25 token matching rule** — for matched-pair concepts, the positive and its negative rewrite differ by ≤25 tokens
- **MD5-based deduplication** — within-class, cross-class, and train/test leak checks
- **Seed-word contamination filter** — negatives must not contain the concept's seed words

See the [GitHub repository](https://github.com/2023mc21517-arch/poolbench) for full construction notes and per-concept source tables.

### Source datasets

| Concept family | Sources |
|---|---|
| Academic/Scientific | `gfissore/arxiv-abstracts-2021`, `qiaojin/PubMedQA` |
| Legal/Government | `lex_glue` (scotus, eurlex), `FiscalNote/billsum` |
| News | `cc_news` |
| Social | `sentence-transformers/reddit`, `Yelp/yelp_review_full` |
| Math | `meta-math/MetaMathQA`, `AI-MO/NuminaMath-CoT` |
| Code | `code_search_net` (python, java, javascript, ruby) |
| Sentiment/Toxicity | `yin001/imdb_dataset_positive_negative`, `tdavidson/hate_speech_offensive`, `google/civil_comments`, Surge-AI toxicity CSV |
| Depression | `mrjunos/depression-reddit-cleaned`, `dlb/mentalreddit` |
| Politeness | `Intel/polite-guard` |
| How-to | `gursi26/wikihow-cleaned` |

---

## Quick Start

```python
from datasets import load_dataset

# Load one concept
ds = load_dataset("nips234678/poolbench", data_dir="academic_tone")

# Load all concepts
from pathlib import Path
concepts = [
    "academic_tone", "bureaucratic", "causation", "code_docs",
    "conditionality", "contrast", "deference", "depression",
    "frustration", "hedging", "imdb_sentiment", "legal_formality",
    "narrative", "negation_density", "numerical_precision",
    "planning", "toxicity",
]
for concept in concepts:
    ds = load_dataset("nips234678/poolbench", data_dir=concept)
```

To reproduce the full pipeline from activations to leaderboard:
```bash
git clone https://github.com/2023mc21517-arch/poolbench.git
cd poolbench
pip install -e ".[dev]"

# Download corpus
huggingface-cli download nips234678/poolbench --repo-type dataset --local-dir data/corpora

# Run full pipeline (requires GPU, ~8h per model on A100)
python scripts/run_model.py --model mistral7b --device cuda:0
```

---

## Related Artifacts

| Artifact | Link | Contents |
|---|---|---|
| BERT Scorer Models (D2) | [nips234678/poolbench-bert-scorers](https://huggingface.co/nips234678/poolbench-bert-scorers) | 17 fine-tuned BERT classifiers (one per concept) used as Classifier B for D2 SCP scoring |
| Activation Files | [nips234678/poolbench-activations](https://huggingface.co/datasets/nips234678/poolbench-activations) | Per-model per-layer hidden states (.npy) for all 3 models — enables D1 evaluation without re-running inference (~390 GB) |
| Steered Outputs | [nips234678/poolbench-steered-outputs](https://huggingface.co/datasets/nips234678/poolbench-steered-outputs) | Generated texts from D2 SCP evaluation for 15 non-sensitive concepts across all 3 models |
| Code & Leaderboard | [GitHub](https://github.com/2023mc21517-arch/poolbench) | Evaluation pipeline, pooling strategy implementations, community submission workflow |

---

## Ethical Considerations

- **Toxicity and depression concepts** include passages containing toxic language and expressions of psychological distress, respectively. These are sourced from publicly available datasets with established research use. Steered text outputs for these two concepts are withheld from public release.
- **No personal information** — all passages are sourced from public corpora with no PII.
- **Model outputs** — steered generation outputs are research artifacts demonstrating steering vector magnitude effects; they are not endorsements of the content.
- **Intended use** — evaluation of pooling strategies in language model representations. Not intended for clinical use, content moderation in production, or as training data for generation models.

---

## Citation

```bibtex
@dataset{poolbench2026,
  title        = {{PoolBench}: A Diagnostic Benchmark for Pooling Strategies in Decoder-Only Language Models},
  author       = {Anonymous},
  year         = {2026},
  publisher    = {HuggingFace},
  url          = {https://huggingface.co/datasets/nips234678/poolbench},
  note         = {Submitted to NeurIPS 2026 Datasets \& Benchmarks Track}
}
```

*Citation will be updated with full author list and DOI upon paper acceptance.*
