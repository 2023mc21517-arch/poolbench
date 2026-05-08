# PoolBench

**PoolBench** is a diagnostic benchmark that systematically evaluates 19 pooling strategies across 3 large language models and 17 semantic concepts, measuring how well each strategy captures concept-level information in hidden representations.

---

## Why PoolBench?

Every embedding pipeline makes an implicit choice: which token(s) to pool when converting a sequence of hidden states into a single vector. The last token? Mean pooling? Attention-weighted? This choice is rarely studied systematically. PoolBench provides:

- **19 ranked pooling strategies** spanning position-anchored, uniform aggregation, window, saliency-weighted, and structural-linguistic families (18 unsupervised + 1 supervised)
- **17 semantic concepts** (hedging, causation, sentiment, toxicity, legal formality, math certainty, and 12 more) with controlled positive/negative corpora
- **3 evaluation metrics**: D1 AUROC (concept separability), D2 SCP (steered concept prevalence), D3 Disentanglement
- **3 models**: Llama-3.1 8B, Gemma-2 9B, Mistral 7B

---

## Repository layout

```
poolbench/
├── poolbench/                       # Importable Python package
│   ├── __init__.py                  # Public top-level API
│   ├── pooling_strategies.py        # All 19+1 pooling functions + registry
│   ├── utils.py                     # Length helpers, JSONL I/O, record builder
│   ├── extract_activations.py       # GPU activation extraction (hook-based)
│   │
│   ├── data/                        # Subpackage: corpus metadata
│   │   ├── concepts.py              # 17 concept definitions
│   │   ├── filters.py               # Per-concept text / label filters
│   │   └── rewriters.py             # Rule-based positive→negative rewriters
│   │
│   ├── construction/                # Subpackage: direction construction
│   │   └── methods.py               # C1 DifMean · C2 PCA · C3 LogReg · C4 RepE 
│   │
│   ├── evaluation/                  # Subpackage: probing & statistics
│   │   └── probe.py                 # AUROC (5-fold CV + bootstrap CI), Nemenyi, ICC
│   │
│   └── strategies/                  # Subpackage: strategy re-exports
│       └── __init__.py              # Re-exports all strategies + registry
│
├── scripts/
│   ├── dataset_builder.py           # CPU corpus construction (Step 1)
│   ├── power_analysis.py            # Pre-experiment power check (Step 2)
│   └── run_model.py                 # Per-model GPU pipeline (Step 3)
│
├── examples/                        # Community contribution templates
│   ├── README.md                    # How to implement and submit a new strategy
│   ├── my_strategy.py               # Strategy function template
│   └── evaluate.py                  # No-GPU evaluation harness (uses HF activations)
│
├── leaderboard/
│   ├── schema.json                  # JSON schema for valid submissions
│   ├── official/
│   │   └── poolbench_v1.json        # Official paper results
│   └── community/
│       └── _example.json            # Example community submission
│
├── data/
│   ├── corpora/                     # Built by dataset_builder.py  [gitignored]
│   └── raw/                         # HF download cache             [gitignored]
│
├── results/                         # All outputs                   [gitignored]
│   ├── activations/                 # .npy activation files
│   ├── auroc/                       # Per-model AUROC JSON
│   ├── linearity/                   # Linearity check results
│   ├── nemenyi/                     # Statistical significance
│   └── icc/                         # Layer ICC values
│
├── tests/                           # Pytest test suite
├── docs/                            # Extended notes
│
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── .gitignore
```

---

## Quick start

### 1. Install

```bash
git clone https://github.com/2023mc21517-arch/poolbench.git
cd poolbench
pip install -e ".[dev]"
```

### 2. Download the pre-built artifacts (~5 minutes)

All corpus JSONL files, activation `.npy` files, and paper results are released on HuggingFace Hub:

```bash
# Install the HF CLI if needed
pip install huggingface_hub

# Download everything (corpus + activations + results)
huggingface-cli download nips234678/poolbench --repo-type dataset --local-dir .
```

This populates:
```
data/corpora/{concept}/
    train_pos.jsonl   (700 passages)
    train_neg.jsonl   (700 passages)
    test_pos.jsonl    (300 passages)
    test_neg.jsonl    (300 passages)

results/activations/   # per-model .npy activation files
results/auroc/         # paper AUROC JSON
results/scp/           # paper SCP JSON
results/disentanglement/
```

You can also load the corpus directly in Python:

```python
from datasets import load_dataset
ds = load_dataset("nips234678/poolbench", name="hedging")
```

> **Rebuilding from source (optional — for verification or new models):**
> `dataset_builder.py` uses the Llama-3 tokenizer, so you will need a HuggingFace account with the Llama-3 licence accepted:
> ```bash
> huggingface-cli login
> python scripts/dataset_builder.py --all   # ~2–4 hours, CPU only
> ```

### 3. Run experiments on a new model (GPU required)

```bash
# Single model — extracts activations then runs all probes
python scripts/run_model.py --model <your_model_id> --device cuda:0

# Skip extraction if activations already exist
python scripts/run_model.py --model <your_model_id> --skip_extraction

# Nemenyi significance test across all models (run after all models complete)
python scripts/run_model.py --nemenyi_only
```

**VRAM requirements per model:**

| Model | VRAM (bfloat16) |
|---|---|
| Llama-3.1 8B | 16 GB |
| Gemma-2 9B | 20 GB |
| Mistral 7B | 16 GB |

---

## Pooling strategies

| ID | Name | Family | Sup |
|---|---|---|---|
| P1 | Last token | Position-anchored | U |
| P2 | First token (BOS) | Position-anchored | U |
| P3_CLS | CLS token | Position-anchored | U |
| A1 | Mean | Uniform aggregation | U |
| A2 | Max | Uniform aggregation | U |
| A3 | Random 50% (noise floor) | Uniform aggregation | U |
| A4 | Normalised mean (L2 per token) | Uniform aggregation | U |
| W1 | Mean last 4 | Window | U |
| W2 | Mean last 8 | Window | U |
| W3 | Mean last 16 | Window | U |
| W4 | Hierarchical chunks | Window | U |
| S1 | Attention-weighted mean | Saliency-weighted | U |
| S2 | SIF-adapted | Saliency-weighted | U |
| S3 | ITI head pooling (supervised) | Saliency-weighted | **S** |
| L1 | POS-filtered (content words) | Structural-linguistic | U |
| L2 | Dependency-relation filtered | Structural-linguistic | U |
| L3 | Named-entity filtered | Structural-linguistic | U |
| L4 | Subword root only | Structural-linguistic | U |
| L5 | SVO skeleton | Structural-linguistic | U |

Off-leaderboard (oracle / appendix only): `G1_IxG` (requires backward pass).
S = supervised (requires labeled corpus at pooling time); U = unsupervised.

---

## Concepts

| Concept | Positive source | Negative source | Construction |
|---|---|---|---|
| `hedging` | gfissore/arxiv-abstracts-2021 | Same (hedge removed) | Rule-based rewrite |
| `legal_formality` | lex\_glue/scotus | Same (legal markers removed) | Rule-based rewrite |
| `frustration` | go\_emotions | go\_emotions | Label filter |
| `imdb_sentiment` | yin001/imdb_dataset_positive_negative | yin001/imdb_dataset_positive_negative | Label filter |
| `toxicity` | SetFit/toxic\_conversations | SetFit/toxic\_conversations | Label filter |
| `depression` | mrjunos/depression\_reddit\_cleaned + dlb/mentalreddit | mrjunos/depression\_reddit\_cleaned + dlb/mentalreddit | Hybrid Reddit split |
| `causation` | gfissore/arxiv-abstracts-2021 | Same (causal connector removed) | Rule-based rewrite |
| `contrast` | nyu-mll/multi\_nli (contradiction) | nyu-mll/multi\_nli (entailment) | Label filter |
| `conditionality` | nyu-mll/multi\_nli (if/when/unless) | nyu-mll/multi\_nli (no conditionals) | Regex filter |
| `academic_tone` | gfissore/arxiv-abstracts-2021 | sentence-transformers/reddit | Domain filter |
| `code_docs` | Nan-Do/code-search-net-python | sentence-transformers/reddit | Domain filter |
| `bureaucratic` | FiscalNote/billsum | Yelp/yelp\_review\_full | Domain filter |
| `narrative` | euclaise/writingprompts | wikimedia/wikipedia 20231101.en | Domain filter |
| `deference` | Intel/polite-guard | `Intel/polite-guard` | Label filter (`polite` + `somewhat polite` vs. `neutral` + `impolite`) |
| `planning` | gursi26/wikihow-cleaned + sentence-transformers/reddit + Yelp/yelp_review_full | gursi26/wikihow-cleaned + sentence-transformers/reddit + Yelp/yelp_review_full | Text/source filter |
| `negation_density` | gfissore/arxiv-abstracts-2021 | gfissore/arxiv-abstracts-2021 | Lexical filter |
| `numerical_precision` | gfissore/arxiv-abstracts-2021 | cc\_news | Lexical filter |

---

## Construction methods

Four methods for constructing the concept direction vector **d** from activations:

| ID | Name | Notes |
|---|---|---|
| C1 | DifMean | Default. `mean(pos) − mean(neg)`, normalised. |
| C2 | PCA | First PC of [pos; neg]. |
| C3 | LogReg | L2-regularised logistic regression weight vector. |
| C4 | RepE | PCA on per-pair differences `pos_i − neg_i` (Zou et al. 2023). |

---

## Reproducing the paper

Pre-built artifacts (corpus, activations, results) are on HuggingFace — you do **not** need to rebuild the corpus or re-run all 7 models to reproduce the numbers:

```bash
# 1. Download artifacts
huggingface-cli download nips234678/poolbench --repo-type dataset --local-dir .

# 2. Re-run probing only (fast — no GPU extraction needed)
python scripts/run_model.py --all --skip_extraction

# 3. Nemenyi significance test
python scripts/run_model.py --nemenyi_only
```

To reproduce from raw source data (full re-build, ~2–4 hours CPU + GPU time per model):

```bash
# Full reproduction (one GPU, sequential)
bash scripts/reproduce.sh

# Or step by step:
python scripts/dataset_builder.py --all
python scripts/power_analysis.py
for model in llama3_8b gemma2_9b mistral_7b; do
    python scripts/run_model.py --model $model --device cuda:0
done
python scripts/run_model.py --nemenyi_only
```

---

## Running tests

```bash
pytest tests/ -v
```

---

## Leaderboard & submitting results

To **submit a new pooling strategy** to the community leaderboard:

1. Copy `examples/my_strategy.py` and implement your function.
2. Run `examples/evaluate.py` against the pre-extracted activations — **no GPU required**.
3. Fill in `leaderboard/community/_example.json`, rename it to `<your_strategy_id>.json`, and open a PR.

PRs are validated automatically by `.github/workflows/validate_submission.yml` (schema check + strategy ID uniqueness). We merge after a quick sanity check. See `examples/README.md` for step-by-step instructions.

> **Do not include model weights, activation `.npy` files, or corpus JSONL files in your PR** — only the small `leaderboard/community/<id>.json` file.

---

## Citation

```bibtex
@article{poolbench2026,
  title   = {PoolBench: A Benchmark for Token-Pooling Choices in Activation Steering},
  author  = {TODO},
  year    = {2026},
  url     = {https://arxiv.org/abs/XXXX.XXXXX}
}
```

---

## License

CC BY 4.0 — see [LICENSE](LICENSE).
