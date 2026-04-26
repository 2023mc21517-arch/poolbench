# PoolBench

**PoolBench** is a diagnostic benchmark that systematically evaluates 19 pooling strategies across 7 large language models and 18 semantic concepts, measuring how well each strategy captures concept-level information in hidden representations.

> Paper submitted to NeurIPS 2026 Datasets & Benchmarks Track.

---

## Why PoolBench?

Every embedding pipeline makes an implicit choice: which token(s) to pool when converting a sequence of hidden states into a single vector. The last token? Mean pooling? Attention-weighted? This choice is rarely studied systematically. PoolBench provides:

- **19 ranked pooling strategies** spanning position-anchored, uniform aggregation, window, saliency-weighted, and structural-linguistic families
- **18 semantic concepts** (hedging, causation, sentiment, toxicity, legal formality, math certainty, and 12 more) with controlled positive/negative corpora
- **3 evaluation metrics**: D1 AUROC (concept separability), D2 SCP (structural consistency probe), D3 Disentanglement
- **7 models**: Llama-3.1 8B, Gemma-2 9B, Mistral 7B, Qwen-2.5 7B, FLAN-T5 XL, Mamba2 2.7B, BERT-base

---

## Repository layout

```
poolbench/
├── poolbench/                  # Importable Python package
│   ├── concepts.py             # 18 concept definitions (single source of truth)
│   ├── utils.py                # Length helpers, JSONL I/O, record builder
│   ├── filters.py              # Per-concept text / label filters
│   ├── rewriters.py            # Rule-based positive→negative rewriters
│   ├── pooling_strategies.py   # All 19+3 pooling functions + registry
│   ├── construction_methods.py # C1 DifMean · C2 PCA · C3 LogReg · C4 RepE · C5 SAE
│   ├── probe_training.py       # AUROC (5-fold CV + bootstrap CI), Nemenyi, ICC
│   └── extract_activations.py  # GPU activation extraction (hook-based)
│
├── scripts/
│   ├── dataset_builder.py      # CPU corpus construction (Step 1)
│   ├── power_analysis.py       # Pre-experiment power check (Step 2)
│   └── run_model.py            # Per-model GPU pipeline (Step 3)
│
├── data/
│   ├── corpora/                # Built by dataset_builder.py  [gitignored]
│   └── raw/                    # HF download cache             [gitignored]
│
├── results/                    # All outputs                   [gitignored]
│   ├── activations/            # .npy activation files
│   ├── auroc/                  # Per-model AUROC JSON
│   ├── linearity/              # Linearity check results
│   ├── nemenyi/                # Statistical significance
│   └── icc/                    # Layer ICC values
│
├── tests/                      # Pytest test suite
├── docs/                       # Extended notes
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
git clone https://github.com/ayushi-agarwall/poolbench.git
cd poolbench
pip install -e ".[dev]"            # installs poolbench package + dev deps
```

> **HuggingFace access**: `dataset_builder.py` uses the Llama-3 tokenizer for length enforcement. Login once:
> ```bash
> huggingface-cli login
> ```
> You need to accept the Llama-3 model licence at [meta-llama/Meta-Llama-3.1-8B](https://huggingface.co/meta-llama/Meta-Llama-3.1-8B).

### 2. Build the corpus (CPU only, ~2–4 hours total)

Build each concept in sequence. Each one downloads its HuggingFace source dataset, filters, optionally rewrites, and saves `train/` + `test/` JSONL files:

```bash
# One concept at a time (recommended — easier to resume)
python scripts/dataset_builder.py --concept hedging
python scripts/dataset_builder.py --concept causation
# ... repeat for all 18 concepts

# Or build all at once
python scripts/dataset_builder.py --all

# Dry-run first to check counts without saving
python scripts/dataset_builder.py --concept hedging --dry-run
```

Each built concept produces:
```
data/corpora/{concept}/
    train_pos.jsonl   (700 passages)
    train_neg.jsonl   (700 passages)
    test_pos.jsonl    (300 passages)
    test_neg.jsonl    (300 passages)
```

### 3. Power analysis (CPU, ~2 minutes)

Confirm that 300 test passages per class give tight enough AUROC CIs before running GPU experiments:

```bash
python scripts/power_analysis.py
# Pass criterion: 95% CI half-width < 0.025
# If it fails, rebuild with --n_train 1000 --n_test 400
```

### 4. Run experiments (GPU required)

```bash
# Single model
python scripts/run_model.py --model llama3_8b --device cuda:0

# All 7 models (run sequentially on one GPU, or in parallel across GPUs)
python scripts/run_model.py --all --device cuda:0

# Skip activation extraction if already done
python scripts/run_model.py --model llama3_8b --skip_extraction

# Nemenyi significance test across all models (run after all models complete)
python scripts/run_model.py --nemenyi_only
```

**VRAM requirements per model:**

| Model | VRAM (bfloat16) |
|---|---|
| Llama-3.1 8B | 16 GB |
| Gemma-2 9B | 20 GB |
| Mistral 7B | 16 GB |
| Qwen-2.5 7B | 16 GB |
| FLAN-T5 XL | 8 GB |
| Mamba2 2.7B | 8 GB |
| BERT-base | 2 GB |

---

## Pooling strategies

| ID | Name | Family |
|---|---|---|
| P1 | Last token | Position-anchored |
| P2 | First token (CLS/BOS) | Position-anchored |
| A1 | Mean | Uniform aggregation |
| A2 | Sum | Uniform aggregation |
| A3 | Max | Uniform aggregation |
| A4 | Min | Uniform aggregation |
| A5 | Median | Uniform aggregation |
| A6 | Random 50% (noise floor) | Uniform aggregation |
| W1 | Mean last 4 | Window |
| W2 | Mean last 8 | Window |
| W3 | Mean last 16 | Window |
| W4 | Hierarchical chunks | Window |
| S1 | Attention-weighted mean | Saliency-weighted |
| S3 | SIF-adapted | Saliency-weighted |
| S4 | ITI-inspired (attn head proxy) | Saliency-weighted |
| L1 | POS-filtered (content words) | Structural-linguistic |
| L2 | Dependency-relation filtered | Structural-linguistic |
| L3 | Named-entity filtered | Structural-linguistic |
| L4 | Subword root only | Structural-linguistic |

Off-leaderboard (oracle / appendix only): `G1_human_span`, `G2_IxG`, `P3_first_last_concat`.

---

## Concepts

| Concept | Positive source | Negative source | Construction |
|---|---|---|---|
| `hedging` | gfissore/arxiv-abstracts-2021 | Same (hedge removed) | Rule-based rewrite |
| `legal_formality` | lex\_glue/scotus | Same (legal markers removed) | Rule-based rewrite |
| `math_certainty` | EleutherAI/hendrycks\_math | Same (problem stmt) | Natural parallel |
| `frustration` | go\_emotions | go\_emotions | Label filter |
| `pos_sentiment` | sst2 | sst2 | Label filter |
| `toxicity` | SetFit/toxic\_conversations | SetFit/toxic\_conversations | Label filter |
| `depression` | solomonk/reddit\_mental\_health\_posts | solomonk/reddit\_mental\_health\_posts | Label filter |
| `causation` | gfissore/arxiv-abstracts-2021 | Same (causal connector removed) | Rule-based rewrite |
| `contrast` | nyu-mll/multi\_nli (contradiction) | nyu-mll/multi\_nli (entailment) | Label filter |
| `conditionality` | nyu-mll/multi\_nli (if/when/unless) | nyu-mll/multi\_nli (no conditionals) | Regex filter |
| `academic_tone` | gfissore/arxiv-abstracts-2021 | sentence-transformers/reddit | Domain filter |
| `code_docs` | Nan-Do/code-search-net-python | sentence-transformers/reddit | Domain filter |
| `bureaucratic` | FiscalNote/billsum | Yelp/yelp\_review\_full | Domain filter |
| `uncertainty` | gfissore/arxiv-abstracts-2021 | gfissore/arxiv-abstracts-2021 | Lexical filter |
| `deference` | gfissore/arxiv-abstracts-2021 | gfissore/arxiv-abstracts-2021 | Regex filter |
| `planning` | tasksource/bigbench (goal\_step\_wikihow) | tasksource/bigbench | Label filter |
| `negation_density` | gfissore/arxiv-abstracts-2021 | gfissore/arxiv-abstracts-2021 | Lexical filter |
| `numerical_precision` | gfissore/arxiv-abstracts-2021 | cc\_news | Lexical filter |

---

## Construction methods

Five methods for constructing the concept direction vector **d** from activations:

| ID | Name | Notes |
|---|---|---|
| C1 | DifMean | Default. `mean(pos) − mean(neg)`, normalised. |
| C2 | PCA | First PC of [pos; neg]. |
| C3 | LogReg | L2-regularised logistic regression weight vector. |
| C4 | RepE | PCA on per-pair differences `pos_i − neg_i` (Zou et al. 2023). |
| C5 | SAE feature | Top-k SAE decoder columns by activation delta. |

---

## Reproducing the paper

```bash
# Full reproduction (one GPU, sequential)
bash scripts/reproduce.sh      # see scripts/reproduce.sh for the exact command sequence

# Or step by step:
python scripts/dataset_builder.py --all
python scripts/power_analysis.py
for model in llama3_8b gemma2_9b mistral_7b qwen25_7b flan_t5_xl mamba2_2b7 bert_base_uncased; do
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

Results for the 7 paper models are in `results/` once you run the full pipeline. To **submit your own model or pooling strategy** to the community leaderboard:

1. Run `scripts/run_model.py --model <your_model>` (or implement a new strategy following `CONTRIBUTING.md`).
2. Fork this repo, add your results JSON under `results/auroc/` and `results/scp/`.
3. Open a Pull Request using the **Leaderboard Submission** template — fill in the model name, strategy ID, and point to your results files.

We merge PRs after a quick sanity check (format validation, no data leakage). The leaderboard table in this README is updated automatically via the `update-leaderboard` GitHub Action on merge.

> **Do not include model weights, activation `.npy` files, or corpus JSONL files in your PR** — only the small results JSON files.

---

## Citation

```bibtex
@article{poolbench2026,
  title   = {PoolBench: A Diagnostic Benchmark for Pooling Strategy Evaluation in LLM Representations},
  author  = {TODO},
  journal = {NeurIPS Datasets and Benchmarks Track},
  year    = {2026},
  url     = {https://arxiv.org/abs/XXXX.XXXXX}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
