---
dataset_info:
  features:
    - name: id
      dtype: string
    - name: text
      dtype: string
    - name: label
      dtype: int32
    - name: domain
      dtype: string
    - name: token_count
      dtype: int32
    - name: matched_pair_id
      dtype: string
    - name: split
      dtype: string
configs:
  - config_name: default
    data_files:
      - split: train
        path: "data/corpora/*/train_*.jsonl"
      - split: test
        path: "data/corpora/*/test_*.jsonl"
---

# PoolBench

**PoolBench** is a benchmark for evaluating 19 hidden-state pooling strategies for activation steering across 17 text concepts and 7 language models.

Each concept corpus contains up to 1,000 passages per class (700 train / 300 test) sourced from real public datasets on HuggingFace. No passages are LLM-generated.

## Dataset structure

Files are organised as `data/corpora/<concept>/{train,test}_{pos,neg}.jsonl`.

Each JSON line has the following fields:

| Field | Type | Description |
|---|---|---|
| `id` | string | Unique record identifier |
| `text` | string | Passage text (300–500 tokens via LLaMA-3.1-8B tokenizer) |
| `label` | int | 1 = concept present (positive), 0 = concept absent (negative) |
| `domain` | string | Source domain (e.g. `academic`, `news`, `social`) |
| `token_count` | int | Token count (LLaMA-3.1-8B tokenizer) |
| `matched_pair_id` | string | Pair ID for matched-pair concepts; `null` for independently-sampled concepts |
| `split` | string | `train` or `test` |

## Concepts (17)

`academic_tone`, `bureaucratic`, `causation`, `code_docs`, `conditionality`, `contrast`, `deference`, `depression`, `frustration`, `hedging`, `imdb_sentiment`, `legal_formality`, `narrative`, `negation_density`, `numerical_precision`, `planning`, `toxicity`

## Citation

```bibtex
@misc{poolbench2026,
  title={PoolBench: Evaluating Pooling Strategies for Activation Steering Vectors},
  author={Anonymous},
  year={2026},
  note={NeurIPS 2026 Datasets and Benchmarks Track}
}
```
