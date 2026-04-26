# Contributing to PoolBench

Thank you for your interest in contributing! PoolBench is a NeurIPS Datasets & Benchmarks submission — we welcome bug reports, corrections, and additions that make the benchmark more rigorous.

## How to contribute

### Bug reports and questions

Open a [GitHub Issue](https://github.com/ayushi-agarwall/poolbench/issues) with:
- A minimal reproducible example
- The output of `python -c "import poolbench; print(poolbench.__version__)"` 
- Your Python and PyTorch versions

### Adding a pooling strategy

1. Implement your strategy function in `poolbench/pooling_strategies.py` following the existing pattern (accepts `h: np.ndarray` plus optional kwargs, returns `(d_model,)` float32).
2. Register it in `STRATEGY_REGISTRY` with a unique ID (`X1_name` format) and family label.
3. Add a unit test in `tests/test_pooling.py`.
4. If the strategy requires extra inputs (like attention weights), handle the dispatching in `compute_all_pooling_strategies()`.

### Adding a concept

1. Add an entry to `CONCEPTS` in `poolbench/concepts.py` with all required fields.
2. Implement `filter_{concept}_positive` and `filter_{concept}_negative` in `poolbench/filters.py` and register them in `TEXT_FILTERS`.
3. Add a builder function in `scripts/dataset_builder.py` and register it in `CONCEPT_BUILDERS`.
4. If the concept uses matched pairs, implement a rewriter in `poolbench/rewriters.py`.
5. Document the HF source dataset and construction type in `README.md`.

## Development setup

```bash
git clone https://github.com/ayushi-agarwall/poolbench.git
cd poolbench
pip install -e ".[dev]"
python -m spacy download en_core_web_sm
pytest tests/ -v
```

## Code style

We use [Ruff](https://docs.astral.sh/ruff/) for linting:

```bash
pip install ruff
ruff check poolbench/ scripts/
ruff format poolbench/ scripts/
```

## Data policy

- **Do not commit corpus JSONL files** — the pre-built corpus is released on HuggingFace Hub (`ayushi-agarwall/poolbench`). Download it with `huggingface-cli download ayushi-agarwall/poolbench --repo-type dataset --local-dir .`
- **Do not commit activation `.npy` files** — these are also released on HuggingFace Hub and are too large for git.
- **Do not commit model weights or HuggingFace tokens**.

## Pull requests

- Keep PRs focused — one logical change per PR.
- All CI checks must pass (`pytest tests/`).
- Describe the motivation and methodology change in the PR description.
- For changes that affect paper results (new concept, new strategy), include a short results table in the PR description.
