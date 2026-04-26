.PHONY: install install-dev lint format test test-cov data-all data-concept clean help

# ── Setup ──────────────────────────────────────────────────────────────────────

install:
	pip install -e .
	python -m spacy download en_core_web_sm

install-dev:
	pip install -e ".[dev]"
	python -m spacy download en_core_web_sm

# ── Quality ────────────────────────────────────────────────────────────────────

lint:
	ruff check poolbench/ scripts/ tests/

format:
	ruff format poolbench/ scripts/ tests/

# ── Tests ──────────────────────────────────────────────────────────────────────

test:
	pytest tests/ -v

test-cov:
	pytest tests/ --cov=poolbench --cov-report=html --cov-report=term-missing

# ── Data pipeline ──────────────────────────────────────────────────────────────

# Build a single concept:  make data-concept CONCEPT=hedging
data-concept:
	python scripts/dataset_builder.py --concept $(CONCEPT)

# Build all 18 concepts (sequential, ~4 hours):
data-all:
	python scripts/dataset_builder.py --all

power:
	python scripts/power_analysis.py

# ── Experiments ────────────────────────────────────────────────────────────────

# Run single model:  make run MODEL=llama3_8b DEVICE=cuda:0
run:
	python scripts/run_model.py --model $(MODEL) --device $(DEVICE)

# Run all models:
run-all:
	python scripts/run_model.py --all --device $(DEVICE)

nemenyi:
	python scripts/run_model.py --nemenyi_only

# Full reproduction:
reproduce:
	bash scripts/reproduce.sh $(DEVICE)

# ── Cleanup ────────────────────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete
	rm -rf .pytest_cache dist build *.egg-info htmlcov .coverage

help:
	@echo ""
	@echo "PoolBench — available make targets:"
	@echo ""
	@echo "  Setup:"
	@echo "    make install          Install package (production)"
	@echo "    make install-dev      Install package + dev/lint tools"
	@echo ""
	@echo "  Quality:"
	@echo "    make lint             Ruff lint check"
	@echo "    make format           Ruff auto-format"
	@echo ""
	@echo "  Tests:"
	@echo "    make test             Run pytest suite"
	@echo "    make test-cov         Run pytest with HTML coverage report"
	@echo ""
	@echo "  Data pipeline:"
	@echo "    make data-concept CONCEPT=hedging   Build one concept corpus"
	@echo "    make data-all                       Build all 18 concept corpora"
	@echo "    make power                          Run pre-experiment power analysis"
	@echo ""
	@echo "  Experiments (requires GPU):"
	@echo "    make run MODEL=llama3_8b DEVICE=cuda:0   Single model"
	@echo "    make run-all DEVICE=cuda:0               All 7 models"
	@echo "    make nemenyi                              Cross-model significance test"
	@echo "    make reproduce DEVICE=cuda:0             Full paper reproduction"
	@echo ""
