PROVIDER ?= ollama
MODEL ?= qwen2.5-coder:7b

.PHONY: install test lint data pairs sandbox-image sandbox-test teacher-pairs

install:
	python -m pip install -e ".[dev]"

test:
	python -m pytest

lint:
	python -m ruff check .

data:
	python -m nlpandas.data.schema_generator --output data/raw --schemas 50 --rows 100

pairs: data
	python -m nlpandas.data.pair_generator --input data/raw --output data/processed/pairs.jsonl

sandbox-image:
	docker build -f docker/Dockerfile.sandbox -t nl-pandas-sandbox:latest .

sandbox-test:
	python -m pytest tests/test_sandbox.py

teacher-pairs: sandbox-image
	python -m nlpandas.data.teacher_pair_generator --provider $(PROVIDER) --model "$(MODEL)" --examples-per-schema 10