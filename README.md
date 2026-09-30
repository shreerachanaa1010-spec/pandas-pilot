<<<<<<< HEAD
# NL-to-pandas Fine-tuning

Fine-tune a small open code model to translate natural-language questions about tabular data into pandas code, then compare it rigorously with prompting baselines. The portfolio story is AI for developer and data tooling: predict, retrieve and reason, then generate and verify.

## Current status

Week 1 foundation is in place: 50 reproducible schemas, 300 verified starter pairs, a Docker-backed pandas sandbox, and Gemini/Ollama teacher adapters. Teacher-generated code is prompted only from train schemas and is accepted only after two matching sandbox results. Live teacher/container execution is unverified here because neither Ollama nor Docker is installed; local tests use provider and sandbox fakes.

The starter pair generator executes only fixed templates authored by this project. Never use it to execute teacher-generated or other untrusted code; route untrusted snippets through the sandbox.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
pre-commit install
docker build -f docker/Dockerfile.sandbox -t nl-pandas-sandbox:latest .
python -m nlpandas.data.schema_generator --output data/raw --schemas 50 --rows 100
python -m nlpandas.data.pair_generator --input data/raw --output data/processed/pairs.jsonl
python -m pytest
```

For a one-schema Ollama smoke run, install/start Ollama, pull `qwen2.5-coder:7b`, then run:

```powershell
python -m nlpandas.data.teacher_pair_generator --provider ollama --schema-limit 1 --examples-per-schema 1
```

For Gemini, set `GEMINI_API_KEY` in the environment and use `--provider gemini`. Do not put API keys in project files or command history. Teacher outputs are written to `data/processed/teacher_pairs.jsonl`; only train-split schemas are used.

Generated tables, verified pairs, and generation reports are git-ignored under `data/`. The table manifest records each schema's split.

Install local experiment tracking tools with `python -m pip install -e ".[tracking]"` when training and evaluation are ready.

See [the design brief](docs/design.md) for success criteria, scope, and risks.
=======
# pandas-pilot
Fine-tuned Qwen2.5-Coder (QLoRA) that turns questions into sandboxed pandas code. CI-gated, quantized, deployed.
>>>>>>> origin/main
