# NL-to-pandas Fine-tuning: Design Brief

## Problem and project question

Data users often know what they want to learn from a table but do not know the pandas operations needed to obtain it. This project will fine-tune Qwen2.5-Coder-1.5B-Instruct with QLoRA to translate natural-language questions into pandas code, execute that code in a constrained sandbox, and compare the result against prompting baselines.

The central question is: **When does fine-tuning beat prompting on correctness, latency, and cost?** The portfolio story is AI for developer and data tooling: predict (Bug-Risk Copilot), retrieve and reason (Sentinel RAG), and generate and verify (this project).

## Scope

In scope: schema-separated synthetic data, verified question/code pairs, a sandbox and adversarial tests, model-agnostic evaluation, zero-/few-shot and larger-model baselines, QLoRA training, focused ablations, one error-analysis fix cycle, MLflow tracking and registry, a CI evaluation gate, GGUF export, one deployed demo, and lightweight input drift checks.

Out of scope: full monitoring dashboards, plotting, and multi-table joins beyond a limited two-table case. Grafana and comprehensive Evidently dashboards are future work.

## Success criteria

- Fine-tuned execution accuracy beats base zero-shot and few-shot baselines on held-out schemas, with 95% confidence intervals reported; inconclusive results are reported honestly.
- DS-1000 pandas performance stays within a pre-agreed regression tolerance.
- The sandbox contains all adversarial tests, and generated code is never executed in the serving process.
- CI blocks unsafe output and a deliberately degraded evaluation candidate.
- The shipped quantized model is re-evaluated, and one diagnose-fix-verify iteration is documented.

## Initial data and evaluation design

Generate 40-60 schemas across several domains with nulls, inconsistent types, dates, and categoricals. Split by schema (70/10/20) so rows from one schema cannot leak across partitions. Retain only examples whose generated code executes successfully and yields stable, non-empty results. Deduplicate by code and semantic similarity, and check contamination against held-out data and DS-1000.

Report execution accuracy with order-sensitive comparison only where question intent requires it, plus validity, unsafe-code rate, format compliance, latency percentiles, and cost per 1,000 queries. Stratify by difficulty and report bootstrap 95% confidence intervals.

## Risks and mitigations

- **Unsafe generated code:** AST screening is not a security boundary by itself. Execute only in an isolated, resource-limited sandbox and keep the API process away from `exec`.
- **Synthetic shortcuts or leakage:** split by schema, keep a hand-verified realism set, run deduplication and contamination checks, and include DS-1000.
- **Training or compute interruptions:** save resumable checkpoints frequently and keep the training configuration and data versions explicit.
- **No measurable fine-tuning gain:** retain all baselines and publish the conditions where prompting is better; do not redefine the test set after inspecting results.
- **Quantization regression or CPU latency:** evaluate the exported GGUF artifact and measure the deployed CPU path rather than extrapolating from GPU results.

## Current implementation boundary

The project has deterministic, distinct schema layouts, a starter corpus of fixed, project-authored templates, a Docker-backed sandbox runner, Gemini/Ollama teacher adapters, and a curation audit. Teacher-generated code is prompted only from train schemas, validated as structured JSON, and sent through the sandbox twice before acceptance. The template generator remains separate and executes only its own trusted templates. The sandbox applies AST preflight and executes accepted snippets in a non-root, network-disabled container with a read-only filesystem and CPU, memory, PID, and timeout limits. Curation removes exact normalized question/code pairs from training when they match held-out or optional benchmark examples; TF-IDF near matches are reported for review, while shared code with different questions is retained. Live teacher/container execution is unverified in environments without Ollama or Docker.