# Changelog

All notable changes to CodeDreamer will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

- [Changed]: Optimized token limits for deeper analysis - n_ctx=32K, reasoning=8K, coder=6K, prompt_context=12K, chunk=5K (Indy Nagpal, 2026-01-06)
- [Added]: Ollama-compatible LLM client (`llm_ollama.py`) for Mac/Docker deployments without CUDA (Indy Nagpal, 2026-01-05)
- [Added]: Ollama-compatible model orchestra (`models_ollama.py`) supporting multi-model architecture via Ollama API (Indy Nagpal, 2026-01-05)
- [Added]: Docker setup for Ollama backend with `Dockerfile.ollama` (Indy Nagpal, 2026-01-05)
- [Added]: Docker Compose configuration in `~/Docker/containers/codedreamer/` for local deployment (Indy Nagpal, 2026-01-05)
- [Added]: Documentation structure with `docs/overview.md` and `docs/changelog/changelog.md` (Indy Nagpal, 2026-01-05)
- [Added]: Ollama settings to config.py (`ollama_host`, `ollama_reasoning_model`, etc.) (Indy Nagpal, 2026-01-05)
- [Fixed]: ManagedModel alias added to models_ollama.py for compatibility with __init__.py exports (Indy Nagpal, 2026-01-05)

## [0.1.0] - Initial Release

- Core dream generation engine with 4-level drilling system
- Multi-model architecture (Reasoning, Coder, Embedding)
- Knowledge graph with temporal decay
- TRM (Temporal Recursive Memory) for context persistence
- Proactive memory for anticipated context
- Loop detection and avoidance prompts
- FastAPI server with live dashboard
- CLI for all operations
- ChromaDB-based codebase indexing

