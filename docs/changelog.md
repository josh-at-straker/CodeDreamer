# Changelog

- Fixed: Proactive memory now reads full source file for import extraction, not just chunk content which may not include imports. Added INFO-level logging to verify proactive memory is working. (Josh Angel, 2026-01-06)
- Changed: Improved repetition handling - exponential penalty for themes exceeding threshold, 24h time-based decay, project-aware domain terms extracted from indexed codebase (Josh Angel, 2026-01-06)
- Fixed: AttributeError in get_random_graph_context - accessing graph._nodes instead of graph.nodes (Josh Angel, 2026-01-06)
- Added: Theme repetition warnings in dream files showing "Theme 'X': occurrence 2/3" to flag approaching limits (Josh Angel, 2026-01-06)
- Added: DREAMER_CLEAR_INDEX_ON_START option to wipe index and graph on startup, useful when iterating on config (Josh Angel, 2026-01-06)
- Added: Configurable exclude patterns for indexing via DREAMER_EXCLUDE_PATTERNS env var. Default excludes tests/, test_*, *_test.py, conftest.py to focus dreams on production code (Josh Angel, 2026-01-06)
- Changed: Disabled uvicorn access logs to reduce noise in container output (Josh Angel, 2026-01-06)

