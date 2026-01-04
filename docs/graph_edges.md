# Knowledge Graph Edge Creation

This document describes the edge creation strategies in CodeDreamer's knowledge graph.

## Current Implementation

Edges are created automatically when nodes are added via `add_node(auto_connect=True)`.

### 1. Same-File Edges (`relation: "same_file"`)

Connects nodes that reference the same source file.

- **Weight:** 0.8 (strong connection)
- **Max edges:** 5 per node
- **Use case:** Links dreams, facts, and concepts about the same file

```
[DREAM: Add caching to indexer.py] --same_file--> [FACT: indexer.py uses ChromaDB]
```

### 2. Same-Cycle Edges (`relation: "same_cycle"`)

Connects nodes created within a temporal window (default: 5 minutes).

- **Weight:** 0.5 (moderate connection)
- **Max edges:** 5 per node
- **Use case:** Links insights discovered during the same dream cycle

```
[CONCEPT: Error handling pattern] --same_cycle--> [DREAM: Add try-except to validator]
```

## Future Enhancements

### 3. Semantic Similarity Edges

Connects nodes with similar content using text similarity (Jaccard index) or embeddings.

- **Weight:** Similarity score (0.0-1.0)
- **Threshold:** Default 0.6 (only connect if >60% similar)
- **Max edges:** 3 per node

```python
# Connect a single node to similar nodes
graph.connect_semantic("dream_abc123", threshold=0.6)

# Backfill all nodes with semantic edges
graph.backfill_edges(include_semantic=True)
```

**CLI:**
```bash
# Backfill with semantic edges
codedreamer graph backfill --semantic
```

**Implementation:**
- Uses Jaccard word-overlap similarity by default (fast, no model needed)
- Optionally accepts an `embedder` function for embedding-based similarity
- Cosine similarity for embedding vectors

### 4. Concept Co-occurrence Edges (Planned)

Track which concepts appear together frequently.

```python
CONCEPT_PATTERNS = {
    "error_handling": ["try", "except", "raise", "Error"],
    "performance": ["cache", "optimize", "slow", "fast"],
    "security": ["auth", "token", "password", "encrypt"],
}
```

### 5. Dependency Edges (Planned)

Analyze import statements to create `depends_on` edges between file-level entities.

```
[ENTITY: server.py] --depends_on--> [ENTITY: graph.py]
```

## API Reference

### `connect_related_nodes(node_id, max_edges=5, temporal_window_sec=300)`

Creates edges from the specified node to related existing nodes.

**Returns:** Number of edges created.

### `add_node(..., auto_connect=True)`

When `auto_connect=True`, automatically calls `connect_related_nodes` after adding.

## Entanglement Scoring

Entanglement measures how connected a node is to the rest of the graph.
Higher entanglement = more central/valuable idea.

**Formula:** `E = (degree / max_degree) * (1 + avg_weight)`

### CLI Commands

```bash
# Show most entangled nodes
codedreamer graph entangled --limit 10

# Stats now include avg_entanglement
codedreamer graph stats
```

### API

```python
# Get entanglement for a specific node
score = graph.get_entanglement("dream_abc123")

# Get most entangled nodes
top_nodes = graph.get_most_entangled(limit=10)
# Returns: [(node, score), ...]
```

## Graph Query Examples

```python
# Get all nodes connected to a specific node
graph = get_graph()
neighbors = list(graph._graph.neighbors("dream_abc123"))

# Get edge details
edge_data = graph._graph.get_edge_data("dream_abc123", "fact_xyz789")
# {'relation': 'same_file', 'weight': 0.8, ...}

# Find all same-file relationships
same_file_edges = [
    (u, v) for u, v, d in graph._graph.edges(data=True)
    if d.get("relation") == "same_file"
]
```

