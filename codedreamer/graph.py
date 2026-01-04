"""
Knowledge graph with temporal decay (GitGraph-inspired).

Implements a node/edge graph with:
- Momentum-based relevance scoring
- Three-tier storage (hot/warm/cold)
- Semantic connections between concepts
- Embedding-based similarity edges
"""

import heapq
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum, auto
from functools import lru_cache
from pathlib import Path
from typing import Any

import networkx as nx

from .config import settings

logger = logging.getLogger(__name__)


class NodeType(Enum):
    """Types of knowledge nodes."""

    FACT = auto()  # Extracted fact from code/conversation
    CONCEPT = auto()  # Abstract concept or pattern
    CODE = auto()  # Code snippet or reference
    DREAM = auto()  # Generated improvement idea
    ENTITY = auto()  # Named entity (file, function, class)


class StorageTier(Enum):
    """Memory tiers based on momentum."""

    HOT = auto()  # momentum >= 0.8, frequently accessed
    WARM = auto()  # 0.4 <= momentum < 0.8
    COLD = auto()  # momentum < 0.4, candidate for archival


# Decay constants
DECAY_RATE = 0.05  # Momentum decay per hour
MIN_MOMENTUM = 0.01  # Below this, node is archived
ACTIVATION_BOOST = 0.3  # Boost when node is accessed
MAX_MOMENTUM = 1.0


@dataclass
class KnowledgeNode:
    """A node in the knowledge graph."""

    id: str
    content: str
    node_type: NodeType
    momentum: float = 1.0
    created_at: float = field(default_factory=time.time)
    last_accessed: float = field(default_factory=time.time)
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def tier(self) -> StorageTier:
        """Determine storage tier based on momentum."""
        if self.momentum >= 0.8:
            return StorageTier.HOT
        elif self.momentum >= 0.4:
            return StorageTier.WARM
        return StorageTier.COLD

    @property
    def age_hours(self) -> float:
        """Hours since creation."""
        return (time.time() - self.created_at) / 3600

    def decay(self, hours: float | None = None) -> None:
        """Apply temporal decay to momentum."""
        if hours is None:
            hours = (time.time() - self.last_accessed) / 3600

        decay_amount = DECAY_RATE * hours
        self.momentum = max(MIN_MOMENTUM, self.momentum - decay_amount)

    def activate(self) -> None:
        """Boost momentum when accessed."""
        self.momentum = min(MAX_MOMENTUM, self.momentum + ACTIVATION_BOOST)
        self.last_accessed = time.time()

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "node_type": self.node_type.name,
            "momentum": self.momentum,
            "created_at": self.created_at,
            "last_accessed": self.last_accessed,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeNode":
        """Deserialize from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            node_type=NodeType[data["node_type"]],
            momentum=data.get("momentum", 1.0),
            created_at=data.get("created_at", time.time()),
            last_accessed=data.get("last_accessed", time.time()),
            metadata=data.get("metadata", {}),
        )


@dataclass
class KnowledgeEdge:
    """An edge connecting two knowledge nodes."""

    source_id: str
    target_id: str
    relation: str  # "relates_to", "implements", "improves", "depends_on"
    weight: float = 1.0
    created_at: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation": self.relation,
            "weight": self.weight,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "KnowledgeEdge":
        """Deserialize from dictionary."""
        return cls(
            source_id=data["source_id"],
            target_id=data["target_id"],
            relation=data["relation"],
            weight=data.get("weight", 1.0),
            created_at=data.get("created_at", time.time()),
        )


class KnowledgeGraph:
    """
    Graph-based knowledge storage with temporal decay.

    Implements GitGraph concepts:
    - Nodes represent facts, concepts, code snippets
    - Edges represent semantic relationships
    - Momentum determines relevance (decays over time)
    - Three tiers: hot (active), warm (recent), cold (archival)
    """

    def __init__(self, path: Path | None = None) -> None:
        """
        Initialize or load the knowledge graph.

        Args:
            path: Path to persist graph. Defaults to settings.graph_path.
        """
        self.path = path or settings.graph_path
        self._graph = nx.DiGraph()
        self._nodes: dict[str, KnowledgeNode] = {}
        self._node_counter = 0

        if self.path.exists():
            self._load()

    def add_node(
        self,
        content: str,
        node_type: NodeType,
        node_id: str | None = None,
        metadata: dict[str, Any] | None = None,
        auto_connect: bool = True,
    ) -> KnowledgeNode:
        """
        Add a knowledge node to the graph.

        Args:
            content: The knowledge content.
            node_type: Type of knowledge.
            node_id: Optional explicit ID. Auto-generated if None.
            metadata: Optional additional metadata.
            auto_connect: If True, automatically create edges to related nodes.

        Returns:
            The created node.
        """
        if node_id is None:
            # Use UUID for unique IDs across instances (per dream_20251229_164337)
            node_id = f"{node_type.name.lower()}_{uuid.uuid4().hex[:8]}"

        node = KnowledgeNode(
            id=node_id,
            content=content,
            node_type=node_type,
            metadata=metadata or {},
        )

        self._nodes[node_id] = node
        self._graph.add_node(node_id, **node.to_dict())

        logger.debug(f"Added node: {node_id} ({node_type.name})")

        # Auto-connect to related nodes
        if auto_connect:
            self.connect_related_nodes(node_id)

        return node

    def add_edge(
        self,
        source_id: str,
        target_id: str,
        relation: str,
        weight: float = 1.0,
    ) -> KnowledgeEdge | None:
        """
        Add an edge between two nodes.

        Args:
            source_id: Source node ID.
            target_id: Target node ID.
            relation: Type of relationship.
            weight: Edge weight (strength of connection).

        Returns:
            The created edge, or None if nodes don't exist.
        """
        if source_id not in self._nodes or target_id not in self._nodes:
            logger.warning("Cannot create edge: node(s) not found")
            return None

        edge = KnowledgeEdge(
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            weight=weight,
        )

        self._graph.add_edge(source_id, target_id, **edge.to_dict())
        logger.debug(f"Added edge: {source_id} --[{relation}]--> {target_id}")
        return edge

    def connect_related_nodes(
        self,
        node_id: str,
        max_edges: int = 5,
        temporal_window_sec: float = 300.0,
    ) -> int:
        """
        Automatically connect a node to related existing nodes.

        Creates edges based on:
        1. Same source file (relation: "same_file")
        2. Temporal proximity - created within window (relation: "same_cycle")

        Future enhancements (see docs/graph_edges.md):
        - Semantic similarity via embeddings
        - Concept co-occurrence analysis

        Args:
            node_id: The node to connect.
            max_edges: Maximum edges to create per relation type.
            temporal_window_sec: Time window for same-cycle detection.

        Returns:
            Number of edges created.
        """
        node = self._nodes.get(node_id)
        if not node:
            return 0

        edges_created = 0
        source_file = node.metadata.get("source") or node.metadata.get("source_file")

        # 1. Same-file edges: connect nodes that reference the same source file
        if source_file:
            same_file_nodes = [
                n for n in self._nodes.values()
                if n.id != node_id
                and (n.metadata.get("source") == source_file
                     or n.metadata.get("source_file") == source_file)
            ]
            # Sort by momentum (connect to most relevant first)
            same_file_nodes.sort(key=lambda n: n.momentum, reverse=True)

            for target in same_file_nodes[:max_edges]:
                # Avoid duplicate edges
                if not self._graph.has_edge(node_id, target.id):
                    self.add_edge(node_id, target.id, "same_file", weight=0.8)
                    edges_created += 1

        # 2. Same-cycle edges: connect nodes created close in time
        node_time = node.created_at
        temporal_nodes = [
            n for n in self._nodes.values()
            if n.id != node_id
            and abs(n.created_at - node_time) <= temporal_window_sec
        ]
        # Sort by time proximity
        temporal_nodes.sort(key=lambda n: abs(n.created_at - node_time))

        for target in temporal_nodes[:max_edges]:
            # Avoid duplicate edges (might already be connected via same_file)
            if not self._graph.has_edge(node_id, target.id):
                self.add_edge(node_id, target.id, "same_cycle", weight=0.5)
                edges_created += 1

        if edges_created > 0:
            logger.debug(f"Auto-connected {node_id}: {edges_created} edges created")

        return edges_created

    def connect_semantic(
        self,
        node_id: str,
        threshold: float = 0.6,
        max_edges: int = 3,
        embedder: Any = None,
    ) -> int:
        """
        Connect a node to semantically similar nodes using embeddings.

        Uses cosine similarity between content embeddings to find related nodes
        regardless of file or timing.

        Args:
            node_id: The node to connect.
            threshold: Minimum similarity score (0.0-1.0) to create edge.
            max_edges: Maximum semantic edges to create.
            embedder: Optional embedding function. If None, uses simple text similarity.

        Returns:
            Number of semantic edges created.
        """
        node = self._nodes.get(node_id)
        if not node:
            return 0

        edges_created = 0
        node_content = node.content.lower()[:500]  # Truncate for efficiency

        # Calculate similarity with all other nodes
        similarities: list[tuple[str, float]] = []

        for other_id, other_node in self._nodes.items():
            if other_id == node_id:
                continue
            # Skip if already connected
            if self._graph.has_edge(node_id, other_id):
                continue

            other_content = other_node.content.lower()[:500]

            if embedder:
                # Use embeddings for similarity
                try:
                    node_emb = embedder(node_content)
                    other_emb = embedder(other_content)
                    similarity = self._cosine_similarity(node_emb, other_emb)
                except Exception as e:
                    logger.debug(f"Embedding failed: {e}")
                    similarity = self._text_similarity(node_content, other_content)
            else:
                # Fallback to text-based similarity
                similarity = self._text_similarity(node_content, other_content)

            if similarity >= threshold:
                similarities.append((other_id, similarity))

        # Sort by similarity and take top N
        similarities.sort(key=lambda x: x[1], reverse=True)

        for target_id, similarity in similarities[:max_edges]:
            self.add_edge(node_id, target_id, "semantic", weight=similarity)
            edges_created += 1

        if edges_created > 0:
            logger.debug(f"Semantic edges for {node_id}: {edges_created} created")

        return edges_created

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple word-overlap similarity (Jaccard index)."""
        words1 = set(text1.split())
        words2 = set(text2.split())

        if not words1 or not words2:
            return 0.0

        intersection = len(words1 & words2)
        union = len(words1 | words2)

        return intersection / union if union > 0 else 0.0

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Cosine similarity between two vectors."""
        import math

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def get_node(self, node_id: str, activate: bool = True) -> KnowledgeNode | None:
        """
        Retrieve a node by ID.

        Args:
            node_id: The node ID.
            activate: If True, boost momentum on access.

        Returns:
            The node, or None if not found.
        """
        node = self._nodes.get(node_id)
        if node and activate:
            node.activate()
        return node

    def query_by_type(
        self,
        node_type: NodeType,
        min_momentum: float = 0.0,
        limit: int | None = None,
    ) -> list[KnowledgeNode]:
        """
        Query nodes by type, sorted by momentum.

        Args:
            node_type: Type to filter by.
            min_momentum: Minimum momentum threshold.
            limit: Maximum results to return.

        Returns:
            List of matching nodes, sorted by momentum descending.
        """
        matches = [
            node
            for node in self._nodes.values()
            if node.node_type == node_type and node.momentum >= min_momentum
        ]

        matches.sort(key=lambda n: n.momentum, reverse=True)

        if limit:
            matches = matches[:limit]

        return matches

    def query_hot(self, limit: int = 10) -> list[KnowledgeNode]:
        """Get the hottest (most relevant) nodes.
        
        Uses heapq.nlargest for O(n) instead of O(n log n) full sort.
        """
        return heapq.nlargest(limit, self._nodes.values(), key=lambda n: n.momentum)

    def get_related(
        self,
        node_id: str,
        relation: str | None = None,
        depth: int = 1,
    ) -> list[KnowledgeNode]:
        """
        Get nodes related to a given node.

        Args:
            node_id: Starting node ID.
            relation: Filter by relation type.
            depth: How many hops to traverse.

        Returns:
            List of related nodes.
        """
        if node_id not in self._graph:
            return []

        related_ids: set[str] = set()

        # BFS traversal
        current_level = {node_id}
        for _ in range(depth):
            next_level: set[str] = set()
            for nid in current_level:
                for successor in self._graph.successors(nid):
                    edge_data = self._graph.get_edge_data(nid, successor)
                    matches = relation is None or edge_data.get("relation") == relation
                    if matches and successor != node_id:
                        related_ids.add(successor)
                        next_level.add(successor)

                for predecessor in self._graph.predecessors(nid):
                    edge_data = self._graph.get_edge_data(predecessor, nid)
                    matches = relation is None or edge_data.get("relation") == relation
                    if matches and predecessor != node_id:
                        related_ids.add(predecessor)
                        next_level.add(predecessor)

            current_level = next_level

        return [self._nodes[nid] for nid in related_ids if nid in self._nodes]

    def decay_all(self) -> int:
        """
        Apply decay to all nodes based on time since last access.

        Returns:
            Number of nodes that decayed below minimum.
        """
        archived_count = 0

        for node in self._nodes.values():
            old_tier = node.tier
            node.decay()

            if node.momentum <= MIN_MOMENTUM:
                archived_count += 1
                logger.debug(f"Node {node.id} decayed to archive threshold")
            elif node.tier != old_tier:
                logger.debug(f"Node {node.id} moved from {old_tier.name} to {node.tier.name}")

        return archived_count

    def prune_cold(self, threshold: float = MIN_MOMENTUM) -> int:
        """
        Remove nodes below momentum threshold.

        Args:
            threshold: Momentum below which nodes are removed.

        Returns:
            Number of nodes removed.
        """
        to_remove = [nid for nid, node in self._nodes.items() if node.momentum < threshold]

        for nid in to_remove:
            self._graph.remove_node(nid)
            del self._nodes[nid]

    def backfill_edges(self, include_semantic: bool = False, embedder: Any = None) -> int:
        """
        Create edges for all existing nodes that don't have connections.

        This is a one-time operation to connect nodes that were created
        before auto_connect was enabled.

        Args:
            include_semantic: If True, also create semantic similarity edges.
            embedder: Optional embedding function for semantic edges.

        Returns:
            Total number of edges created.
        """
        total_edges = 0
        nodes_processed = 0
        semantic_edges = 0

        for node_id in list(self._nodes.keys()):
            # Skip nodes that already have edges (for basic edges)
            if self._graph.degree(node_id) == 0:
                edges = self.connect_related_nodes(node_id)
                total_edges += edges
                nodes_processed += 1

            # Semantic edges for all nodes if requested
            if include_semantic:
                sem_edges = self.connect_semantic(node_id, embedder=embedder)
                semantic_edges += sem_edges

        logger.info(
            f"Backfill complete: {nodes_processed} nodes processed, "
            f"{total_edges} basic edges, {semantic_edges} semantic edges"
        )
        return total_edges + semantic_edges

        return len(to_remove)

    def get_entanglement(self, node_id: str) -> float:
        """
        Calculate entanglement score for a node.

        Entanglement measures how connected a node is to the rest of the graph.
        Higher entanglement = more valuable/central idea.

        Formula: E = degree / max_possible_degree * weight_factor
        where weight_factor accounts for edge weights.

        Returns:
            Entanglement score between 0.0 and 1.0
        """
        if node_id not in self._nodes or len(self._nodes) <= 1:
            return 0.0

        degree = self._graph.degree(node_id)
        max_degree = len(self._nodes) - 1  # Max possible connections

        # Base score from degree
        base_score = degree / max_degree if max_degree > 0 else 0.0

        # Weight factor: average weight of connected edges
        edges = list(self._graph.edges(node_id, data=True))
        if edges:
            avg_weight = sum(e[2].get("weight", 1.0) for e in edges) / len(edges)
            weight_factor = avg_weight
        else:
            weight_factor = 0.0

        # Combined score (capped at 1.0)
        return min(1.0, base_score * (1 + weight_factor))

    def get_most_entangled(self, limit: int = 10) -> list[tuple[KnowledgeNode, float]]:
        """
        Get the most entangled (highly connected) nodes.

        These represent the most central/valuable ideas in the knowledge graph.

        Args:
            limit: Maximum number of nodes to return.

        Returns:
            List of (node, entanglement_score) tuples, sorted by score descending.
        """
        scored = [
            (node, self.get_entanglement(node.id))
            for node in self._nodes.values()
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:limit]

    def stats(self) -> dict[str, Any]:
        """Get graph statistics."""
        tier_counts = dict.fromkeys(StorageTier, 0)
        type_counts = dict.fromkeys(NodeType, 0)

        for node in self._nodes.values():
            tier_counts[node.tier] += 1
            type_counts[node.node_type] += 1

        # Calculate average entanglement
        if self._nodes:
            total_entanglement = sum(
                self.get_entanglement(nid) for nid in self._nodes
            )
            avg_entanglement = total_entanglement / len(self._nodes)
        else:
            avg_entanglement = 0.0

        return {
            "total_nodes": len(self._nodes),
            "total_edges": self._graph.number_of_edges(),
            "avg_entanglement": round(avg_entanglement, 3),
            "tiers": {t.name: c for t, c in tier_counts.items()},
            "types": {t.name: c for t, c in type_counts.items()},
        }

    def save(self) -> None:
        """Persist graph to disk."""
        data = {
            "nodes": [node.to_dict() for node in self._nodes.values()],
            "edges": [
                KnowledgeEdge(
                    source_id=u,
                    target_id=v,
                    relation=d.get("relation", "relates_to"),
                    weight=d.get("weight", 1.0),
                    created_at=d.get("created_at", time.time()),
                ).to_dict()
                for u, v, d in self._graph.edges(data=True)
            ],
            "node_counter": self._node_counter,
        }

        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(data, indent=2))
        logger.info(f"Saved graph: {len(self._nodes)} nodes, {self._graph.number_of_edges()} edges")

    def _load(self) -> None:
        """Load graph from disk."""
        try:
            data = json.loads(self.path.read_text())

            self._node_counter = data.get("node_counter", 0)

            for node_data in data.get("nodes", []):
                node = KnowledgeNode.from_dict(node_data)
                self._nodes[node.id] = node
                self._graph.add_node(node.id, **node.to_dict())

            for edge_data in data.get("edges", []):
                edge = KnowledgeEdge.from_dict(edge_data)
                self._graph.add_edge(edge.source_id, edge.target_id, **edge.to_dict())

            logger.info(
                f"Loaded graph: {len(self._nodes)} nodes, "
                f"{self._graph.number_of_edges()} edges"
            )

        except Exception as e:
            logger.warning(f"Could not load graph: {e}")


# Singleton
_graph: KnowledgeGraph | None = None


def get_graph() -> KnowledgeGraph:
    """Get or create the knowledge graph singleton."""
    global _graph
    if _graph is None:
        _graph = KnowledgeGraph()
    return _graph

