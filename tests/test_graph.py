"""Tests for knowledge graph."""

import tempfile
from pathlib import Path

from codedreamer.graph import KnowledgeGraph, KnowledgeNode, NodeType, StorageTier


def test_add_node() -> None:
    """Test adding nodes to the graph."""
    with tempfile.TemporaryDirectory() as tmpdir:
        graph = KnowledgeGraph(path=Path(tmpdir) / "test.json")

        node = graph.add_node(
            content="Test fact about Python",
            node_type=NodeType.FACT,
        )

        assert node.id.startswith("fact_")
        assert node.content == "Test fact about Python"
        assert node.node_type == NodeType.FACT
        assert node.momentum == 1.0
        assert node.tier == StorageTier.HOT


def test_add_edge() -> None:
    """Test adding edges between nodes."""
    with tempfile.TemporaryDirectory() as tmpdir:
        graph = KnowledgeGraph(path=Path(tmpdir) / "test.json")

        node1 = graph.add_node("Concept A", NodeType.CONCEPT)
        node2 = graph.add_node("Concept B", NodeType.CONCEPT)

        edge = graph.add_edge(node1.id, node2.id, "relates_to")

        assert edge is not None
        assert edge.source_id == node1.id
        assert edge.target_id == node2.id
        assert edge.relation == "relates_to"


def test_momentum_decay() -> None:
    """Test that momentum decays over time."""
    node = KnowledgeNode(
        id="test",
        content="Test content",
        node_type=NodeType.FACT,
        momentum=1.0,
    )

    # Decay for 10 hours
    node.decay(hours=10)

    # Should be less than 1.0 now
    assert node.momentum < 1.0
    assert node.momentum > 0.0


def test_momentum_activation() -> None:
    """Test that accessing a node boosts momentum."""
    node = KnowledgeNode(
        id="test",
        content="Test content",
        node_type=NodeType.FACT,
        momentum=0.5,
    )

    node.activate()

    assert node.momentum > 0.5
    assert node.momentum <= 1.0


def test_tier_classification() -> None:
    """Test storage tier assignment based on momentum."""
    hot_node = KnowledgeNode(id="hot", content="", node_type=NodeType.FACT, momentum=0.9)
    warm_node = KnowledgeNode(id="warm", content="", node_type=NodeType.FACT, momentum=0.5)
    cold_node = KnowledgeNode(id="cold", content="", node_type=NodeType.FACT, momentum=0.2)

    assert hot_node.tier == StorageTier.HOT
    assert warm_node.tier == StorageTier.WARM
    assert cold_node.tier == StorageTier.COLD


def test_graph_save_load() -> None:
    """Test graph persistence."""
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "test.json"

        # Create and save
        graph1 = KnowledgeGraph(path=path)
        graph1.add_node("Test fact", NodeType.FACT)
        graph1.add_node("Test concept", NodeType.CONCEPT)
        graph1.save()

        # Load in new instance
        graph2 = KnowledgeGraph(path=path)

        assert len(graph2._nodes) == 2
        stats = graph2.stats()
        assert stats["total_nodes"] == 2



