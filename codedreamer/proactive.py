"""
Proactive Memory - Anticipates context before it's needed.

Features:
- Predicts what context will be useful based on current focus
- Uses graph edges, imports, and TRM to build anticipatory context
- Reduces "cold start" for each dream cycle
"""

import ast
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import settings
from .graph import KnowledgeGraph, NodeType, get_graph
from .trm import TRMStream, get_trm

logger = logging.getLogger(__name__)


@dataclass
class ProactiveContext:
    """Context package assembled proactively."""

    source_file: str
    related_files: list[str] = field(default_factory=list)
    imported_modules: list[str] = field(default_factory=list)
    import_snippets: dict[str, str] = field(default_factory=dict)  # module -> code snippet
    graph_context: list[str] = field(default_factory=list)
    trm_context: str = ""
    codebase_overview: str = ""  # Auto-generated summary of key files
    confidence: float = 0.0

    def to_prompt_section(self) -> str:
        """Format as a prompt section for the LLM."""
        sections = []

        # Codebase overview (extended context mode)
        if self.codebase_overview:
            sections.append(f"**Codebase Overview**:\n{self.codebase_overview}")

        if self.imported_modules:
            sections.append(
                f"**Imports**: This file uses: {', '.join(self.imported_modules[:10])}"
            )

        # Import snippets (extended context mode)
        if self.import_snippets:
            snippets_section = "**Imported Code Context**:\n"
            for module, snippet in list(self.import_snippets.items())[:3]:
                snippets_section += f"\n`{module}`:\n```\n{snippet[:500]}...\n```\n"
            sections.append(snippets_section)

        if self.related_files:
            sections.append(
                f"**Related Files**: Often seen with: {', '.join(self.related_files[:5])}"
            )

        if self.graph_context:
            sections.append(
                "**Previous Insights**:\n" + "\n".join(f"- {c}" for c in self.graph_context[:3])
            )

        if self.trm_context:
            sections.append(f"**Recent Thoughts**:\n{self.trm_context}")

        if not sections:
            return ""

        return "## Proactive Context (anticipated relevant info)\n\n" + "\n\n".join(sections)


class ProactiveMemory:
    """
    Anticipates and pre-fetches context before it's needed.

    Uses multiple signals:
    1. Import analysis - what modules does this file depend on?
    2. Graph neighbors - what's connected to previous dreams about this file?
    3. TRM stream - what recent insights might be relevant?
    4. Co-occurrence - what files are often mentioned together?
    """

    def __init__(
        self,
        graph: KnowledgeGraph | None = None,
        trm: TRMStream | None = None,
    ) -> None:
        self._graph = graph or get_graph()
        self._trm = trm or get_trm()
        self._file_cooccurrence: dict[str, set[str]] = {}
        self._import_cache: dict[str, list[str]] = {}

    def get_context(self, source_file: str, code_content: str | None = None) -> ProactiveContext:
        """
        Build proactive context for a given source file.

        Args:
            source_file: Path to the file being analyzed.
            code_content: Optional code content (avoids re-reading file).

        Returns:
            ProactiveContext with anticipated relevant information.
        """
        ctx = ProactiveContext(source_file=source_file)

        # 1. Extract imports from the code
        # Note: code_content may be a chunk that doesn't include imports
        # So we always try to read the full file for imports
        file_path = Path(source_file)
        logger.debug(f"Proactive: source_file={source_file}, exists={file_path.exists()}")
        
        if file_path.exists():
            try:
                full_content = file_path.read_text(errors="ignore")
                logger.debug(f"Read {len(full_content)} chars from {source_file}")
                ctx.imported_modules = self._extract_imports(full_content)
            except Exception as e:
                logger.warning(f"Could not read {source_file}: {e}")
                # Fallback to chunk content
                if code_content:
                    ctx.imported_modules = self._extract_imports(code_content)
        elif code_content:
            # File doesn't exist locally (Docker path), try chunk
            logger.debug(f"File not found, using chunk content ({len(code_content)} chars)")
            ctx.imported_modules = self._extract_imports(code_content)
        else:
            logger.warning(f"No source available for import extraction: {source_file}")

        # 2. Find related files from graph (nodes about same file)
        ctx.related_files = self._find_related_files(source_file)

        # 3. Get relevant graph context (previous insights about this file)
        ctx.graph_context = self._get_graph_context(source_file)

        # 4. Get TRM context (recent relevant thoughts)
        ctx.trm_context = self._get_trm_context(source_file)
        
        # 5. Extended context mode: import snippets and codebase overview
        if settings.extended_context:
            ctx.import_snippets = self._get_import_snippets(source_file, ctx.imported_modules)
            ctx.codebase_overview = self._get_codebase_overview(source_file)

        # Calculate confidence based on how much context we found
        signals = [
            len(ctx.imported_modules) > 0,
            len(ctx.related_files) > 0,
            len(ctx.graph_context) > 0,
            len(ctx.trm_context) > 0,
        ]
        ctx.confidence = sum(signals) / len(signals)

        # Log at INFO level so we can verify it's working
        extended_info = f", snippets={len(ctx.import_snippets)}" if settings.extended_context else ""
        logger.info(
            f"Proactive context for {Path(source_file).name}: "
            f"confidence={ctx.confidence:.2f}, "
            f"imports={len(ctx.imported_modules)}, "
            f"related={len(ctx.related_files)}, "
            f"graph={len(ctx.graph_context)}, "
            f"trm={len(ctx.trm_context) > 0}{extended_info}"
        )
        
        # Log what we're actually providing
        if ctx.imported_modules:
            logger.info(f"  Imports: {ctx.imported_modules[:5]}")
        if ctx.graph_context:
            logger.info(f"  Graph insights: {len(ctx.graph_context)} found")

        return ctx

    def _extract_imports(self, code: str) -> list[str]:
        """Extract import statements from Python code."""
        imports = []
        
        if not code or len(code) < 10:
            logger.debug(f"Code too short for import extraction: {len(code) if code else 0} chars")
            return imports

        try:
            tree = ast.parse(code)
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        imports.append(alias.name.split(".")[0])
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        imports.append(node.module.split(".")[0])
            logger.debug(f"AST extracted {len(imports)} imports")
        except SyntaxError as e:
            # Fallback to regex for non-Python or invalid syntax
            logger.debug(f"AST parse failed ({e}), using regex fallback")
            import_pattern = r"^(?:from|import)\s+([a-zA-Z_][a-zA-Z0-9_]*)"
            for match in re.finditer(import_pattern, code, re.MULTILINE):
                imports.append(match.group(1))
            logger.debug(f"Regex extracted {len(imports)} imports")

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for imp in imports:
            if imp not in seen:
                seen.add(imp)
                unique.append(imp)

        return unique

    def _find_related_files(self, source_file: str) -> list[str]:
        """Find files that are often mentioned alongside this one."""
        related = set()
        source_name = Path(source_file).name

        # Look at graph nodes about this file and find their neighbors
        for node in self._graph._nodes.values():
            node_source = node.metadata.get("source") or node.metadata.get("source_file", "")
            if source_name in str(node_source):
                # Get neighbors of this node
                try:
                    neighbors = list(self._graph._graph.neighbors(node.id))
                    for neighbor_id in neighbors:
                        neighbor = self._graph._nodes.get(neighbor_id)
                        if neighbor:
                            neighbor_source = (
                                neighbor.metadata.get("source")
                                or neighbor.metadata.get("source_file", "")
                            )
                            if neighbor_source and source_name not in str(neighbor_source):
                                related.add(Path(str(neighbor_source)).name)
                except Exception:
                    pass

        return list(related)[:5]

    def _get_graph_context(self, source_file: str) -> list[str]:
        """Get previous insights from the graph about this file."""
        context = []
        source_name = Path(source_file).name

        # Find nodes about this file, sorted by momentum
        relevant_nodes = []
        for node in self._graph._nodes.values():
            node_source = node.metadata.get("source") or node.metadata.get("source_file", "")
            if source_name in str(node_source):
                relevant_nodes.append(node)

        # Sort by momentum and take top 3
        relevant_nodes.sort(key=lambda n: n.momentum, reverse=True)

        for node in relevant_nodes[:3]:
            # Truncate content for context
            snippet = node.content[:150].replace("\n", " ").strip()
            if snippet:
                context.append(f"[{node.node_type.name}] {snippet}...")

        return context

    def _get_trm_context(self, source_file: str) -> str:
        """Get relevant TRM fragments for this file."""
        source_name = Path(source_file).name

        # Get TRM fragments, filter for relevance
        fragments = self._trm.get_relevant(source_file, max_fragments=2)

        if fragments:
            return "\n".join(f"- {f.content[:100]}..." for f in fragments)
        return ""

    def _get_import_snippets(self, source_file: str, imports: list[str]) -> dict[str, str]:
        """
        Get code snippets from imported modules (extended context mode).
        
        For local project imports, reads the first ~500 chars of the file
        to give context about what the import provides.
        """
        snippets = {}
        source_dir = Path(source_file).parent
        
        for module in imports[:5]:  # Limit to 5 imports
            # Skip standard library modules
            if module in {"os", "sys", "re", "json", "time", "datetime", "logging", 
                          "typing", "pathlib", "asyncio", "collections", "functools",
                          "dataclasses", "abc", "enum", "uuid", "hashlib", "base64"}:
                continue
            
            # Try to find the module file in the same project
            possible_paths = [
                source_dir / f"{module}.py",
                source_dir.parent / f"{module}.py",
                source_dir.parent / module / "__init__.py",
                source_dir / module / "__init__.py",
            ]
            
            for module_path in possible_paths:
                if module_path.exists():
                    try:
                        content = module_path.read_text(errors="ignore")
                        # Get first ~500 chars (usually includes imports and first class/function)
                        snippets[module] = content[:500]
                        logger.debug(f"Extended context: loaded snippet from {module_path}")
                        break
                    except Exception as e:
                        logger.debug(f"Could not read {module_path}: {e}")
        
        return snippets

    def _get_codebase_overview(self, source_file: str) -> str:
        """
        Generate a brief codebase overview (extended context mode).
        
        Provides the model with architectural context about the project structure.
        """
        source_dir = Path(source_file).parent
        
        # Find the project root (look for common markers)
        project_root = source_dir
        for _ in range(5):  # Max 5 levels up
            if (project_root / "pyproject.toml").exists() or \
               (project_root / "setup.py").exists() or \
               (project_root / "requirements.txt").exists():
                break
            if project_root.parent == project_root:
                break
            project_root = project_root.parent
        
        # List key Python files in the project
        try:
            py_files = list(project_root.rglob("*.py"))
            # Filter out tests, __pycache__, etc.
            py_files = [f for f in py_files 
                       if "__pycache__" not in str(f) 
                       and "test" not in f.name.lower()
                       and f.name != "__init__.py"]
            
            if not py_files:
                return ""
            
            # Build a simple overview
            overview_parts = [f"Project has {len(py_files)} Python files."]
            
            # Group by directory
            dirs = {}
            for f in py_files[:20]:  # Limit
                rel_path = f.relative_to(project_root)
                dir_name = str(rel_path.parent) if rel_path.parent != Path(".") else "root"
                if dir_name not in dirs:
                    dirs[dir_name] = []
                dirs[dir_name].append(f.name)
            
            for dir_name, files in list(dirs.items())[:5]:
                overview_parts.append(f"- {dir_name}/: {', '.join(files[:5])}")
            
            return "\n".join(overview_parts)
        except Exception as e:
            logger.debug(f"Could not generate codebase overview: {e}")
            return ""

    def record_cooccurrence(self, file1: str, file2: str) -> None:
        """Record that two files appeared together (for future predictions)."""
        name1 = Path(file1).name
        name2 = Path(file2).name

        if name1 not in self._file_cooccurrence:
            self._file_cooccurrence[name1] = set()
        if name2 not in self._file_cooccurrence:
            self._file_cooccurrence[name2] = set()

        self._file_cooccurrence[name1].add(name2)
        self._file_cooccurrence[name2].add(name1)


# Singleton instance
_proactive_memory: ProactiveMemory | None = None


def get_proactive_memory() -> ProactiveMemory:
    """Get or create the ProactiveMemory singleton."""
    global _proactive_memory
    if _proactive_memory is None:
        _proactive_memory = ProactiveMemory()
    return _proactive_memory


