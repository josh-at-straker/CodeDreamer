"""
Codebase indexer for semantic search.

Parses source files, generates embeddings, and stores in ChromaDB
for retrieval during dream generation.

Uses AST-aware chunking for Python files to extract semantically
complete units (functions, classes, methods) rather than arbitrary
character-based splits.
"""

import ast
import hashlib
import logging
import os
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from .config import settings

logger = logging.getLogger(__name__)

# Supported file extensions and their language identifiers
SUPPORTED_EXTENSIONS: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".jsx": "javascript",
    ".tsx": "typescript",
    ".cpp": "cpp",
    ".c": "c",
    ".h": "cpp",
    ".hpp": "cpp",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".rb": "ruby",
    ".php": "php",
}

# Files/directories to skip
IGNORE_PATTERNS: set[str] = {
    "__pycache__",
    "node_modules",
    ".git",
    ".venv",
    "venv",
    ".pixi",
    "dist",
    "build",
    ".mypy_cache",
    ".pytest_cache",
    ".ruff_cache",
}


def assess_codebase_complexity(codebase_path: Path) -> int:
    """
    Assess the complexity of a codebase and return a score.

    The score is based on:
    - Total lines of code
    - Number of functions/classes
    - Average file length

    This can be used to dynamically adjust n_ctx.
    (From dream_20251228_142330_code_fix.md)

    Args:
        codebase_path: Path to the codebase root.

    Returns:
        Complexity score (higher = more complex).
        Suggested n_ctx = min(max(score // 10, MIN_CTX), MAX_CTX)
    """
    if not codebase_path.exists():
        return 4096  # Default

    total_lines = 0
    total_functions = 0
    file_count = 0

    for ext in SUPPORTED_EXTENSIONS:
        for file_path in codebase_path.rglob(f"*{ext}"):
            # Skip ignored directories
            if any(p in file_path.parts for p in IGNORE_PATTERNS):
                continue

            try:
                content = file_path.read_text(errors="replace")
                lines = content.splitlines()
                total_lines += len(lines)
                file_count += 1

                # Count function/class definitions (rough heuristic)
                pattern = r"^\s*(def|class|function|func)\s+"
                total_functions += len(re.findall(pattern, content, flags=re.MULTILINE))
            except (OSError, UnicodeDecodeError):
                continue

    # Score: weighted combination
    # More functions = more complex relationships
    # More lines = more context needed
    score = (total_lines // 10) + (total_functions * 50)

    logger.debug(
        f"Codebase complexity: {file_count} files, {total_lines} lines, "
        f"{total_functions} functions -> score {score}"
    )

    return score


@dataclass
class CodeChunk:
    """A chunk of code with metadata."""

    content: str
    file_path: str
    start_line: int
    end_line: int
    language: str
    chunk_type: str  # "function", "class", "module", "block"
    name: str | None = None
    docstring: str | None = None

    @property
    def id(self) -> str:
        """Generate unique ID for this chunk."""
        content_hash = hashlib.md5(self.content.encode()).hexdigest()[:8]
        return f"{self.file_path}:{self.start_line}-{self.end_line}:{content_hash}"

    @property
    def display_name(self) -> str:
        """Human-readable identifier."""
        if self.name:
            return f"{self.chunk_type}:{self.name}"
        return f"{self.chunk_type}@{self.file_path}:{self.start_line}"


@dataclass
class IndexStats:
    """Statistics from an indexing operation."""

    files_processed: int = 0
    chunks_created: int = 0
    chunks_skipped: int = 0
    errors: list[str] = field(default_factory=list)


class CodebaseIndexer:
    """Index a codebase for semantic search during dreaming."""

    def __init__(
        self,
        db_path: Path | None = None,
        collection_name: str = "codebase",
    ) -> None:
        """
        Initialize the indexer.

        Args:
            db_path: Path to ChromaDB storage. Defaults to settings.db_path.
            collection_name: Name of the ChromaDB collection.

        Raises:
            ValueError: If db_path parent directory doesn't exist.
            PermissionError: If db_path is not writable.
        """
        self.db_path = Path(db_path) if db_path else settings.db_path
        self.collection_name = collection_name

        # Validate db_path - ensure parent exists and is writable
        parent_dir = self.db_path.parent
        if not parent_dir.exists():
            try:
                parent_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created database directory: {parent_dir}")
            except OSError as e:
                raise ValueError(f"Cannot create database directory {parent_dir}: {e}") from e

        # Check write permissions by testing if we can create/access the path
        if self.db_path.exists() and not os.access(self.db_path, os.W_OK):
            raise PermissionError(f"No write access to database path: {self.db_path}")

        # Initialize ChromaDB with persistent storage
        self._client = chromadb.PersistentClient(
            path=str(self.db_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"description": "Indexed codebase for dream generation"},
        )

        logger.info(f"Initialized indexer with db at {self.db_path}")

    @property
    def collection(self) -> chromadb.Collection:
        """Get the ChromaDB collection."""
        return self._collection

    def index_directory(
        self,
        path: Path,
        chunk_size: int | None = None,
        overlap: int = 200,
    ) -> IndexStats:
        """
        Index all supported files in a directory.

        Args:
            path: Root directory to index.
            chunk_size: Target size for code chunks (in characters).
            overlap: Overlap between chunks for context continuity.

        Returns:
            IndexStats with processing summary.
        """
        stats = IndexStats()
        path = Path(path).resolve()
        
        # Use configured chunk size if not specified
        if chunk_size is None:
            chunk_size = settings.chunk_size

        if not path.exists():
            stats.errors.append(f"Path does not exist: {path}")
            return stats

        logger.info(f"Indexing directory: {path} (chunk_size={chunk_size})")

        for file_path in self._find_source_files(path):
            try:
                chunks = list(self._chunk_file(file_path, chunk_size, overlap))
                stats.files_processed += 1

                for chunk in chunks:
                    if self._add_chunk(chunk):
                        stats.chunks_created += 1
                    else:
                        stats.chunks_skipped += 1

                logger.debug(f"Indexed {file_path.name}: {len(chunks)} chunks")

            except Exception as e:
                error_msg = f"Error indexing {file_path}: {e}"
                stats.errors.append(error_msg)
                logger.warning(error_msg)

        logger.info(
            f"Indexing complete: {stats.files_processed} files, "
            f"{stats.chunks_created} chunks created"
        )
        return stats

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: dict | None = None,
    ) -> list[CodeChunk]:
        """
        Search for relevant code chunks.

        Args:
            query_text: Search query (natural language or code).
            n_results: Maximum number of results.
            where: Optional metadata filter.

        Returns:
            List of matching CodeChunk objects.
        """
        results = self._collection.query(
            query_texts=[query_text],
            n_results=n_results,
            where=where,
        )

        chunks = []
        if results["documents"] and results["metadatas"]:
            for doc, meta in zip(results["documents"][0], results["metadatas"][0], strict=False):
                chunk = CodeChunk(
                    content=doc,
                    file_path=meta.get("file_path", ""),
                    start_line=meta.get("start_line", 0),
                    end_line=meta.get("end_line", 0),
                    language=meta.get("language", ""),
                    chunk_type=meta.get("chunk_type", "block"),
                    name=meta.get("name"),
                    docstring=meta.get("docstring"),
                )
                chunks.append(chunk)

        return chunks

    def get_random_chunk(self) -> CodeChunk | None:
        """Get a random chunk for dream seeding."""
        # ChromaDB doesn't have random sampling, so we get all IDs and pick one
        import random

        all_ids = self._collection.get()["ids"]
        if not all_ids:
            return None

        random_id = random.choice(all_ids)
        result = self._collection.get(ids=[random_id], include=["documents", "metadatas"])

        if result["documents"] and result["metadatas"]:
            meta = result["metadatas"][0]
            return CodeChunk(
                content=result["documents"][0],
                file_path=meta.get("file_path", ""),
                start_line=meta.get("start_line", 0),
                end_line=meta.get("end_line", 0),
                language=meta.get("language", ""),
                chunk_type=meta.get("chunk_type", "block"),
                name=meta.get("name"),
            )
        return None

    def clear(self) -> None:
        """Clear all indexed data."""
        self._client.delete_collection(self.collection_name)
        self._collection = self._client.create_collection(
            name=self.collection_name,
            metadata={"description": "Indexed codebase for dream generation"},
        )
        logger.info("Cleared all indexed data")

    def _find_source_files(self, root: Path) -> Iterator[Path]:
        """Find all supported source files in directory tree."""
        import fnmatch
        
        # Parse configurable exclude patterns
        exclude_patterns = [p.strip() for p in settings.exclude_patterns.split(",") if p.strip()]
        
        for path in root.rglob("*"):
            # Skip ignored directories (hardcoded system dirs)
            if any(ignored in path.parts for ignored in IGNORE_PATTERNS):
                continue
            
            # Skip configurable exclusions (directories or file patterns)
            relative_path = path.relative_to(root)
            skip = False
            for pattern in exclude_patterns:
                # Check if pattern matches any directory component
                if pattern in relative_path.parts:
                    skip = True
                    break
                # Check if pattern matches filename (glob pattern)
                if fnmatch.fnmatch(path.name, pattern):
                    skip = True
                    break
            if skip:
                continue

            # Check if supported extension
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
                yield path

    def _chunk_python_ast(
        self,
        file_path: Path,
        content: str,
    ) -> Iterator[CodeChunk]:
        """
        Use Python AST to extract semantically complete chunks.

        Extracts:
        - Module-level docstring
        - Classes (with all methods as one chunk)
        - Top-level functions
        - Standalone code blocks

        This produces better embeddings than arbitrary character splits
        because each chunk represents a complete semantic unit.
        """
        try:
            tree = ast.parse(content, filename=str(file_path))
        except SyntaxError as e:
            logger.debug(f"AST parse failed for {file_path}: {e}")
            return  # Caller will fall back to line-based chunking

        lines = content.split("\n")

        # Extract module docstring
        if (
            tree.body
            and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)
        ):
            docstring = tree.body[0].value.value
            yield CodeChunk(
                content=f'"""{docstring}"""',
                file_path=str(file_path),
                start_line=1,
                end_line=tree.body[0].end_lineno or 1,
                language="python",
                chunk_type="module_doc",
                name=file_path.stem,
                docstring=docstring[:200] if len(docstring) > 200 else docstring,
            )

        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                # Extract entire class as one chunk, INCLUDING decorators
                # Decorators appear before node.lineno, so check decorator_list
                if node.decorator_list:
                    start = node.decorator_list[0].lineno
                else:
                    start = node.lineno
                end = node.end_lineno or start
                class_content = "\n".join(lines[start - 1 : end])

                # Get class docstring
                docstring = ast.get_docstring(node)

                yield CodeChunk(
                    content=class_content,
                    file_path=str(file_path),
                    start_line=start,
                    end_line=end,
                    language="python",
                    chunk_type="class",
                    name=node.name,
                    docstring=docstring[:200] if docstring and len(docstring) > 200 else docstring,
                )

                # Also extract individual methods for fine-grained search
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        # Include method decorators (@property, @staticmethod, etc.)
                        if item.decorator_list:
                            method_start = item.decorator_list[0].lineno
                        else:
                            method_start = item.lineno
                        method_end = item.end_lineno or method_start
                        method_content = "\n".join(lines[method_start - 1 : method_end])
                        method_doc = ast.get_docstring(item)

                        yield CodeChunk(
                            content=method_content,
                            file_path=str(file_path),
                            start_line=method_start,
                            end_line=method_end,
                            language="python",
                            chunk_type="method",
                            name=f"{node.name}.{item.name}",
                            docstring=method_doc[:200] if method_doc and len(method_doc) > 200 else method_doc,
                        )

            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Top-level function, INCLUDING decorators
                if node.decorator_list:
                    start = node.decorator_list[0].lineno
                else:
                    start = node.lineno
                end = node.end_lineno or start
                func_content = "\n".join(lines[start - 1 : end])
                docstring = ast.get_docstring(node)

                # Build signature for better embedding
                args = []
                for arg in node.args.args:
                    arg_str = arg.arg
                    if arg.annotation:
                        try:
                            arg_str += f": {ast.unparse(arg.annotation)}"
                        except Exception:
                            pass
                    args.append(arg_str)

                signature = f"def {node.name}({', '.join(args)})"
                if node.returns:
                    try:
                        signature += f" -> {ast.unparse(node.returns)}"
                    except Exception:
                        pass

                yield CodeChunk(
                    content=func_content,
                    file_path=str(file_path),
                    start_line=start,
                    end_line=end,
                    language="python",
                    chunk_type="function",
                    name=node.name,
                    docstring=docstring[:200] if docstring and len(docstring) > 200 else docstring,
                )

    def _chunk_js_ts_regex(
        self,
        file_path: Path,
        content: str,
        language: str,
    ) -> Iterator[CodeChunk]:
        """
        Use regex patterns to extract JS/TS functions and classes.

        Not as accurate as AST but avoids heavy dependencies like tree-sitter.
        Falls back gracefully when patterns don't match.
        """
        lines = content.split("\n")

        # Patterns for common JS/TS constructs
        patterns = [
            # Arrow functions: const name = (...) => { ... }
            (r"(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s*)?\([^)]*\)\s*(?::\s*[^=]+)?\s*=>", "function"),
            # Regular functions: function name(...) { ... }
            (r"(?:export\s+)?(?:async\s+)?function\s+(\w+)", "function"),
            # Classes: class Name { ... }
            (r"(?:export\s+)?class\s+(\w+)", "class"),
            # Interfaces: interface Name { ... }
            (r"(?:export\s+)?interface\s+(\w+)", "interface"),
            # Type aliases: type Name = ...
            (r"(?:export\s+)?type\s+(\w+)", "type"),
        ]

        # Find all matches with their positions
        matches = []
        for pattern, chunk_type in patterns:
            for match in re.finditer(pattern, content, re.MULTILINE):
                matches.append((match.start(), match.group(1), chunk_type))

        # Sort by position
        matches.sort(key=lambda x: x[0])

        # Extract chunks between matches
        for i, (pos, name, chunk_type) in enumerate(matches):
            # Find line number
            start_line = content[:pos].count("\n") + 1

            # Find end: next match or brace counting
            if i + 1 < len(matches):
                next_pos = matches[i + 1][0]
                chunk_content = content[pos:next_pos].rstrip()
            else:
                chunk_content = content[pos:].rstrip()

            # Try to find proper end via brace matching (simplified)
            brace_count = 0
            end_pos = 0
            in_chunk = False
            for j, char in enumerate(chunk_content):
                if char == "{":
                    brace_count += 1
                    in_chunk = True
                elif char == "}":
                    brace_count -= 1
                    if in_chunk and brace_count == 0:
                        end_pos = j + 1
                        break

            if end_pos > 0:
                chunk_content = chunk_content[:end_pos]

            end_line = start_line + chunk_content.count("\n")

            if len(chunk_content.strip()) > 50:
                yield CodeChunk(
                    content=chunk_content,
                    file_path=str(file_path),
                    start_line=start_line,
                    end_line=end_line,
                    language=language,
                    chunk_type=chunk_type,
                    name=name,
                )

    def _chunk_file(
        self,
        file_path: Path,
        chunk_size: int,
        overlap: int,
    ) -> Iterator[CodeChunk]:
        """
        Split a file into chunks for indexing.

        Uses AST-aware chunking for Python files to extract semantically
        complete units. Falls back to line-based chunking for other languages
        or when AST parsing fails.
        """
        language = SUPPORTED_EXTENSIONS.get(file_path.suffix.lower(), "text")

        try:
            content = file_path.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning(f"Could not read {file_path}: {e}")
            return

        # Try AST-aware chunking first
        ast_chunks = []

        if language == "python":
            ast_chunks = list(self._chunk_python_ast(file_path, content))
            if ast_chunks:
                logger.debug(f"AST chunking {file_path.name}: {len(ast_chunks)} semantic units")
                yield from ast_chunks
                return

        elif language in ("javascript", "typescript"):
            ast_chunks = list(self._chunk_js_ts_regex(file_path, content, language))
            if ast_chunks:
                logger.debug(f"Regex chunking {file_path.name}: {len(ast_chunks)} units")
                yield from ast_chunks
                return

        # Fall back to line-based chunking
        logger.debug(f"Line-based chunking for {file_path.name}")
        lines = content.split("\n")
        current_chunk: list[str] = []
        current_start = 1
        current_size = 0

        for i, line in enumerate(lines, start=1):
            line_size = len(line) + 1  # +1 for newline
            current_chunk.append(line)
            current_size += line_size

            if current_size >= chunk_size:
                chunk_content = "\n".join(current_chunk)
                yield CodeChunk(
                    content=chunk_content,
                    file_path=str(file_path),
                    start_line=current_start,
                    end_line=i,
                    language=language,
                    chunk_type="block",
                    name=self._extract_chunk_name(chunk_content, language),
                )

                # Keep overlap lines for next chunk
                overlap_lines = max(1, overlap // 50)  # Approximate lines from char overlap
                current_chunk = current_chunk[-overlap_lines:]
                current_start = max(1, i - overlap_lines + 1)
                current_size = sum(len(line) + 1 for line in current_chunk)

        # Handle remaining content
        if current_chunk:
            chunk_content = "\n".join(current_chunk)
            yield CodeChunk(
                content=chunk_content,
                file_path=str(file_path),
                start_line=current_start,
                end_line=len(lines),
                language=language,
                chunk_type="block",
                name=self._extract_chunk_name(chunk_content, language),
            )

    def _extract_chunk_name(self, content: str, language: str) -> str | None:
        """Try to extract a meaningful name from code chunk."""
        # Simple heuristics for common patterns
        patterns = {
            "python": [
                r"^(?:async\s+)?def\s+(\w+)",
                r"^class\s+(\w+)",
            ],
            "javascript": [
                r"(?:function|const|let|var)\s+(\w+)",
                r"class\s+(\w+)",
            ],
            "typescript": [
                r"(?:function|const|let|var)\s+(\w+)",
                r"class\s+(\w+)",
                r"interface\s+(\w+)",
            ],
            "cpp": [
                r"(?:void|int|bool|auto|static)\s+(\w+)\s*\(",
                r"class\s+(\w+)",
                r"struct\s+(\w+)",
            ],
        }

        for pattern in patterns.get(language, []):
            match = re.search(pattern, content, re.MULTILINE)
            if match:
                return match.group(1)

        return None

    def _add_chunk(self, chunk: CodeChunk) -> bool:
        """Add a chunk to the collection. Returns True if added, False if skipped."""
        # Skip very small chunks
        if len(chunk.content.strip()) < 50:
            return False

        # Build rich document for embedding
        # Include name and docstring for better semantic matching
        doc_parts = []
        if chunk.name:
            doc_parts.append(f"# {chunk.chunk_type}: {chunk.name}")
        if chunk.docstring:
            doc_parts.append(f"# {chunk.docstring}")
        doc_parts.append(chunk.content)
        rich_document = "\n".join(doc_parts)

        self._collection.add(
            ids=[chunk.id],
            documents=[rich_document],
            metadatas=[
                {
                    "file_path": chunk.file_path,
                    "start_line": chunk.start_line,
                    "end_line": chunk.end_line,
                    "language": chunk.language,
                    "chunk_type": chunk.chunk_type,
                    "name": chunk.name or "",
                    "docstring": chunk.docstring or "",
                }
            ],
        )
        return True

