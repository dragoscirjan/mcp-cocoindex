"""CocoIndex indexer - handles code indexing and search operations."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cocoindex
import cocoindex.flow as _cocoindex_flow
import psycopg
from pgvector.psycopg import register_vector
from psycopg import sql

from .analyzer import (
    detect_language,
    extract_symbols_from_file,
    has_tree_sitter_support,
)
from .config import CocoIndexConfig

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """A single search result."""

    file_path: str
    content: str
    score: float
    start_line: int | None = None
    end_line: int | None = None
    language: str | None = None


@dataclass
class IndexInfo:
    """Information about an index."""

    name: str
    repository_path: str
    file_count: int
    chunk_count: int
    created_at: str | None = None


@dataclass
class SymbolSearchResult:
    """A single symbol search result."""

    file_path: str
    symbol_name: str
    symbol_kind: str
    category: str
    line: int
    column: int
    context: str
    parent_name: str | None = None


@dataclass
class SymbolIndexInfo:
    """Information about a symbol index."""

    name: str
    repository_path: str
    file_count: int
    symbol_count: int
    definition_count: int
    reference_count: int


# Default embedding model
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# Transform flow for generating embeddings - can be used at index and query time
@cocoindex.transform_flow()
def text_to_embedding(
    text: cocoindex.DataSlice[str],
) -> cocoindex.DataSlice[list[float]]:
    """Transform text to embedding vector."""
    return text.transform(
        cocoindex.functions.SentenceTransformerEmbed(model=DEFAULT_EMBEDDING_MODEL)
    )


class CocoIndexer:
    """Handles CocoIndex operations for code indexing and search."""

    def __init__(self, config: CocoIndexConfig):
        """Initialize the indexer with configuration."""
        self.config = config
        self._initialized = False
        self._flows: dict[str, Any] = {}

    async def initialize(self) -> None:
        """Initialize CocoIndex with the configured settings."""
        if self._initialized:
            return

        # Set database URL environment variable for CocoIndex
        os.environ["COCOINDEX_DATABASE_URL"] = self.config.postgres.connection_string

        # Initialize cocoindex
        cocoindex.init()
        self._initialized = True
        logger.info("CocoIndex initialized")

    async def close(self) -> None:
        """Clean up resources."""
        self._flows.clear()
        self._initialized = False
        logger.info("CocoIndex closed")

    def _get_index_name(self, repo_path: str) -> str:
        """Generate a consistent index name from repository path."""
        path = Path(repo_path).resolve()
        return f"{path.parent.name}_{path.name}".replace("-", "_").replace(".", "_")

    def _create_flow_def(
        self,
        index_name: str,
        repo_path: str,
        include_patterns: list[str],
        exclude_patterns: list[str],
    ):
        """Create a flow definition for indexing a repository.

        Returns a plain function (not decorated) to be passed to open_flow.
        """
        embedding_model = f"sentence-transformers/{self.config.embedding.model_name}"

        # Note: Do NOT use @cocoindex.flow_def decorator here!
        # open_flow will handle registration. Using the decorator would
        # double-register the flow.
        def code_index_flow(
            flow_builder: cocoindex.FlowBuilder,
            data_scope: cocoindex.DataScope,
        ):
            # Add source: local files from repository
            data_scope["files"] = flow_builder.add_source(
                cocoindex.sources.LocalFile(
                    path=repo_path,
                    included_patterns=include_patterns,
                    excluded_patterns=exclude_patterns,
                )
            )

            # Collector for chunks to be exported
            code_chunks = data_scope.add_collector()

            # Process each file
            with data_scope["files"].row() as file:
                # Split file content into chunks (language-aware)
                file["chunks"] = file["content"].transform(
                    cocoindex.functions.SplitRecursively(),
                    language="python",  # Default, could detect per-file
                    chunk_size=1000,
                    chunk_overlap=200,
                )

                # Process each chunk
                with file["chunks"].row() as chunk:
                    # Generate embedding for the chunk
                    chunk["embedding"] = chunk["text"].transform(
                        cocoindex.functions.SentenceTransformerEmbed(model=embedding_model)
                    )

                    # Collect chunk data
                    code_chunks.collect(
                        file_path=file["filename"],
                        location=chunk["location"],
                        content=chunk["text"],
                        embedding=chunk["embedding"],
                    )

            # Export to Postgres with vector index
            code_chunks.export(
                f"code_index_{index_name}",
                cocoindex.targets.Postgres(
                    table_name=f"code_index_{index_name}",
                ),
                primary_key_fields=["file_path", "location"],
                vector_indexes=[
                    cocoindex.VectorIndexDef(
                        field_name="embedding",
                        metric=cocoindex.VectorSimilarityMetric.COSINE_SIMILARITY,
                    )
                ],
            )

        return code_index_flow

    async def index_repository(
        self,
        repo_path: str,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
    ) -> IndexInfo:
        """
        Index a repository for semantic search.

        Args:
            repo_path: Path to the repository to index
            include_patterns: Glob patterns for files to include
            exclude_patterns: Glob patterns for files to exclude

        Returns:
            Information about the created index
        """
        if not self._initialized:
            await self.initialize()

        repo_path = str(Path(repo_path).resolve())
        index_name = self._get_index_name(repo_path)

        # Default patterns for code files
        if include_patterns is None:
            include_patterns = [
                "**/*.py",
                "**/*.js",
                "**/*.ts",
                "**/*.tsx",
                "**/*.jsx",
                "**/*.go",
                "**/*.rs",
                "**/*.java",
                "**/*.c",
                "**/*.cpp",
                "**/*.h",
                "**/*.hpp",
                "**/*.cs",
                "**/*.rb",
                "**/*.php",
                "**/*.swift",
                "**/*.kt",
                "**/*.scala",
                "**/*.md",
            ]

        if exclude_patterns is None:
            exclude_patterns = [
                "**/node_modules/**",
                "**/.git/**",
                "**/vendor/**",
                "**/dist/**",
                "**/build/**",
                "**/__pycache__/**",
                "**/.venv/**",
                "**/venv/**",
            ]

        logger.info("Indexing repository: %s as %s", repo_path, index_name)

        # Create flow definition (plain function, not decorated)
        flow_def = self._create_flow_def(index_name, repo_path, include_patterns, exclude_patterns)

        # Close any existing flow with this name (instance-level and global registry)
        if index_name in self._flows:
            old_flow = self._flows[index_name].get("flow")
            if old_flow:
                try:
                    old_flow.close()
                except Exception:
                    pass
            del self._flows[index_name]

        # Also close from global registry if another indexer instance registered it
        try:
            existing = _cocoindex_flow.flow_by_name(index_name)
            existing.close()
        except (KeyError, Exception):
            pass

        # Open the flow and setup backend resources
        # NOTE: flow.setup() (sync) must be used here. flow.setup_async() does NOT
        # create the PostgreSQL table correctly when called from within an async context.
        flow = cocoindex.open_flow(index_name, flow_def)
        flow.setup()

        # Run the flow using FlowLiveUpdater
        updater = cocoindex.FlowLiveUpdater(flow)
        await updater.start_async()
        await updater.wait_async()

        # Store reference to the flow
        self._flows[index_name] = {
            "flow": flow,
            "updater": updater,
            "repo_path": repo_path,
        }

        # Get stats
        file_count = await self._count_files(index_name)
        chunk_count = await self._count_chunks(index_name)

        return IndexInfo(
            name=index_name,
            repository_path=repo_path,
            file_count=file_count,
            chunk_count=chunk_count,
        )

    async def _count_files(self, index_name: str) -> int:
        """Count distinct files in an index."""
        try:
            with psycopg.connect(self.config.postgres.connection_string) as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT COUNT(DISTINCT file_path) FROM code_index_{index_name}")
                    result = cur.fetchone()
                    return result[0] if result else 0
        except psycopg.Error:
            return 0

    async def _count_chunks(self, index_name: str) -> int:
        """Count chunks in an index."""
        try:
            with psycopg.connect(self.config.postgres.connection_string) as conn:
                with conn.cursor() as cur:
                    cur.execute(f"SELECT COUNT(*) FROM code_index_{index_name}")
                    result = cur.fetchone()
                    return result[0] if result else 0
        except psycopg.Error:
            return 0

    async def search(
        self,
        query: str,
        index_name: str | None = None,
        limit: int = 10,
    ) -> list[SearchResult]:
        """
        Search for code semantically.

        Args:
            query: Natural language search query
            index_name: Specific index to search (or None for all)
            limit: Maximum number of results

        Returns:
            List of search results
        """
        if not self._initialized:
            await self.initialize()

        results: list[SearchResult] = []

        # Generate query embedding using the transform flow
        query_embedding = await text_to_embedding.eval_async(query)
        # Convert to list if needed (eval_async may return list or ndarray)
        if hasattr(query_embedding, "tolist"):
            query_embedding = query_embedding.tolist()

        # Search in postgres using pgvector
        with psycopg.connect(self.config.postgres.connection_string) as conn:
            register_vector(conn)

            with conn.cursor() as cur:
                # Find tables to search
                if index_name:
                    tables = [f"code_index_{index_name}"]
                else:
                    cur.execute(
                        """
                        SELECT table_name FROM information_schema.tables
                        WHERE table_name LIKE 'code_index_%'
                        AND table_schema = 'public'
                    """
                    )
                    tables = [row[0] for row in cur.fetchall()]

                for table in tables:
                    try:
                        # Semantic search using pgvector cosine similarity
                        cur.execute(
                            f"""
                            SELECT file_path, content,
                                   1 - (embedding <=> %s::vector) as score
                            FROM {table}
                            ORDER BY embedding <=> %s::vector
                            LIMIT %s
                        """,
                            (query_embedding, query_embedding, limit),
                        )

                        for row in cur.fetchall():
                            results.append(
                                SearchResult(
                                    file_path=row[0],
                                    content=row[1],
                                    score=float(row[2]),
                                )
                            )
                    except (psycopg.Error, ValueError) as e:
                        logger.warning("Failed to search table %s: %s", table, e)

        # Sort by score and limit
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:limit]

    async def list_indexes(self) -> list[IndexInfo]:
        """List all available indexes."""
        if not self._initialized:
            await self.initialize()

        indexes: list[IndexInfo] = []

        try:
            with psycopg.connect(self.config.postgres.connection_string) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT table_name FROM information_schema.tables
                        WHERE table_name LIKE 'code_index_%'
                        AND table_schema = 'public'
                    """
                    )
                    tables = [row[0] for row in cur.fetchall()]

                    for table in tables:
                        # Extract index name from table name
                        name = table.replace("code_index_", "")

                        # Get chunk count
                        cur.execute(f"SELECT COUNT(*) FROM {table}")
                        chunk_count = cur.fetchone()[0]

                        # Get file count
                        cur.execute(f"SELECT COUNT(DISTINCT file_path) FROM {table}")
                        file_count = cur.fetchone()[0]

                        indexes.append(
                            IndexInfo(
                                name=name,
                                repository_path=self._flows.get(name, {}).get(
                                    "repo_path", "unknown"
                                ),
                                file_count=file_count,
                                chunk_count=chunk_count,
                            )
                        )
        except psycopg.Error as e:
            logger.error("Failed to list indexes: %s", e)

        return indexes

    async def get_index(self, index_name: str) -> IndexInfo | None:
        """Get information about a specific index."""
        indexes = await self.list_indexes()
        for idx in indexes:
            if idx.name == index_name:
                return idx
        return None

    async def delete_index(self, index_name: str) -> bool:
        """Delete an index."""
        if not self._initialized:
            await self.initialize()

        try:
            # Get the flow object from instance registry or global registry
            flow = None
            if index_name in self._flows:
                flow = self._flows[index_name].get("flow")
                del self._flows[index_name]

            if flow is None:
                # Try global cocoindex registry
                try:
                    flow = _cocoindex_flow.flow_by_name(index_name)
                except KeyError:
                    pass

            if flow is None:
                # No live flow — need to reconstruct one to drop via cocoindex
                # Fallback: manually drop the tables
                table_name = f"code_index_{index_name}"
                tracking_table = f"{index_name}__cocoindex_tracking"
                with psycopg.connect(self.config.postgres.connection_string) as conn:
                    with conn.cursor() as cur:
                        cur.execute(f"DROP TABLE IF EXISTS {table_name}")
                        cur.execute(f"DROP TABLE IF EXISTS {tracking_table}")
                        cur.execute(
                            "DELETE FROM cocoindex_setup_metadata WHERE flow_name = %s",
                            (index_name,),
                        )
                    conn.commit()
            else:
                # Use cocoindex's drop() to properly clean up tables AND metadata
                flow.drop()
                flow.close()

            logger.info("Deleted index: %s", index_name)
            return True
        except Exception as e:
            logger.error("Failed to delete index %s: %s", index_name, e)
            return False

    # =========================================================================
    # Symbol Indexing Methods (Tree-sitter based structural analysis)
    # =========================================================================

    async def index_symbols(
        self,
        repo_path: str,
        include_patterns: list[str] | None = None,
        exclude_patterns: list[str] | None = None,
        use_llm: bool = False,
    ) -> SymbolIndexInfo:
        """
        Index a repository for structural symbol analysis using tree-sitter.

        Args:
            repo_path: Path to the repository to index
            include_patterns: Glob patterns for files to include
            exclude_patterns: Glob patterns for files to exclude
            use_llm: Whether to use LLM for languages without tree-sitter support

        Returns:
            Information about the created symbol index
        """
        if not self._initialized:
            await self.initialize()

        repo_path = str(Path(repo_path).resolve())
        index_name = self._get_index_name(repo_path)

        # Default patterns for code files
        if include_patterns is None:
            include_patterns = [
                "**/*.py",
                "**/*.js",
                "**/*.ts",
                "**/*.tsx",
                "**/*.jsx",
                "**/*.go",
                "**/*.rs",
                "**/*.java",
                "**/*.c",
                "**/*.cpp",
                "**/*.h",
                "**/*.hpp",
                "**/*.cs",
                "**/*.rb",
                "**/*.php",
                "**/*.swift",
                "**/*.kt",
                "**/*.scala",
            ]

        if exclude_patterns is None:
            exclude_patterns = [
                "**/node_modules/**",
                "**/.git/**",
                "**/vendor/**",
                "**/dist/**",
                "**/build/**",
                "**/__pycache__/**",
                "**/.venv/**",
                "**/venv/**",
            ]

        logger.info("Indexing symbols for repository: %s as %s", repo_path, index_name)

        # Create the symbol index table directly (not using CocoIndex flow for now
        # because tree-sitter parsing needs to happen in Python, not Rust engine)
        symbol_table = f"symbol_index_{index_name}"

        with psycopg.connect(self.config.postgres.connection_string) as conn:
            with conn.cursor() as cur:
                # Create the symbol table
                cur.execute(
                    sql.SQL(
                        """
                    CREATE TABLE IF NOT EXISTS {} (
                        file_path TEXT NOT NULL,
                        symbol_name TEXT NOT NULL,
                        symbol_kind TEXT NOT NULL,
                        category TEXT NOT NULL,
                        line INTEGER NOT NULL,
                        col INTEGER NOT NULL,
                        end_line INTEGER NOT NULL,
                        end_col INTEGER NOT NULL,
                        context TEXT,
                        parent_name TEXT,
                        parent_kind TEXT,
                        PRIMARY KEY (file_path, line, col, symbol_name)
                    )
                """
                    ).format(sql.Identifier(symbol_table))
                )

                # Create indexes for fast lookups
                cur.execute(
                    sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (symbol_name)").format(
                        sql.Identifier(f"idx_{symbol_table}_name"),
                        sql.Identifier(symbol_table),
                    )
                )
                cur.execute(
                    sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (symbol_kind, category)").format(
                        sql.Identifier(f"idx_{symbol_table}_kind"),
                        sql.Identifier(symbol_table),
                    )
                )
                cur.execute(
                    sql.SQL("CREATE INDEX IF NOT EXISTS {} ON {} (file_path)").format(
                        sql.Identifier(f"idx_{symbol_table}_file"),
                        sql.Identifier(symbol_table),
                    )
                )

                # Clear existing data for this index
                cur.execute(sql.SQL("DELETE FROM {}").format(sql.Identifier(symbol_table)))

            conn.commit()

        # Process files
        file_count = 0
        symbol_count = 0
        definition_count = 0
        reference_count = 0

        # Find all matching files
        import fnmatch
        from pathlib import Path as PathLib

        repo_root = PathLib(repo_path)
        all_files: list[PathLib] = []

        for pattern in include_patterns:
            all_files.extend(repo_root.glob(pattern))

        # Filter out excluded files
        filtered_files: list[PathLib] = []
        for file_path in all_files:
            rel_path = str(file_path.relative_to(repo_root))
            excluded = False
            for exclude_pattern in exclude_patterns:
                if fnmatch.fnmatch(rel_path, exclude_pattern):
                    excluded = True
                    break
            if not excluded and file_path.is_file():
                filtered_files.append(file_path)

        # Process each file
        symbols_batch: list[tuple[Any, ...]] = []
        batch_size = 100

        for file_path in filtered_files:
            try:
                content = file_path.read_text(encoding="utf-8", errors="replace")
                rel_path = str(file_path.relative_to(repo_root))

                # Check if language is supported
                language = detect_language(rel_path)

                if has_tree_sitter_support(language):
                    # Use tree-sitter extraction
                    symbols = extract_symbols_from_file(content, rel_path)
                elif use_llm and self.config.llm.is_configured:
                    # LLM fallback (to be implemented)
                    # For now, skip
                    logger.debug("LLM extraction not yet implemented for %s", rel_path)
                    symbols = []
                else:
                    symbols = []

                if symbols:
                    file_count += 1
                    for sym in symbols:
                        symbol_count += 1
                        if sym.category == "definition":
                            definition_count += 1
                        else:
                            reference_count += 1

                        symbols_batch.append(
                            (
                                rel_path,
                                sym.name,
                                sym.kind,
                                sym.category,
                                sym.line,
                                sym.column,
                                sym.end_line,
                                sym.end_column,
                                sym.context[:500] if sym.context else None,  # Truncate context
                                sym.parent_name,
                                sym.parent_kind,
                            )
                        )

                        if len(symbols_batch) >= batch_size:
                            self._insert_symbols_batch(symbol_table, symbols_batch)
                            symbols_batch = []

            except Exception as e:  # noqa: BLE001
                logger.warning("Failed to process %s: %s", file_path, e)

        # Insert remaining symbols
        if symbols_batch:
            self._insert_symbols_batch(symbol_table, symbols_batch)

        # Store metadata about this symbol index
        self._flows[f"symbols_{index_name}"] = {
            "type": "symbol_index",
            "repo_path": repo_path,
            "table": symbol_table,
        }

        return SymbolIndexInfo(
            name=index_name,
            repository_path=repo_path,
            file_count=file_count,
            symbol_count=symbol_count,
            definition_count=definition_count,
            reference_count=reference_count,
        )

    def _insert_symbols_batch(self, table_name: str, symbols: list[tuple[Any, ...]]) -> None:
        """Insert a batch of symbols into the database."""
        if not symbols:
            return

        with psycopg.connect(self.config.postgres.connection_string) as conn:
            with conn.cursor() as cur:
                cur.executemany(
                    sql.SQL(
                        """
                        INSERT INTO {} (
                            file_path, symbol_name, symbol_kind, category,
                            line, col, end_line, end_col,
                            context, parent_name, parent_kind
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (file_path, line, col, symbol_name) DO UPDATE SET
                            symbol_kind = EXCLUDED.symbol_kind,
                            category = EXCLUDED.category,
                            context = EXCLUDED.context,
                            parent_name = EXCLUDED.parent_name,
                            parent_kind = EXCLUDED.parent_kind
                    """
                    ).format(sql.Identifier(table_name)),
                    symbols,
                )
            conn.commit()

    async def find_usages(
        self,
        symbol_name: str,
        index_name: str | None = None,
        kind: str | None = None,
        category: str | None = None,
        pattern: bool = False,
        limit: int = 100,
    ) -> list[SymbolSearchResult]:
        """
        Find all usages of a symbol.

        Args:
            symbol_name: Symbol name to search for (exact match or LIKE pattern if pattern=True)
            index_name: Specific index to search (or None for all)
            kind: Filter by symbol kind (class, function, method, variable, attribute, etc.)
            category: Filter by category (definition, reference)
            pattern: If True, treat symbol_name as a SQL LIKE pattern (use % for wildcard)
            limit: Maximum number of results

        Returns:
            List of symbol search results
        """
        if not self._initialized:
            await self.initialize()

        results: list[SymbolSearchResult] = []

        with psycopg.connect(self.config.postgres.connection_string) as conn:
            with conn.cursor() as cur:
                # Find symbol tables to search
                if index_name:
                    tables = [f"symbol_index_{index_name}"]
                else:
                    cur.execute(
                        """
                        SELECT table_name FROM information_schema.tables
                        WHERE table_name LIKE 'symbol_index_%'
                        AND table_schema = 'public'
                    """
                    )
                    tables = [row[0] for row in cur.fetchall()]

                for table in tables:
                    try:
                        # Build query with filters
                        conditions = []
                        params: list[Any] = []

                        if pattern:
                            conditions.append("symbol_name LIKE %s")
                        else:
                            conditions.append("symbol_name = %s")
                        params.append(symbol_name)

                        if kind:
                            conditions.append("symbol_kind = %s")
                            params.append(kind)

                        if category:
                            conditions.append("category = %s")
                            params.append(category)

                        where_clause = " AND ".join(conditions)
                        params.append(limit)

                        query = sql.SQL(
                            """
                            SELECT file_path, symbol_name, symbol_kind, category,
                                   line, col, context, parent_name
                            FROM {}
                            WHERE {}
                            ORDER BY file_path, line
                            LIMIT %s
                        """
                        ).format(
                            sql.Identifier(table),
                            sql.SQL(where_clause),
                        )

                        cur.execute(query, params)

                        for row in cur.fetchall():
                            results.append(
                                SymbolSearchResult(
                                    file_path=row[0],
                                    symbol_name=row[1],
                                    symbol_kind=row[2],
                                    category=row[3],
                                    line=row[4],
                                    column=row[5],
                                    context=row[6] or "",
                                    parent_name=row[7],
                                )
                            )
                    except psycopg.Error as e:
                        logger.warning("Failed to search table %s: %s", table, e)

        return results[:limit]

    async def find_definitions(
        self,
        symbol_name: str,
        index_name: str | None = None,
        kind: str | None = None,
        limit: int = 50,
    ) -> list[SymbolSearchResult]:
        """
        Find where a symbol is defined.

        Args:
            symbol_name: Symbol name to search for
            index_name: Specific index to search (or None for all)
            kind: Filter by symbol kind (class, function, method, etc.)
            limit: Maximum number of results

        Returns:
            List of definition locations
        """
        return await self.find_usages(
            symbol_name=symbol_name,
            index_name=index_name,
            kind=kind,
            category="definition",
            limit=limit,
        )

    async def list_symbols(
        self,
        index_name: str | None = None,
        file_path: str | None = None,
        kind: str | None = None,
        pattern: str | None = None,
        limit: int = 100,
    ) -> list[dict[str, Any]]:
        """
        List symbols in an index with aggregated counts.

        Args:
            index_name: Specific index to search (or None for all)
            file_path: Filter by file path
            kind: Filter by symbol kind
            pattern: LIKE pattern for symbol names
            limit: Maximum number of results

        Returns:
            List of symbol summaries with counts
        """
        if not self._initialized:
            await self.initialize()

        results: list[dict[str, Any]] = []

        with psycopg.connect(self.config.postgres.connection_string) as conn:
            with conn.cursor() as cur:
                # Find symbol tables
                if index_name:
                    tables = [f"symbol_index_{index_name}"]
                else:
                    cur.execute(
                        """
                        SELECT table_name FROM information_schema.tables
                        WHERE table_name LIKE 'symbol_index_%'
                        AND table_schema = 'public'
                    """
                    )
                    tables = [row[0] for row in cur.fetchall()]

                for table in tables:
                    try:
                        conditions = []
                        params: list[Any] = []

                        if file_path:
                            conditions.append("file_path = %s")
                            params.append(file_path)

                        if kind:
                            conditions.append("symbol_kind = %s")
                            params.append(kind)

                        if pattern:
                            conditions.append("symbol_name LIKE %s")
                            params.append(pattern)

                        where_clause = "WHERE " + " AND ".join(conditions) if conditions else ""
                        params.append(limit)

                        query = sql.SQL(
                            """
                            SELECT symbol_name, symbol_kind,
                                   COUNT(*) as total_count,
                                   SUM(CASE WHEN category = 'definition'
                                       THEN 1 ELSE 0 END) as def_count,
                                   SUM(CASE WHEN category = 'reference'
                                       THEN 1 ELSE 0 END) as ref_count
                            FROM {} {}
                            GROUP BY symbol_name, symbol_kind
                            ORDER BY total_count DESC
                            LIMIT %s
                        """
                        ).format(
                            sql.Identifier(table),
                            sql.SQL(where_clause),
                        )

                        cur.execute(query, params)

                        for row in cur.fetchall():
                            results.append(
                                {
                                    "symbol_name": row[0],
                                    "symbol_kind": row[1],
                                    "total_count": row[2],
                                    "definition_count": row[3],
                                    "reference_count": row[4],
                                    "index_name": table.replace("symbol_index_", ""),
                                }
                            )
                    except psycopg.Error as e:
                        logger.warning("Failed to list symbols from %s: %s", table, e)

        # Sort by total count and limit
        results.sort(key=lambda x: x["total_count"], reverse=True)
        return results[:limit]

    async def get_symbol_index(self, index_name: str) -> SymbolIndexInfo | None:
        """Get information about a specific symbol index."""
        if not self._initialized:
            await self.initialize()

        table_name = f"symbol_index_{index_name}"

        try:
            with psycopg.connect(self.config.postgres.connection_string) as conn:
                with conn.cursor() as cur:
                    # Check if table exists
                    cur.execute(
                        """
                        SELECT EXISTS (
                            SELECT FROM information_schema.tables
                            WHERE table_name = %s AND table_schema = 'public'
                        )
                    """,
                        (table_name,),
                    )
                    exists = cur.fetchone()[0]  # type: ignore[index]
                    if not exists:
                        return None

                    # Get counts
                    cur.execute(
                        sql.SQL("SELECT COUNT(*) FROM {}").format(sql.Identifier(table_name))
                    )
                    symbol_count = cur.fetchone()[0]  # type: ignore[index]

                    cur.execute(
                        sql.SQL("SELECT COUNT(DISTINCT file_path) FROM {}").format(
                            sql.Identifier(table_name)
                        )
                    )
                    file_count = cur.fetchone()[0]  # type: ignore[index]

                    cur.execute(
                        sql.SQL("SELECT COUNT(*) FROM {} WHERE category = 'definition'").format(
                            sql.Identifier(table_name)
                        )
                    )
                    definition_count = cur.fetchone()[0]  # type: ignore[index]

                    cur.execute(
                        sql.SQL("SELECT COUNT(*) FROM {} WHERE category = 'reference'").format(
                            sql.Identifier(table_name)
                        )
                    )
                    reference_count = cur.fetchone()[0]  # type: ignore[index]

                    # Get repo path from metadata
                    repo_path = self._flows.get(f"symbols_{index_name}", {}).get(
                        "repo_path", "unknown"
                    )

                    return SymbolIndexInfo(
                        name=index_name,
                        repository_path=repo_path,
                        file_count=file_count,
                        symbol_count=symbol_count,
                        definition_count=definition_count,
                        reference_count=reference_count,
                    )
        except psycopg.Error as e:
            logger.error("Failed to get symbol index info: %s", e)
            return None

    async def list_symbol_indexes(self) -> list[SymbolIndexInfo]:
        """List all available symbol indexes."""
        if not self._initialized:
            await self.initialize()

        indexes: list[SymbolIndexInfo] = []

        try:
            with psycopg.connect(self.config.postgres.connection_string) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        """
                        SELECT table_name FROM information_schema.tables
                        WHERE table_name LIKE 'symbol_index_%'
                        AND table_schema = 'public'
                    """
                    )
                    tables = [row[0] for row in cur.fetchall()]

                    for table in tables:
                        name = table.replace("symbol_index_", "")
                        info = await self.get_symbol_index(name)
                        if info:
                            indexes.append(info)
        except psycopg.Error as e:
            logger.error("Failed to list symbol indexes: %s", e)

        return indexes

    async def delete_symbol_index(self, index_name: str) -> bool:
        """Delete a symbol index."""
        if not self._initialized:
            await self.initialize()

        table_name = f"symbol_index_{index_name}"

        try:
            with psycopg.connect(self.config.postgres.connection_string) as conn:
                with conn.cursor() as cur:
                    cur.execute(
                        sql.SQL("DROP TABLE IF EXISTS {}").format(sql.Identifier(table_name))
                    )
                conn.commit()

            # Remove from flow registry
            if f"symbols_{index_name}" in self._flows:
                del self._flows[f"symbols_{index_name}"]

            logger.info("Deleted symbol index: %s", index_name)
            return True
        except psycopg.Error as e:
            logger.error("Failed to delete symbol index %s: %s", index_name, e)
            return False
