"""Tests for OpenCode MCP integration.

These tests verify that the MCP server integrates correctly with OpenCode,
by testing the MCP protocol interface that OpenCode would use.

These tests require:
1. PostgreSQL with pgvector running (task start)
2. This repo indexed (task index)
3. Environment variables set (or defaults used)

Run with: uv run pytest tests/test_opencode.py -v
"""
# pylint: disable=redefined-outer-name,protected-access,import-outside-toplevel
# pylint: disable=broad-exception-caught,unused-argument

from __future__ import annotations

import json
import os
from collections.abc import AsyncGenerator
from pathlib import Path

import psycopg
import pytest

from mcp_coco_index.config import CocoIndexConfig
from mcp_coco_index.indexer import CocoIndexer, IndexInfo


# Skip all tests if PostgreSQL is not available
def postgres_available() -> bool:
    """Check if PostgreSQL is available."""
    try:
        config = CocoIndexConfig.from_env()
        with psycopg.connect(config.postgres.connection_string, connect_timeout=5):
            return True
    except (psycopg.Error, OSError):
        return False


pytestmark = pytest.mark.skipif(
    not postgres_available(),
    reason="PostgreSQL not available. Start with: task start",
)

# The repo to index for tests - this repo itself
THIS_REPO = str(Path(__file__).parent.parent.resolve())


@pytest.fixture(scope="module")
def config() -> CocoIndexConfig:
    """Create test configuration."""
    return CocoIndexConfig.from_env()


@pytest.fixture(scope="module")
async def indexed_repo(config: CocoIndexConfig) -> AsyncGenerator[tuple[CocoIndexer, str], None]:
    """Index this repository and return the indexer and index name."""
    indexer = CocoIndexer(config)
    await indexer.initialize()

    # Index only Python files from this repo for speed
    result = await indexer.index_repository(
        repo_path=THIS_REPO,
        include_patterns=["**/*.py"],
        exclude_patterns=["**/__pycache__/**", "**/.venv/**", "**/dist/**"],
    )

    index_name = result.name
    yield indexer, index_name

    # Cleanup after module tests complete
    await indexer.delete_index(index_name)
    await indexer.close()


class TestOpenCodeMCPTools:
    """Tests simulating how OpenCode uses the MCP server tools.

    OpenCode uses MCP tools to index codebases and search for relevant
    code when answering questions. These tests verify the tool interface
    works as expected for the OpenCode use case.
    """

    @pytest.mark.asyncio
    async def test_list_tools_available(self) -> None:
        """Test that all expected MCP tools are registered and available.

        OpenCode needs these tools to be discoverable via MCP protocol.
        """
        from mcp_coco_index.server import list_tools

        tools = await list_tools()

        tool_names = {t.name for t in tools}
        assert "index_repository" in tool_names, "index_repository tool must be available"
        assert "search_code" in tool_names, "search_code tool must be available"
        assert "list_indexes" in tool_names, "list_indexes tool must be available"
        assert "get_index" in tool_names, "get_index tool must be available"
        assert "delete_index" in tool_names, "delete_index tool must be available"

    @pytest.mark.asyncio
    async def test_index_this_repo(self, config: CocoIndexConfig) -> None:
        """Test indexing this very repository (mcp-cocoindex).

        OpenCode would use this to index the current project for context.
        """
        from mcp_coco_index.server import call_tool, get_indexer

        indexer = get_indexer()
        try:
            result = await call_tool(
                "index_repository",
                {
                    "path": THIS_REPO,
                    "include_patterns": ["**/*.py"],
                    "exclude_patterns": ["**/__pycache__/**", "**/.venv/**"],
                },
            )

            assert result.isError is not True
            data = json.loads(result.content[0].text)
            assert data["success"] is True
            assert "index_name" in data
            assert data["file_count"] >= 1
            assert data["chunk_count"] >= 1

        finally:
            index_name = indexer._get_index_name(THIS_REPO)
            await indexer.delete_index(index_name)

    @pytest.mark.asyncio
    async def test_search_for_mcp_server_code(self, indexed_repo: tuple[CocoIndexer, str]) -> None:
        """Test searching for MCP server implementation code.

        This simulates OpenCode searching for 'how is the MCP server structured'
        within the indexed codebase.
        """
        from mcp_coco_index.server import call_tool

        indexer, index_name = indexed_repo

        result = await call_tool(
            "search_code",
            {
                "query": "MCP server tool registration and handling",
                "index_name": index_name,
                "limit": 5,
            },
        )

        assert result.isError is not True
        data = json.loads(result.content[0].text)
        assert "results" in data
        results = data["results"]
        assert isinstance(results, list)

        # Should find server.py content
        if len(results) > 0:
            found_server = any("server" in r["file_path"] for r in results)
            assert found_server, "Should find server-related code"

    @pytest.mark.asyncio
    async def test_search_for_config_code(self, indexed_repo: tuple[CocoIndexer, str]) -> None:
        """Test searching for configuration code.

        OpenCode would search for 'database configuration' to understand
        how the project connects to PostgreSQL.
        """
        from mcp_coco_index.server import call_tool

        indexer, index_name = indexed_repo

        result = await call_tool(
            "search_code",
            {
                "query": "PostgreSQL database connection configuration",
                "index_name": index_name,
                "limit": 5,
            },
        )

        assert result.isError is not True
        data = json.loads(result.content[0].text)
        results = data["results"]
        assert isinstance(results, list)
        assert len(results) >= 1

        # Should find config.py or indexer.py
        found_config = any(
            "config" in r["file_path"] or "indexer" in r["file_path"] for r in results
        )
        assert found_config, "Should find database config code"

    @pytest.mark.asyncio
    async def test_search_for_indexer_code(self, indexed_repo: tuple[CocoIndexer, str]) -> None:
        """Test searching for the indexer implementation.

        OpenCode would search for 'code indexing and embedding generation'
        to understand how files get indexed.
        """
        from mcp_coco_index.server import call_tool

        indexer, index_name = indexed_repo

        result = await call_tool(
            "search_code",
            {
                "query": "code indexing embedding generation vector search",
                "index_name": index_name,
                "limit": 5,
            },
        )

        assert result.isError is not True
        data = json.loads(result.content[0].text)
        results = data["results"]
        assert isinstance(results, list)
        assert len(results) >= 1

        # Should find indexer.py
        found_indexer = any("indexer" in r["file_path"] for r in results)
        assert found_indexer, "Should find indexer code"

    @pytest.mark.asyncio
    async def test_list_indexes_shows_this_repo(
        self, indexed_repo: tuple[CocoIndexer, str]
    ) -> None:
        """Test that list_indexes shows this repo after indexing.

        OpenCode can use list_indexes to discover what's already indexed.
        """
        from mcp_coco_index.server import call_tool

        indexer, index_name = indexed_repo

        result = await call_tool("list_indexes", {})

        assert result.isError is not True
        data = json.loads(result.content[0].text)
        assert "indexes" in data

        # Find our index
        our_index = next((idx for idx in data["indexes"] if idx["name"] == index_name), None)
        assert our_index is not None, f"Index {index_name} should be listed"
        assert our_index["file_count"] >= 1
        assert our_index["chunk_count"] >= 1

    @pytest.mark.asyncio
    async def test_get_index_details(self, indexed_repo: tuple[CocoIndexer, str]) -> None:
        """Test getting details about the indexed repo.

        OpenCode uses get_index to check if a repo is already indexed
        and get its statistics.
        """
        from mcp_coco_index.server import call_tool

        indexer, index_name = indexed_repo

        result = await call_tool("get_index", {"name": index_name})

        assert result.isError is not True
        data = json.loads(result.content[0].text)
        assert data["name"] == index_name
        assert data["file_count"] >= 1
        assert data["chunk_count"] >= 1

    @pytest.mark.asyncio
    async def test_search_result_structure(self, indexed_repo: tuple[CocoIndexer, str]) -> None:
        """Test that search results have the expected structure for OpenCode.

        OpenCode relies on the file_path, content, and score fields to
        present relevant code context to the user.
        """
        from mcp_coco_index.server import call_tool

        indexer, index_name = indexed_repo

        result = await call_tool(
            "search_code",
            {
                "query": "initialize and setup cocoindex",
                "index_name": index_name,
                "limit": 3,
            },
        )

        assert result.isError is not True
        data = json.loads(result.content[0].text)
        results = data["results"]

        for r in results:
            assert "file_path" in r, "Result must have file_path"
            assert "content" in r, "Result must have content"
            assert "score" in r, "Result must have score"
            assert isinstance(r["score"], float), "Score must be a float"
            assert 0 <= r["score"] <= 1, "Score must be between 0 and 1"
            assert len(r["content"]) > 0, "Content must not be empty"

    @pytest.mark.asyncio
    async def test_search_with_no_results(self, indexed_repo: tuple[CocoIndexer, str]) -> None:
        """Test search gracefully handles queries with no strong matches.

        OpenCode should handle gracefully when search doesn't find strong matches.
        """
        from mcp_coco_index.server import call_tool

        indexer, index_name = indexed_repo

        result = await call_tool(
            "search_code",
            {
                "query": "quantum computing neural interface blockchain",
                "index_name": index_name,
                "limit": 5,
            },
        )

        # Should not error, just return potentially low-scoring results
        assert result.isError is not True
        data = json.loads(result.content[0].text)
        assert "results" in data
        assert isinstance(data["results"], list)

    @pytest.mark.asyncio
    async def test_get_nonexistent_index_returns_error(self) -> None:
        """Test that get_index returns error for non-existent index.

        OpenCode needs to handle this error case to know when to re-index.
        """
        from mcp_coco_index.server import call_tool, get_indexer

        indexer = get_indexer()
        await indexer.initialize()

        result = await call_tool("get_index", {"name": "nonexistent_index_opencode_test"})

        assert result.isError is True
        data = json.loads(result.content[0].text)
        assert "error" in data
