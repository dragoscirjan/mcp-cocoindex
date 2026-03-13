"""Tests for Claude Code MCP integration.

These tests verify that the MCP server integrates correctly with Claude Code
(claude.ai code editor), by testing the MCP protocol interface that
Claude Code would use.

These tests require:
1. PostgreSQL with pgvector running (task start)
2. This repo indexed (task index)
3. Environment variables set (or defaults used)

Run with: uv run pytest tests/test_claudecode.py -v
"""
# pylint: disable=redefined-outer-name,protected-access,import-outside-toplevel
# pylint: disable=broad-exception-caught,unused-argument

from __future__ import annotations

import json
from collections.abc import AsyncGenerator
from pathlib import Path

import psycopg
import pytest

from mcp_coco_index.config import CocoIndexConfig
from mcp_coco_index.indexer import CocoIndexer


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

    # Index all code files from this repo
    result = await indexer.index_repository(
        repo_path=THIS_REPO,
        include_patterns=["**/*.py", "**/*.md", "**/*.yml", "**/*.yaml", "**/*.toml"],
        exclude_patterns=["**/__pycache__/**", "**/.venv/**", "**/dist/**"],
    )

    index_name = result.name
    yield indexer, index_name

    # Cleanup after module tests complete
    await indexer.delete_index(index_name)
    await indexer.close()


class TestClaudeCodeMCPTools:
    """Tests simulating how Claude Code uses the MCP server tools.

    Claude Code uses MCP tools to build context about a codebase and
    answer questions about code structure, patterns, and implementation.
    These tests verify the tool interface works for Claude Code workflows.
    """

    @pytest.mark.asyncio
    async def test_tools_have_descriptions(self) -> None:
        """Test that all tools have proper descriptions for Claude Code.

        Claude Code uses tool descriptions to decide which tools to call.
        Each tool must have a clear, informative description.
        """
        from mcp_coco_index.server import list_tools

        tools = await list_tools()

        for tool in tools:
            assert tool.name, "Tool must have a name"
            assert tool.description, f"Tool {tool.name} must have a description"
            assert len(tool.description) > 10, f"Tool {tool.name} description too short"
            assert tool.inputSchema, f"Tool {tool.name} must have an input schema"

    @pytest.mark.asyncio
    async def test_tools_have_valid_schemas(self) -> None:
        """Test that all tools have valid JSON schemas for Claude Code.

        Claude Code parses the inputSchema to understand tool parameters.
        """
        from mcp_coco_index.server import list_tools

        tools = await list_tools()

        for tool in tools:
            schema = tool.inputSchema
            assert schema.get("type") == "object", f"Tool {tool.name} schema must be object type"
            assert "properties" in schema, f"Tool {tool.name} schema must have properties"

    @pytest.mark.asyncio
    async def test_index_and_search_workflow(self, config: CocoIndexConfig) -> None:
        """Test the full index-then-search workflow as Claude Code would use it.

        Claude Code workflow:
        1. Index the current project
        2. Search for relevant code when answering user questions
        3. Use results to provide contextual answers
        """
        from mcp_coco_index.server import call_tool, get_indexer

        indexer = get_indexer()
        index_name = None
        try:
            # Step 1: Index this repository
            index_result = await call_tool(
                "index_repository",
                {
                    "path": THIS_REPO,
                    "include_patterns": ["**/*.py"],
                    "exclude_patterns": ["**/__pycache__/**", "**/.venv/**"],
                },
            )

            assert index_result.isError is not True
            index_data = json.loads(index_result.content[0].text)
            assert index_data["success"] is True
            index_name = index_data["index_name"]

            # Step 2: Search for code relevant to a user's question
            search_result = await call_tool(
                "search_code",
                {
                    "query": "how does the MCP server handle tool calls",
                    "index_name": index_name,
                    "limit": 5,
                },
            )

            assert search_result.isError is not True
            search_data = json.loads(search_result.content[0].text)
            results = search_data["results"]

            # Step 3: Verify we got useful context
            assert len(results) >= 1
            # Should have found server.py
            assert any("server" in r["file_path"] for r in results)

        finally:
            if index_name:
                await indexer.delete_index(index_name)

    @pytest.mark.asyncio
    async def test_search_for_test_patterns(self, indexed_repo: tuple[CocoIndexer, str]) -> None:
        """Test searching for test patterns in the codebase.

        Claude Code would search for 'how are tests written' to help
        a user add new tests following existing patterns.
        """
        from mcp_coco_index.server import call_tool

        indexer, index_name = indexed_repo

        result = await call_tool(
            "search_code",
            {
                "query": "pytest fixtures and test setup patterns",
                "index_name": index_name,
                "limit": 5,
            },
        )

        assert result.isError is not True
        data = json.loads(result.content[0].text)
        results = data["results"]

        # Should find test files
        assert len(results) >= 1
        found_test = any("test" in r["file_path"] for r in results)
        assert found_test, "Should find test files"

    @pytest.mark.asyncio
    async def test_search_for_error_handling(self, indexed_repo: tuple[CocoIndexer, str]) -> None:
        """Test searching for error handling patterns.

        Claude Code would search for 'error handling' to explain
        how the codebase handles exceptions.
        """
        from mcp_coco_index.server import call_tool

        indexer, index_name = indexed_repo

        result = await call_tool(
            "search_code",
            {
                "query": "exception handling and error logging",
                "index_name": index_name,
                "limit": 5,
            },
        )

        assert result.isError is not True
        data = json.loads(result.content[0].text)
        results = data["results"]
        assert isinstance(results, list)

        # Should find error handling code
        if len(results) > 0:
            found_error_handling = any(
                "except" in r["content"] or "logger" in r["content"] for r in results
            )
            assert found_error_handling, "Should find error handling code"

    @pytest.mark.asyncio
    async def test_search_returns_file_paths(self, indexed_repo: tuple[CocoIndexer, str]) -> None:
        """Test that search results include absolute file paths.

        Claude Code needs absolute file paths to open files or
        show relevant code snippets to the user.
        """
        from mcp_coco_index.server import call_tool

        indexer, index_name = indexed_repo

        result = await call_tool(
            "search_code",
            {
                "query": "CocoIndex configuration settings",
                "index_name": index_name,
                "limit": 3,
            },
        )

        assert result.isError is not True
        data = json.loads(result.content[0].text)
        results = data["results"]

        for r in results:
            assert r["file_path"], "File path must not be empty"
            # File paths should be valid
            path = Path(r["file_path"])
            assert path.suffix in {".py", ".md", ".yml", ".yaml", ".toml"}, (
                f"Unexpected file type: {path.suffix}"
            )

    @pytest.mark.asyncio
    async def test_search_scores_are_meaningful(
        self, indexed_repo: tuple[CocoIndexer, str]
    ) -> None:
        """Test that similarity scores are semantically meaningful.

        Claude Code uses scores to rank and filter results.
        Higher scores should indicate better semantic matches.
        """
        from mcp_coco_index.server import call_tool

        indexer, index_name = indexed_repo

        # Search with a very specific query that should match config.py well
        result = await call_tool(
            "search_code",
            {
                "query": "COCOINDEX_DB_HOST COCOINDEX_DB_PORT environment variable",
                "index_name": index_name,
                "limit": 5,
            },
        )

        assert result.isError is not True
        data = json.loads(result.content[0].text)
        results = data["results"]

        if len(results) >= 2:
            # Results should be sorted by descending score
            scores = [r["score"] for r in results]
            assert scores == sorted(scores, reverse=True), "Results should be sorted by score"

    @pytest.mark.asyncio
    async def test_multiple_searches_in_session(
        self, indexed_repo: tuple[CocoIndexer, str]
    ) -> None:
        """Test multiple sequential searches as Claude Code would do in a session.

        During a conversation, Claude Code may search multiple times for
        different aspects of the codebase to answer a complex question.
        """
        from mcp_coco_index.server import call_tool

        indexer, index_name = indexed_repo

        queries = [
            "how to initialize the MCP server",
            "what tools does the server expose",
            "how are search results ranked and returned",
        ]

        for query in queries:
            result = await call_tool(
                "search_code",
                {
                    "query": query,
                    "index_name": index_name,
                    "limit": 3,
                },
            )

            assert result.isError is not True, f"Search failed for query: {query}"
            data = json.loads(result.content[0].text)
            assert "results" in data
            assert isinstance(data["results"], list)

    @pytest.mark.asyncio
    async def test_index_then_delete_workflow(self, config: CocoIndexConfig) -> None:
        """Test the index-then-delete workflow for cleanup.

        Claude Code or its users may want to clean up indexes that
        are no longer needed.
        """
        from mcp_coco_index.server import call_tool, get_indexer

        indexer = get_indexer()
        index_name = None

        # Index
        index_result = await call_tool(
            "index_repository",
            {
                "path": THIS_REPO,
                "include_patterns": ["**/*.py"],
                "exclude_patterns": ["**/__pycache__/**", "**/.venv/**"],
            },
        )

        assert index_result.isError is not True
        index_data = json.loads(index_result.content[0].text)
        index_name = index_data["index_name"]

        # Verify it exists
        get_result = await call_tool("get_index", {"name": index_name})
        assert get_result.isError is not True

        # Delete
        del_result = await call_tool("delete_index", {"name": index_name})
        assert del_result.isError is not True
        del_data = json.loads(del_result.content[0].text)
        assert del_data["success"] is True

        # Verify it's gone
        get_after_del = await call_tool("get_index", {"name": index_name})
        assert get_after_del.isError is True
