"""MCP Server for CocoIndex - semantic code indexing and search."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolResult,
    TextContent,
    Tool,
)

from .config import CocoIndexConfig
from .indexer import CocoIndexer

logger = logging.getLogger(__name__)

# Create the MCP server
server = Server("mcp-coco-index")

# Global indexer instance (module-level singleton)
_indexer: CocoIndexer | None = None  # pylint: disable=invalid-name


def get_indexer() -> CocoIndexer:
    """Get or create the indexer instance."""
    global _indexer  # pylint: disable=global-statement
    if _indexer is None:
        config = CocoIndexConfig.from_env()
        _indexer = CocoIndexer(config)
    return _indexer


# Tool definitions
TOOLS = [
    Tool(
        name="index_repository",
        description=(
            "Index a code repository for semantic search. "
            "Creates embeddings for code chunks and stores them in a vector database."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the repository to index",
                },
                "include_patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Glob patterns for files to include (e.g., '**/*.py')",
                },
                "exclude_patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Glob patterns for files to exclude",
                },
            },
            "required": ["path"],
        },
    ),
    Tool(
        name="search_code",
        description=(
            "Search for code semantically using natural language. "
            "Returns relevant code snippets matching the query."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query (e.g., 'authentication function')",
                },
                "index_name": {
                    "type": "string",
                    "description": "Specific index to search (optional, searches all)",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 10)",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
    ),
    Tool(
        name="list_indexes",
        description="List all available code indexes with their statistics.",
        inputSchema={
            "type": "object",
            "properties": {},
        },
    ),
    Tool(
        name="get_index",
        description="Get detailed information about a specific index.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the index to get information about",
                },
            },
            "required": ["name"],
        },
    ),
    Tool(
        name="delete_index",
        description="Delete a code index and all its data.",
        inputSchema={
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Name of the index to delete",
                },
            },
            "required": ["name"],
        },
    ),
    # Symbol indexing tools
    Tool(
        name="index_symbols",
        description=(
            "Index symbols (functions, classes, variables) from a code repository "
            "using tree-sitter. Creates a searchable index of all code symbols "
            "with their definitions and usages."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "path": {
                    "type": "string",
                    "description": "Absolute path to the repository to index",
                },
                "include_patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Glob patterns for files to include (e.g., '**/*.py')",
                },
                "exclude_patterns": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Glob patterns for files to exclude",
                },
            },
            "required": ["path"],
        },
    ),
    Tool(
        name="find_usages",
        description=(
            "Find all usages of a symbol (function, class, variable) across indexed code. "
            "Returns locations where the symbol is referenced or called."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol_name": {
                    "type": "string",
                    "description": "Name of the symbol to find usages for",
                },
                "index_name": {
                    "type": "string",
                    "description": "Specific symbol index to search (optional, searches all)",
                },
                "exact_match": {
                    "type": "boolean",
                    "description": (
                        "If true, only exact matches; "
                        "if false, includes partial matches (default: true)"
                    ),
                    "default": True,
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return (default: 50)",
                    "default": 50,
                },
            },
            "required": ["symbol_name"],
        },
    ),
    Tool(
        name="find_definitions",
        description=(
            "Find where a symbol (function, class, variable) is defined in indexed code. "
            "Returns the definition location(s) of the symbol."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "symbol_name": {
                    "type": "string",
                    "description": "Name of the symbol to find the definition for",
                },
                "index_name": {
                    "type": "string",
                    "description": "Specific symbol index to search (optional, searches all)",
                },
                "exact_match": {
                    "type": "boolean",
                    "description": (
                        "If true, only exact matches; "
                        "if false, includes partial matches (default: true)"
                    ),
                    "default": True,
                },
            },
            "required": ["symbol_name"],
        },
    ),
    Tool(
        name="list_symbols",
        description=(
            "List all unique symbols found in indexed code. "
            "Useful for exploring what functions, classes, and variables exist in a codebase."
        ),
        inputSchema={
            "type": "object",
            "properties": {
                "index_name": {
                    "type": "string",
                    "description": "Specific symbol index to query (optional, queries all)",
                },
                "symbol_kind": {
                    "type": "string",
                    "description": (
                        "Filter by symbol kind (e.g., 'function', 'class', 'variable', 'method')"
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of symbols to return (default: 100)",
                    "default": 100,
                },
            },
        },
    ),
]


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return TOOLS


@server.call_tool()
async def call_tool(  # pylint: disable=too-many-return-statements
    name: str, arguments: dict[str, Any]
) -> CallToolResult:
    """Handle tool calls."""
    try:
        indexer = get_indexer()

        if name == "index_repository":
            result = await indexer.index_repository(
                repo_path=arguments["path"],
                include_patterns=arguments.get("include_patterns"),
                exclude_patterns=arguments.get("exclude_patterns"),
            )
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "success": True,
                                "index_name": result.name,
                                "repository_path": result.repository_path,
                                "file_count": result.file_count,
                                "chunk_count": result.chunk_count,
                            },
                            indent=2,
                        ),
                    )
                ],
            )

        if name == "search_code":
            results = await indexer.search(
                query=arguments["query"],
                index_name=arguments.get("index_name"),
                limit=arguments.get("limit", 10),
            )
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "results": [
                                    {
                                        "file_path": r.file_path,
                                        "content": r.content,
                                        "score": r.score,
                                        "language": r.language,
                                        "start_line": r.start_line,
                                        "end_line": r.end_line,
                                    }
                                    for r in results
                                ],
                            },
                            indent=2,
                        ),
                    )
                ],
            )

        if name == "list_indexes":
            indexes = await indexer.list_indexes()
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "indexes": [
                                    {
                                        "name": idx.name,
                                        "repository_path": idx.repository_path,
                                        "file_count": idx.file_count,
                                        "chunk_count": idx.chunk_count,
                                    }
                                    for idx in indexes
                                ],
                            },
                            indent=2,
                        ),
                    )
                ],
            )

        if name == "get_index":
            idx = await indexer.get_index(arguments["name"])
            if idx is None:
                return CallToolResult(
                    content=[
                        TextContent(
                            type="text",
                            text=json.dumps({"error": f"Index '{arguments['name']}' not found"}),
                        )
                    ],
                    isError=True,
                )
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "name": idx.name,
                                "repository_path": idx.repository_path,
                                "file_count": idx.file_count,
                                "chunk_count": idx.chunk_count,
                            },
                            indent=2,
                        ),
                    )
                ],
            )

        if name == "delete_index":
            success = await indexer.delete_index(arguments["name"])
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "success": success,
                                "message": f"Index '{arguments['name']}' deleted"
                                if success
                                else "Failed to delete index",
                            }
                        ),
                    )
                ],
            )

        # Symbol indexing tools
        if name == "index_symbols":
            result = await indexer.index_symbols(
                repo_path=arguments["path"],
                include_patterns=arguments.get("include_patterns"),
                exclude_patterns=arguments.get("exclude_patterns"),
            )
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "success": True,
                                "index_name": result.name,
                                "repository_path": result.repository_path,
                                "file_count": result.file_count,
                                "symbol_count": result.symbol_count,
                            },
                            indent=2,
                        ),
                    )
                ],
            )

        if name == "find_usages":
            # exact_match=True means pattern=False (exact), exact_match=False means pattern=True
            pattern = not arguments.get("exact_match", True)
            results = await indexer.find_usages(
                symbol_name=arguments["symbol_name"],
                index_name=arguments.get("index_name"),
                pattern=pattern,
                limit=arguments.get("limit", 50),
            )
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "symbol": arguments["symbol_name"],
                                "usage_count": len(results),
                                "usages": [
                                    {
                                        "file_path": r.file_path,
                                        "line": r.line,
                                        "column": r.column,
                                        "symbol_name": r.symbol_name,
                                        "symbol_kind": r.symbol_kind,
                                        "category": r.category,
                                        "context": r.context,
                                    }
                                    for r in results
                                ],
                            },
                            indent=2,
                        ),
                    )
                ],
            )

        if name == "find_definitions":
            # exact_match=True means pattern=False (exact), exact_match=False means pattern=True
            pattern = not arguments.get("exact_match", True)
            # find_definitions doesn't have pattern param, uses find_usages internally
            results = await indexer.find_usages(
                symbol_name=arguments["symbol_name"],
                index_name=arguments.get("index_name"),
                category="definition",
                pattern=pattern,
            )
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "symbol": arguments["symbol_name"],
                                "definition_count": len(results),
                                "definitions": [
                                    {
                                        "file_path": r.file_path,
                                        "line": r.line,
                                        "column": r.column,
                                        "symbol_name": r.symbol_name,
                                        "symbol_kind": r.symbol_kind,
                                        "context": r.context,
                                    }
                                    for r in results
                                ],
                            },
                            indent=2,
                        ),
                    )
                ],
            )

        if name == "list_symbols":
            results = await indexer.list_symbols(
                index_name=arguments.get("index_name"),
                kind=arguments.get("symbol_kind"),
                limit=arguments.get("limit", 100),
            )
            return CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps(
                            {
                                "symbol_count": len(results),
                                "symbols": [
                                    {
                                        "symbol_name": r["symbol_name"],
                                        "symbol_kind": r["symbol_kind"],
                                        "total_count": r["total_count"],
                                        "definition_count": r["definition_count"],
                                        "reference_count": r["reference_count"],
                                    }
                                    for r in results
                                ],
                            },
                            indent=2,
                        ),
                    )
                ],
            )

        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps({"error": f"Unknown tool: {name}"}),
                )
            ],
            isError=True,
        )

    except Exception as e:  # pylint: disable=broad-exception-caught
        logger.exception("Error calling tool %s", name)
        return CallToolResult(
            content=[
                TextContent(
                    type="text",
                    text=json.dumps({"error": str(e)}),
                )
            ],
            isError=True,
        )


async def run_server() -> None:
    """Run the MCP server."""
    indexer = get_indexer()

    try:
        await indexer.initialize()
        logger.info("CocoIndex MCP server starting...")

        async with stdio_server() as (read_stream, write_stream):
            await server.run(
                read_stream,
                write_stream,
                server.create_initialization_options(),
            )
    finally:
        await indexer.close()


def main() -> None:
    """Entry point for the MCP server."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stderr,
    )

    try:
        asyncio.run(run_server())
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception:  # pylint: disable=broad-exception-caught
        logger.exception("Server error")
        sys.exit(1)


if __name__ == "__main__":
    main()
