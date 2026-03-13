"""Integration tests for CocoIndex MCP server.

These tests require:
1. PostgreSQL with pgvector running (task start)
2. Environment variables set (or defaults used)

Run with: uv run pytest tests/test_integration.py -v
"""
# pylint: disable=redefined-outer-name,protected-access,import-outside-toplevel
# pylint: disable=broad-exception-caught,unused-argument

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Generator

import psycopg
import pytest

from mcp_coco_index.config import CocoIndexConfig
from mcp_coco_index.indexer import CocoIndexer, IndexInfo, SearchResult


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


@pytest.fixture
def config() -> CocoIndexConfig:
    """Create test configuration."""
    return CocoIndexConfig.from_env()


@pytest.fixture
def indexer(config: CocoIndexConfig) -> Generator[CocoIndexer, None, None]:
    """Create and initialize indexer."""
    idx = CocoIndexer(config)
    yield idx
    # Cleanup is handled in tests that create indexes


@pytest.fixture
def sample_repo() -> Generator[Path, None, None]:
    """Create a temporary repository with sample code files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        repo_path = Path(tmpdir)

        # Create sample Python file
        py_file = repo_path / "main.py"
        py_file.write_text('''"""Main application module."""

def authenticate_user(username: str, password: str) -> bool:
    """Authenticate a user with username and password.

    Args:
        username: The user\'s username
        password: The user\'s password

    Returns:
        True if authentication succeeds, False otherwise
    """
    # In a real app, this would check against a database
    if not username or not password:
        return False
    return username == "admin" and password == "secret"


def get_user_profile(user_id: int) -> dict:
    """Get user profile information.

    Args:
        user_id: The user\'s ID

    Returns:
        Dictionary containing user profile data
    """
    return {
        "id": user_id,
        "name": "Test User",
        "email": "test@example.com",
    }


class UserService:
    """Service for managing users."""

    def __init__(self, database_url: str):
        """Initialize the user service.

        Args:
            database_url: URL for the database connection
        """
        self.database_url = database_url

    def create_user(self, username: str, email: str) -> int:
        """Create a new user.

        Args:
            username: The user\'s username
            email: The user\'s email

        Returns:
            The new user\'s ID
        """
        # Would insert into database
        return 1

    def delete_user(self, user_id: int) -> bool:
        """Delete a user by ID.

        Args:
            user_id: The user\'s ID

        Returns:
            True if deleted, False otherwise
        """
        return True
''')

        # Create sample TypeScript file
        ts_dir = repo_path / "src"
        ts_dir.mkdir()
        ts_file = ts_dir / "api.ts"
        ts_file.write_text("""/**
 * API client for the backend service.
 */

export interface User {
  id: number;
  username: string;
  email: string;
}

export interface AuthResponse {
  token: string;
  expiresAt: Date;
}

/**
 * Authenticate user and get access token.
 * @param username - The user's username
 * @param password - The user's password
 * @returns Authentication response with token
 */
export async function login(username: string, password: string): Promise<AuthResponse> {
  const response = await fetch('/api/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  });

  if (!response.ok) {
    throw new Error('Authentication failed');
  }

  return response.json();
}

/**
 * Fetch current user's profile.
 * @param token - The authentication token
 * @returns User profile data
 */
export async function getProfile(token: string): Promise<User> {
  const response = await fetch('/api/user/profile', {
    headers: { Authorization: `Bearer ${token}` },
  });

  if (!response.ok) {
    throw new Error('Failed to fetch profile');
  }

  return response.json();
}

/**
 * Handle API errors gracefully.
 * @param error - The error to handle
 */
export function handleApiError(error: Error): void {
  console.error('API Error:', error.message);
  // Could send to error tracking service
}
""")

        # Create a README
        readme = repo_path / "README.md"
        readme.write_text("""# Sample Project

This is a sample project for testing CocoIndex.

## Features

- User authentication with JWT tokens
- User profile management
- API error handling

## Installation

```bash
npm install
```

## Usage

```typescript
import { login, getProfile } from './src/api';

const auth = await login('admin', 'password');
const profile = await getProfile(auth.token);
```
""")

        yield repo_path


class TestCocoIndexer:
    """Integration tests for CocoIndexer."""

    @pytest.mark.asyncio
    async def test_initialize(self, indexer: CocoIndexer) -> None:
        """Test indexer initialization."""
        await indexer.initialize()
        assert indexer._initialized is True

    @pytest.mark.asyncio
    async def test_index_repository(self, indexer: CocoIndexer, sample_repo: Path) -> None:
        """Test indexing a repository."""
        try:
            result = await indexer.index_repository(str(sample_repo))

            assert isinstance(result, IndexInfo)
            assert result.name is not None
            assert result.repository_path == str(sample_repo)
            assert result.file_count >= 1  # At least the Python file
            assert result.chunk_count >= 1

        finally:
            # Cleanup
            index_name = indexer._get_index_name(str(sample_repo))
            await indexer.delete_index(index_name)

    @pytest.mark.asyncio
    async def test_search_code(self, indexer: CocoIndexer, sample_repo: Path) -> None:
        """Test semantic code search."""
        try:
            # First index the repo
            await indexer.index_repository(str(sample_repo))
            index_name = indexer._get_index_name(str(sample_repo))

            # Search for authentication-related code
            results = await indexer.search(
                query="user authentication with password",
                index_name=index_name,
                limit=5,
            )

            assert isinstance(results, list)
            # Should find results related to authentication
            if len(results) > 0:
                assert isinstance(results[0], SearchResult)
                assert results[0].file_path is not None
                assert results[0].content is not None
                assert results[0].score > 0

        finally:
            # Cleanup
            index_name = indexer._get_index_name(str(sample_repo))
            await indexer.delete_index(index_name)

    @pytest.mark.asyncio
    async def test_search_finds_relevant_results(
        self, indexer: CocoIndexer, sample_repo: Path
    ) -> None:
        """Test that search returns semantically relevant results."""
        try:
            await indexer.index_repository(str(sample_repo))
            index_name = indexer._get_index_name(str(sample_repo))

            # Search for error handling
            results = await indexer.search(
                query="how to handle API errors",
                index_name=index_name,
                limit=3,
            )

            # Should find the handleApiError function or related code
            assert len(results) >= 1
            # Check that at least one result contains error-related content
            found_error_handling = any(
                "error" in r.content.lower() or "Error" in r.content for r in results
            )
            assert found_error_handling, "Should find error handling code"

        finally:
            index_name = indexer._get_index_name(str(sample_repo))
            await indexer.delete_index(index_name)

    @pytest.mark.asyncio
    async def test_list_indexes(self, indexer: CocoIndexer, sample_repo: Path) -> None:
        """Test listing indexes."""
        try:
            # Index the repo first
            await indexer.index_repository(str(sample_repo))
            index_name = indexer._get_index_name(str(sample_repo))

            # List indexes
            indexes = await indexer.list_indexes()

            assert isinstance(indexes, list)
            # Find our index
            our_index = next((idx for idx in indexes if idx.name == index_name), None)
            assert our_index is not None
            assert our_index.file_count >= 1
            assert our_index.chunk_count >= 1

        finally:
            index_name = indexer._get_index_name(str(sample_repo))
            await indexer.delete_index(index_name)

    @pytest.mark.asyncio
    async def test_get_index(self, indexer: CocoIndexer, sample_repo: Path) -> None:
        """Test getting a specific index."""
        try:
            await indexer.index_repository(str(sample_repo))
            index_name = indexer._get_index_name(str(sample_repo))

            # Get the index
            idx = await indexer.get_index(index_name)

            assert idx is not None
            assert idx.name == index_name
            assert idx.file_count >= 1

        finally:
            index_name = indexer._get_index_name(str(sample_repo))
            await indexer.delete_index(index_name)

    @pytest.mark.asyncio
    async def test_get_nonexistent_index(self, indexer: CocoIndexer) -> None:
        """Test getting a non-existent index returns None."""
        await indexer.initialize()
        idx = await indexer.get_index("nonexistent_index_12345")
        assert idx is None

    @pytest.mark.asyncio
    async def test_delete_index(self, indexer: CocoIndexer, sample_repo: Path) -> None:
        """Test deleting an index."""
        # Index first
        await indexer.index_repository(str(sample_repo))
        index_name = indexer._get_index_name(str(sample_repo))

        # Verify it exists
        idx = await indexer.get_index(index_name)
        assert idx is not None

        # Delete
        success = await indexer.delete_index(index_name)
        assert success is True

        # Verify it's gone
        idx = await indexer.get_index(index_name)
        assert idx is None

    @pytest.mark.asyncio
    async def test_delete_nonexistent_index(self, indexer: CocoIndexer) -> None:
        """Test deleting a non-existent index."""
        await indexer.initialize()
        # Should return True (DROP TABLE IF EXISTS succeeds)
        success = await indexer.delete_index("nonexistent_index_12345")
        assert success is True

    @pytest.mark.asyncio
    async def test_index_with_custom_patterns(
        self, indexer: CocoIndexer, sample_repo: Path
    ) -> None:
        """Test indexing with custom include/exclude patterns."""
        try:
            # Only include Python files
            result = await indexer.index_repository(
                str(sample_repo),
                include_patterns=["**/*.py"],
                exclude_patterns=["**/__pycache__/**"],
            )

            assert result.file_count >= 1
            # Should only have Python file chunks

        finally:
            index_name = indexer._get_index_name(str(sample_repo))
            await indexer.delete_index(index_name)

    @pytest.mark.asyncio
    async def test_search_all_indexes(self, indexer: CocoIndexer, sample_repo: Path) -> None:
        """Test searching across all indexes."""
        try:
            await indexer.index_repository(str(sample_repo))

            # Search without specifying index
            results = await indexer.search(
                query="user profile",
                limit=5,
            )

            assert isinstance(results, list)
            # Should find profile-related code
            if len(results) > 0:
                found_profile = any(
                    "profile" in r.content.lower() or "Profile" in r.content for r in results
                )
                assert found_profile, "Should find profile-related code"

        finally:
            index_name = indexer._get_index_name(str(sample_repo))
            await indexer.delete_index(index_name)


class TestMCPServer:
    """Integration tests for MCP server tools."""

    @pytest.mark.asyncio
    async def test_server_tool_index_repository(
        self, sample_repo: Path, config: CocoIndexConfig
    ) -> None:
        """Test the index_repository tool through the server interface."""
        from mcp_coco_index.server import call_tool, get_indexer

        indexer = get_indexer()
        try:
            result = await call_tool(
                "index_repository",
                {"path": str(sample_repo)},
            )

            assert result.isError is not True
            assert len(result.content) > 0
            # Parse the JSON response
            data = json.loads(result.content[0].text)
            assert data["success"] is True
            assert "index_name" in data
            assert data["file_count"] >= 1

        finally:
            # Cleanup
            index_name = indexer._get_index_name(str(sample_repo))
            await indexer.delete_index(index_name)

    @pytest.mark.asyncio
    async def test_server_tool_search_code(
        self, sample_repo: Path, config: CocoIndexConfig
    ) -> None:
        """Test the search_code tool through the server interface."""
        from mcp_coco_index.server import call_tool, get_indexer

        indexer = get_indexer()
        try:
            # Index first
            await call_tool("index_repository", {"path": str(sample_repo)})

            # Search
            result = await call_tool(
                "search_code",
                {"query": "authenticate user", "limit": 3},
            )

            assert result.isError is not True
            data = json.loads(result.content[0].text)
            assert "results" in data
            assert isinstance(data["results"], list)

        finally:
            index_name = indexer._get_index_name(str(sample_repo))
            await indexer.delete_index(index_name)

    @pytest.mark.asyncio
    async def test_server_tool_list_indexes(self, config: CocoIndexConfig) -> None:
        """Test the list_indexes tool through the server interface."""
        from mcp_coco_index.server import call_tool, get_indexer

        # Just ensure indexer is initialized
        indexer = get_indexer()
        await indexer.initialize()

        result = await call_tool("list_indexes", {})

        assert result.isError is not True
        data = json.loads(result.content[0].text)
        assert "indexes" in data
        assert isinstance(data["indexes"], list)

    @pytest.mark.asyncio
    async def test_server_tool_unknown(self) -> None:
        """Test calling an unknown tool."""
        from mcp_coco_index.server import call_tool

        result = await call_tool("unknown_tool", {})

        assert result.isError is True
        data = json.loads(result.content[0].text)
        assert "error" in data
