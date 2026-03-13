# mcp-cocoindex

An [MCP (Model Context Protocol)](https://modelcontextprotocol.io) server for semantic code indexing and search, powered by [CocoIndex](https://cocoindex.io). It lets AI coding assistants (OpenCode, Claude Code, and any MCP-compatible client) index your repositories and search them with natural language queries — all backed by a local PostgreSQL + pgvector store.

## Features

- **Semantic code search** — natural language queries over your codebase using sentence embeddings
- **Multi-language support** — indexes `.py`, `.js`, `.ts`, `.tsx`, `.jsx`, `.go`, `.rs`, `.java`, `.c`, `.cpp`, `.h`, `.hpp`, `.cs`, `.rb`, `.php`, `.swift`, `.kt`, `.scala`, `.md` out of the box
- **Multiple indexes** — maintain separate indexes for different repositories, search across all at once
- **MCP-native** — exposes five tools over stdio transport, compatible with any MCP client
- **Local-first** — embeddings are generated locally with `sentence-transformers/all-MiniLM-L6-v2`, no external API calls required
- **Easy setup** — one `docker compose up` for the database, one command to start the server

## Requirements

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) (package manager)
- [mise](https://mise.jdx.dev/) (optional, for tool version management)
- Docker (for the PostgreSQL + pgvector database)

## Quick Start

### 1. Start the database

```bash
docker compose up -d
```

This starts a `pgvector/pgvector:pg17` container with the default credentials. The server will wait for it to be healthy automatically.

### 2. Configure environment (optional)

Copy `.env.example` and adjust if needed:

```bash
cp .env.example .env
```

All settings have sensible defaults that match the Docker Compose configuration.

### 3. Install dependencies

```bash
uv sync
```

### 4. Run the MCP server

```bash
uv run mcp-coco-index
```

The server runs over **stdio** and is ready to accept MCP client connections.

## Configuration

All configuration is via environment variables:

### PostgreSQL

| Variable | Default | Description |
|---|---|---|
| `COCOINDEX_DB_HOST` | `localhost` | Database host |
| `COCOINDEX_DB_PORT` | `5432` | Database port |
| `COCOINDEX_DB_NAME` | `cocoindex` | Database name |
| `COCOINDEX_DB_USER` | `cocoindex` | Database user |
| `COCOINDEX_DB_PASSWORD` | `cocoindex` | Database password |

### Embedding Model

| Variable | Default | Description |
|---|---|---|
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Sentence-transformers model name |
| `EMBEDDING_DIMENSIONS` | `384` | Embedding vector dimensions |

### General

| Variable | Default | Description |
|---|---|---|
| `COCOINDEX_DATA_DIR` | `/tmp/cocoindex` | Local data directory |
| `LOG_LEVEL` | `INFO` | Logging verbosity |

## MCP Tools

The server exposes five tools to MCP clients:

### `index_repository`

Index a code repository for semantic search. Creates embeddings for code chunks and stores them in the vector database.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `path` | string | yes | Absolute path to the repository |
| `include_patterns` | string[] | no | Glob patterns for files to include (e.g. `**/*.py`) |
| `exclude_patterns` | string[] | no | Glob patterns for files to exclude |

**Returns:** `{ success, index_name, repository_path, file_count, chunk_count }`

---

### `search_code`

Search for code semantically using a natural language query. Returns the most relevant code snippets ranked by similarity score.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `query` | string | yes | Natural language search query |
| `index_name` | string | no | Index to search (omit to search all indexes) |
| `limit` | integer | no | Maximum results to return (default: 10) |

**Returns:** `{ results: [{ file_path, content, score, language, start_line, end_line }] }`

---

### `list_indexes`

List all available code indexes with their statistics (file count, chunk count, repository path).

**Returns:** `{ indexes: [{ name, repository_path, file_count, chunk_count }] }`

---

### `get_index`

Get detailed information about a specific index by name.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `name` | string | yes | Index name |

**Returns:** `{ name, repository_path, file_count, chunk_count }` or an error response.

---

### `delete_index`

Delete a code index and all its stored data.

| Parameter | Type | Required | Description |
|---|---|---|---|
| `name` | string | yes | Index name to delete |

**Returns:** `{ success, message }`

## Client Integration

### OpenCode

An `opencode.json` is included in the repository root. It registers the MCP server automatically when you open the project in OpenCode.

To use it from a different project, add the following to your project's `opencode.json`:

```json
{
  "$schema": "https://opencode.ai/config.json",
  "mcp": {
    "mcp-coco-index": {
      "type": "local",
      "command": ["uv", "run", "--project", "/path/to/mcp-cocoindex", "mcp-coco-index"],
      "enabled": true,
      "environment": {
        "COCOINDEX_DB_HOST": "localhost",
        "COCOINDEX_DB_PORT": "5432",
        "COCOINDEX_DB_NAME": "cocoindex",
        "COCOINDEX_DB_USER": "cocoindex",
        "COCOINDEX_DB_PASSWORD": "cocoindex"
      }
    }
  }
}
```

### Claude Code

A `.mcp.json` is included in the repository root. To use it from another project, add the following to that project's `.mcp.json`:

```json
{
  "mcpServers": {
    "mcp-coco-index": {
      "command": "uv",
      "args": ["run", "--project", "/path/to/mcp-cocoindex", "mcp-coco-index"],
      "env": {
        "COCOINDEX_DB_HOST": "localhost",
        "COCOINDEX_DB_PORT": "5432",
        "COCOINDEX_DB_NAME": "cocoindex",
        "COCOINDEX_DB_USER": "cocoindex",
        "COCOINDEX_DB_PASSWORD": "cocoindex"
      }
    }
  }
}
```

### Other MCP Clients

The server communicates over **stdio** with no special transport requirements. Point any MCP-compatible client at:

```
uv run --project /path/to/mcp-cocoindex mcp-coco-index
```

## Development

All tasks use [Task](https://taskfile.dev) via the included `Taskfile.yml`.

### Available Tasks

```bash
task start           # Start the PostgreSQL container
task stop            # Stop the PostgreSQL container
task logs            # Follow container logs

task test            # Run all tests
task test:unit       # Config tests only (no database required)
task test:integration  # Integration tests (requires database)
task test:opencode   # OpenCode-specific tests
task test:claudecode # Claude Code-specific tests
task test:coverage   # Run tests with HTML coverage report

task format          # Auto-format with ruff
task lint            # Lint and auto-fix with ruff
task build           # Build the package

task dev             # Run the server in development mode
task index           # Index this repository
task search QUERY="your query"  # Search indexed code
task list:indexes    # List all indexes

task validate        # Full pipeline: format → lint → build → test → docs
task clean           # Remove build artifacts and caches
task clean:infra     # Stop containers and remove all volumes (destructive)
```

### Running the Test Suite

```bash
task start   # ensure database is running
task test    # runs all 37 tests
```

### Project Structure

```
mcp-cocoindex/
├── src/mcp_coco_index/
│   ├── config.py       # Configuration dataclasses (env-driven)
│   ├── indexer.py      # Core indexing, search, and CocoIndex flow logic
│   └── server.py       # MCP server and tool definitions
├── tests/
│   ├── test_config.py        # Unit tests (no database)
│   ├── test_integration.py   # Full integration tests
│   ├── test_opencode.py      # OpenCode integration tests
│   └── test_claudecode.py    # Claude Code integration tests
├── docker-compose.yml  # PostgreSQL + pgvector service
├── Taskfile.yml        # All development tasks
├── opencode.json       # OpenCode MCP registration
├── .mcp.json           # Claude Code MCP registration
└── .env.example        # Environment variable template
```

## How It Works

1. **Indexing** — when `index_repository` is called, the server uses CocoIndex to build a processing flow that reads source files, splits them into overlapping chunks (1000 tokens, 200-token overlap), generates a 384-dimensional embedding for each chunk via `sentence-transformers/all-MiniLM-L6-v2`, and stores everything in a PostgreSQL table with a pgvector COSINE_SIMILARITY index.

2. **Searching** — when `search_code` is called, the query is embedded with the same model, then a cosine similarity search is run against all relevant `code_index_*` tables. Results are ranked by similarity score and the top `limit` chunks are returned.

3. **Storage** — each repository gets its own `code_index_{name}` table in PostgreSQL. The table stores `file_path`, `location`, `content`, and the `embedding` vector. CocoIndex also maintains a tracking table for incremental updates.

## License

MIT
