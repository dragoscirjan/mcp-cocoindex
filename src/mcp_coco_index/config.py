"""Configuration for CocoIndex MCP server."""

import os
from dataclasses import dataclass, field


@dataclass
class PostgresConfig:
    """PostgreSQL connection configuration."""

    host: str = field(default_factory=lambda: os.getenv("COCOINDEX_DB_HOST", "localhost"))
    port: int = field(default_factory=lambda: int(os.getenv("COCOINDEX_DB_PORT", "5432")))
    database: str = field(default_factory=lambda: os.getenv("COCOINDEX_DB_NAME", "cocoindex"))
    user: str = field(default_factory=lambda: os.getenv("COCOINDEX_DB_USER", "cocoindex"))
    password: str = field(default_factory=lambda: os.getenv("COCOINDEX_DB_PASSWORD", "cocoindex"))

    @property
    def connection_string(self) -> str:
        """Get the PostgreSQL connection string."""
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.database}"


@dataclass
class EmbeddingConfig:
    """Embedding model configuration."""

    model_name: str = field(
        default_factory=lambda: os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
    )
    dimensions: int = field(default_factory=lambda: int(os.getenv("EMBEDDING_DIMENSIONS", "384")))


@dataclass
class LlmConfig:
    """LLM configuration for symbol extraction fallback (non-tree-sitter languages).

    Supported api_type values (maps to cocoindex.LlmApiType):
    - "openrouter" - OpenRouter API (requires LLM_API_KEY)
    - "openai" - OpenAI API (requires LLM_API_KEY)
    - "anthropic" - Anthropic API (requires LLM_API_KEY)
    - "ollama" - Local Ollama (no API key needed, uses LLM_ADDRESS)
    - "gemini" - Google Gemini (requires LLM_API_KEY)
    - "azure_openai" - Azure OpenAI (requires LLM_API_KEY)
    - "bedrock" - AWS Bedrock
    - "vertex_ai" - Google Vertex AI
    - "vllm" - vLLM server (uses LLM_ADDRESS)
    - "litellm" - LiteLLM proxy
    """

    api_type: str = field(default_factory=lambda: os.getenv("LLM_API_TYPE", "ollama"))
    model: str = field(default_factory=lambda: os.getenv("LLM_MODEL", "llama3.2"))
    api_key: str | None = field(default_factory=lambda: os.getenv("LLM_API_KEY"))
    address: str | None = field(default_factory=lambda: os.getenv("LLM_ADDRESS"))

    @property
    def is_configured(self) -> bool:
        """Check if LLM is properly configured for use."""
        # Ollama and vLLM don't require API keys
        if self.api_type in ("ollama", "vllm"):
            return True
        # Cloud providers require API keys
        return self.api_key is not None


@dataclass
class CocoIndexConfig:
    """Main configuration for CocoIndex MCP server."""

    postgres: PostgresConfig = field(default_factory=PostgresConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    llm: LlmConfig = field(default_factory=LlmConfig)
    data_dir: str = field(default_factory=lambda: os.getenv("COCOINDEX_DATA_DIR", "/tmp/cocoindex"))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    @classmethod
    def from_env(cls) -> "CocoIndexConfig":
        """Create configuration from environment variables."""
        return cls()
