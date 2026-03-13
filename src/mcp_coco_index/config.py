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
class CocoIndexConfig:
    """Main configuration for CocoIndex MCP server."""

    postgres: PostgresConfig = field(default_factory=PostgresConfig)
    embedding: EmbeddingConfig = field(default_factory=EmbeddingConfig)
    data_dir: str = field(default_factory=lambda: os.getenv("COCOINDEX_DATA_DIR", "/tmp/cocoindex"))
    log_level: str = field(default_factory=lambda: os.getenv("LOG_LEVEL", "INFO"))

    @classmethod
    def from_env(cls) -> "CocoIndexConfig":
        """Create configuration from environment variables."""
        return cls()
