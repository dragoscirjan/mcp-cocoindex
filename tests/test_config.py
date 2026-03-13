"""Tests for configuration."""

from mcp_coco_index.config import CocoIndexConfig, PostgresConfig


def test_default_config():
    """Test default configuration values."""
    config = CocoIndexConfig()

    assert config.postgres.host == "localhost"
    assert config.postgres.port == 5432
    assert config.postgres.database == "cocoindex"
    assert config.postgres.user == "cocoindex"
    assert config.postgres.password == "cocoindex"
    assert config.embedding.model_name == "all-MiniLM-L6-v2"
    assert config.log_level == "INFO"


def test_postgres_connection_string():
    """Test PostgreSQL connection string generation."""
    postgres = PostgresConfig(
        host="db.example.com",
        port=5433,
        database="testdb",
        user="testuser",
        password="testpass",
    )

    expected = "postgresql://testuser:testpass@db.example.com:5433/testdb"
    assert postgres.connection_string == expected


def test_config_from_env(monkeypatch):
    """Test configuration from environment variables."""
    monkeypatch.setenv("COCOINDEX_DB_HOST", "envhost")
    monkeypatch.setenv("COCOINDEX_DB_PORT", "5434")
    monkeypatch.setenv("EMBEDDING_MODEL", "custom-model")

    config = CocoIndexConfig.from_env()

    assert config.postgres.host == "envhost"
    assert config.postgres.port == 5434
    assert config.embedding.model_name == "custom-model"
