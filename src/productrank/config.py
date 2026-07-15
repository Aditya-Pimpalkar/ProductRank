"""Central, env-driven configuration.

One settings object, validated by pydantic-settings, read once at import. Keeping
config in a single place (rather than scattered os.getenv calls) makes every tunable
visible and testable, and keeps the OpenAI key off of any code path that logs.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- OpenAI (embeddings only) ---
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    embedding_model: str = Field(default="text-embedding-3-small", alias="EMBEDDING_MODEL")
    embedding_dim: int = Field(default=1536, alias="EMBEDDING_DIM")

    # --- Postgres ---
    postgres_user: str = Field(default="productrank", alias="POSTGRES_USER")
    postgres_password: str = Field(default="productrank", alias="POSTGRES_PASSWORD")
    postgres_db: str = Field(default="productrank", alias="POSTGRES_DB")
    postgres_host: str = Field(default="localhost", alias="POSTGRES_HOST")
    postgres_port: int = Field(default=5433, alias="POSTGRES_PORT")

    # --- Multi-dataset (Path 2): two databases inside one ParadeDB instance ---
    # Each dataset is a separate database (different dbname, same host/instance), so the
    # tested retrieval SQL runs unchanged against whichever connection it is given, and
    # cross-corpus contamination is impossible by construction. Override per environment
    # (locally, point msmarco at an existing db to reuse its embeddings).
    db_name_msmarco: str = Field(default="productrank_msmarco", alias="DB_NAME_MSMARCO")
    db_name_fiqa: str = Field(default="productrank_fiqa", alias="DB_NAME_FIQA")

    # --- Redis ---
    redis_url: str = Field(default="redis://localhost:6379/0", alias="REDIS_URL")

    # --- Retrieval / rerank ---
    rerank_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2", alias="RERANK_MODEL")
    rrf_k: int = Field(default=60, alias="RRF_K")
    default_top_k: int = Field(default=10, alias="DEFAULT_TOP_K")
    rerank_candidates: int = Field(default=100, alias="RERANK_CANDIDATES")
    ivfflat_lists: int = Field(default=100, alias="IVFFLAT_LISTS")
    ivfflat_probes: int = Field(default=10, alias="IVFFLAT_PROBES")

    # --- App ---
    frontend_origin: str = Field(default="http://localhost:3000", alias="FRONTEND_ORIGIN")
    log_level: str = Field(default="INFO", alias="LOG_LEVEL")

    def _url(self, dbname: str) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{dbname}"
        )

    @property
    def database_url(self) -> str:
        """URL for the default dataset's database (used by Alembic/tests by default)."""
        return self._url(self.dataset_dbnames[DEFAULT_DATASET])

    @property
    def dataset_dbnames(self) -> dict[str, str]:
        return {"msmarco": self.db_name_msmarco, "fiqa": self.db_name_fiqa}

    def database_url_for(self, dataset: str) -> str:
        """Build the connection URL for a dataset. Validates against the allowlist so a
        raw/unknown value can never reach dbname construction (defense in depth — the API
        boundary also rejects unknown datasets via the Pydantic enum)."""
        if dataset not in DATASETS:
            raise ValueError(f"unknown dataset {dataset!r}; allowed: {DATASETS}")
        return self._url(self.dataset_dbnames[dataset])


# --- Dataset allowlist (single source of truth) -------------------------------------
DATASETS: tuple[str, ...] = ("msmarco", "fiqa")
DEFAULT_DATASET = "msmarco"
# Which BEIR qrels split each dataset is evaluated on (drives results-file naming).
DATASET_SPLIT: dict[str, str] = {"msmarco": "dev", "fiqa": "test"}


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
