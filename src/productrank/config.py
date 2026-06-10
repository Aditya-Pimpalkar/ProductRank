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

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
