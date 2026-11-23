"""Application settings, loaded from environment variables / a .env file.

Everything configurable lives here so that the rest of the code never reads
``os.environ`` directly. That keeps modules testable: tests build a
``Settings`` object by hand instead of patching the environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy.engine import URL, make_url

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SUPPORTED_LLM_PROVIDERS = ("openai", "anthropic", "openai_compatible", "ollama")
SUPPORTED_EMBEDDINGS = ("default", "openai", "hashing")


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _get_int(name: str, default: int, minimum: int | None = None) -> int:
    raw = os.getenv(name)
    value = default if raw is None or raw.strip() == "" else int(raw)
    if minimum is not None and value < minimum:
        raise ValueError(f"{name} must be >= {minimum}, got {value}")
    return value


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    return default if raw is None or raw.strip() == "" else float(raw)


def _get_list(name: str) -> list[str]:
    raw = os.getenv(name, "")
    return [item.strip() for item in raw.split(",") if item.strip()]


@dataclass
class Settings:
    # --- database (read-only credentials!) ---------------------------------
    database_url: str = ""
    db_schema: str = "public"
    allowed_tables: list[str] = field(default_factory=list)  # empty => every readable table in the schema
    require_read_only_role: bool = True

    # --- query safety -------------------------------------------------------
    query_timeout_seconds: int = 15
    max_rows: int = 1000
    max_retries: int = 2

    # --- LLM ----------------------------------------------------------------
    llm_provider: str = "openai"
    llm_model: str = ""
    llm_api_key: str = ""
    llm_base_url: str = ""
    llm_temperature: float = 0.0
    llm_max_output_tokens: int = 2048
    llm_timeout_seconds: int = 60
    context_token_budget: int = 12000  # how much retrieved schema/docs may go in the prompt

    # --- RAG ----------------------------------------------------------------
    embedding_provider: str = "default"
    embedding_model: str = "text-embedding-3-small"
    vector_db_path: str = str(PROJECT_ROOT / "data" / "chroma")
    knowledge_dir: str = str(PROJECT_ROOT / "knowledge")
    n_results: int = 8
    sample_column_values: bool = True
    auto_train: bool = True

    # --- results ------------------------------------------------------------
    summary_enabled: bool = True
    summary_max_rows: int = 50
    history_turns: int = 5

    @classmethod
    def from_env(cls, env_file: str | os.PathLike | None = None) -> Settings:
        load_dotenv(env_file or PROJECT_ROOT / ".env", override=False)

        provider = os.getenv("LLM_PROVIDER", "openai").strip().lower()
        settings = cls(
            database_url=os.getenv("DATABASE_URL", "").strip() or _url_from_parts(),
            db_schema=os.getenv("DB_SCHEMA", "public").strip() or "public",
            allowed_tables=_get_list("ALLOWED_TABLES"),
            require_read_only_role=_get_bool("REQUIRE_READ_ONLY_ROLE", True),
            query_timeout_seconds=_get_int("QUERY_TIMEOUT_SECONDS", 15, minimum=1),
            max_rows=_get_int("MAX_ROWS", 1000, minimum=1),
            max_retries=_get_int("MAX_RETRIES", 2, minimum=0),
            llm_provider=provider,
            llm_model=os.getenv("LLM_MODEL", "").strip(),
            llm_api_key=os.getenv("LLM_API_KEY", "").strip() or os.getenv(f"{provider.upper()}_API_KEY", "").strip(),
            llm_base_url=os.getenv("LLM_BASE_URL", "").strip(),
            llm_temperature=_get_float("LLM_TEMPERATURE", 0.0),
            llm_max_output_tokens=_get_int("LLM_MAX_OUTPUT_TOKENS", 2048, minimum=64),
            llm_timeout_seconds=_get_int("LLM_TIMEOUT_SECONDS", 60, minimum=1),
            context_token_budget=_get_int("CONTEXT_TOKEN_BUDGET", 12000, minimum=1000),
            embedding_provider=os.getenv("EMBEDDING_PROVIDER", "default").strip().lower(),
            embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small").strip(),
            vector_db_path=os.getenv("VECTOR_DB_PATH", str(PROJECT_ROOT / "data" / "chroma")),
            knowledge_dir=os.getenv("KNOWLEDGE_DIR", str(PROJECT_ROOT / "knowledge")),
            n_results=_get_int("RAG_N_RESULTS", 8, minimum=1),
            sample_column_values=_get_bool("SAMPLE_COLUMN_VALUES", True),
            auto_train=_get_bool("AUTO_TRAIN", True),
            summary_enabled=_get_bool("SUMMARY_ENABLED", True),
            summary_max_rows=_get_int("SUMMARY_MAX_ROWS", 50, minimum=1),
            history_turns=_get_int("HISTORY_TURNS", 5, minimum=0),
        )
        settings.validate()
        return settings

    def validate(self) -> None:
        if not self.database_url:
            raise ValueError("Set DATABASE_URL or DB_HOST/DB_NAME/DB_READONLY_USER/DB_READONLY_PASSWORD.")
        if self.llm_provider not in SUPPORTED_LLM_PROVIDERS:
            raise ValueError(f"LLM_PROVIDER must be one of {SUPPORTED_LLM_PROVIDERS}, got {self.llm_provider!r}")
        if self.embedding_provider not in SUPPORTED_EMBEDDINGS:
            raise ValueError(f"EMBEDDING_PROVIDER must be one of {SUPPORTED_EMBEDDINGS}")

    @property
    def safe_database_url(self) -> str:
        """Database URL with the password masked, for display and logs."""
        try:
            return make_url(self.database_url).render_as_string(hide_password=True)
        except Exception:  # pragma: no cover - malformed URL
            return "<invalid database url>"

    @property
    def llm_label(self) -> str:
        return f"{self.llm_provider}:{self.llm_model or '(model not set)'}"


def _url_from_parts() -> str:
    host = os.getenv("DB_HOST", "").strip()
    name = os.getenv("DB_NAME", "").strip()
    if not host or not name:
        return ""
    return URL.create(
        "postgresql+psycopg",
        username=os.getenv("DB_READONLY_USER", "text2sql_reader"),
        password=os.getenv("DB_READONLY_PASSWORD", ""),
        host=host,
        port=int(os.getenv("DB_PORT", "5432")),
        database=name,
    ).render_as_string(hide_password=False)
