"""Wires all components together from ``Settings``. Used by the UI and the CLI."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy.engine import Engine

from .agent import TextToSQLAgent
from .config import Settings
from .db import ConnectionStatus, check_connection, create_db_engine
from .executor import QueryExecutor
from .knowledge_base import build_training_plan, is_trained, train
from .llm import create_vanna
from .schema_inspector import SchemaInfo, inspect_schema
from .sql_generator import SQLGenerator
from .sql_validator import SQLValidator

logger = logging.getLogger(__name__)


class DatabaseUnavailableError(RuntimeError):
    pass


class UnsafeDatabaseRoleError(RuntimeError):
    pass


@dataclass
class AppServices:
    settings: Settings
    engine: Engine
    status: ConnectionStatus
    schema: str
    schema_info: SchemaInfo  # tables approved for this session
    full_schema_info: SchemaInfo  # every readable, configured table (used for training)
    validator: SQLValidator
    executor: QueryExecutor
    vn: object
    generator: SQLGenerator
    agent: TextToSQLAgent

    @property
    def approved_tables(self) -> list[str]:
        return self.validator.allowed_table_list

    def train(self, reset: bool = False) -> dict[str, int]:
        full_validator = SQLValidator(
            self.full_schema_info.qualified_table_names, default_schema=self.schema, max_rows=self.settings.max_rows
        )
        plan = build_training_plan(self.full_schema_info, self.settings.knowledge_dir, full_validator)
        counts = train(self.vn, plan, reset=reset)
        if plan.skipped_examples:
            counts["skipped_examples"] = len(plan.skipped_examples)
        return counts


def connect(settings: Settings, engine: Engine | None = None) -> tuple[Engine, ConnectionStatus]:
    engine = engine or create_db_engine(settings.database_url, settings.query_timeout_seconds)
    status = check_connection(engine)
    if not status.ok:
        raise DatabaseUnavailableError(status.error)
    if settings.require_read_only_role and status.warnings:
        raise UnsafeDatabaseRoleError(
            "The database role is not read-only: "
            + " ".join(status.warnings)
            + " Use read-only credentials, or set REQUIRE_READ_ONLY_ROLE=false (not recommended)."
        )
    return engine, status


def build_services(
    settings: Settings,
    schema: str | None = None,
    approved_tables: list[str] | None = None,
    engine: Engine | None = None,
    vanna_instance=None,
) -> AppServices:
    """Create every component for ``schema``.

    ``approved_tables`` (from the UI) can only narrow the configured allow-list,
    never widen it.
    """
    schema = schema or settings.db_schema
    engine, status = connect(settings, engine)

    full_schema_info = inspect_schema(
        engine,
        schema,
        include_tables=settings.allowed_tables or None,
        sample_values=settings.sample_column_values,
    )
    schema_info = full_schema_info
    if approved_tables is not None:
        keep = {t.split(".")[-1] for t in approved_tables}
        schema_info = SchemaInfo(schema=schema, tables={k: v for k, v in full_schema_info.tables.items() if k in keep})

    validator = SQLValidator(schema_info.qualified_table_names, default_schema=schema, max_rows=settings.max_rows)
    executor = QueryExecutor(engine, settings.query_timeout_seconds, settings.max_rows)
    vn = vanna_instance or create_vanna(settings, status.database, schema)
    generator = SQLGenerator(
        vn,
        allowed_tables=validator.allowed_table_list,
        max_rows=settings.max_rows,
        history_turns=settings.history_turns,
        context_token_budget=settings.context_token_budget,
    )
    agent = TextToSQLAgent(
        vn,
        generator,
        validator,
        executor,
        schema_info=schema_info,
        max_retries=settings.max_retries,
        summary_enabled=settings.summary_enabled,
        summary_max_rows=settings.summary_max_rows,
    )
    services = AppServices(
        settings=settings,
        engine=engine,
        status=status,
        schema=schema,
        schema_info=schema_info,
        full_schema_info=full_schema_info,
        validator=validator,
        executor=executor,
        vn=vn,
        generator=generator,
        agent=agent,
    )
    if settings.auto_train and not is_trained(vn):
        logger.info("Knowledge base for %s.%s is empty: training…", status.database, schema)
        services.train()
    return services
