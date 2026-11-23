"""PostgreSQL connectivity: engine creation, health checks and role auditing."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from sqlalchemy import create_engine, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)


def create_db_engine(database_url: str, query_timeout_seconds: int = 15) -> Engine:
    """Create a pooled SQLAlchemy engine.

    Every connection is opened with ``default_transaction_read_only=on`` and a
    ``statement_timeout`` as session defaults. These are a second line of defence
    behind the read-only database role and the SQL validator.
    """
    timeout_ms = int(query_timeout_seconds * 1000)
    return create_engine(
        database_url,
        pool_pre_ping=True,
        pool_size=5,
        max_overflow=5,
        pool_recycle=1800,
        connect_args={
            "connect_timeout": 10,
            "application_name": "text2sql-agent",
            "options": f"-c default_transaction_read_only=on -c statement_timeout={timeout_ms}",
        },
    )


@dataclass
class ConnectionStatus:
    ok: bool
    database: str = ""
    user: str = ""
    server_version: str = ""
    read_only: bool = False
    warnings: list[str] = field(default_factory=list)
    error: str = ""


def check_connection(engine: Engine) -> ConnectionStatus:
    """Connect, report server info, and audit whether the role is truly read-only."""
    try:
        with engine.connect() as conn:
            row = conn.execute(
                text(
                    "SELECT current_database(), current_user, "
                    "current_setting('server_version'), "
                    "current_setting('default_transaction_read_only')"
                )
            ).one()
            status = ConnectionStatus(
                ok=True,
                database=row[0],
                user=row[1],
                server_version=row[2],
            )
            status.warnings = audit_role_privileges(conn)
            status.read_only = not status.warnings
            conn.rollback()
            return status
    except Exception as exc:  # noqa: BLE001 - surfaced to the UI as a status message
        logger.warning("Database connection failed: %s", exc)
        return ConnectionStatus(ok=False, error=_short_error(exc))


def audit_role_privileges(conn) -> list[str]:
    """Return a list of problems that make the current role NOT read-only."""
    problems: list[str] = []
    role = conn.execute(
        text("SELECT rolsuper, rolcreatedb, rolcreaterole, rolbypassrls FROM pg_roles WHERE rolname = current_user")
    ).one()
    if role.rolsuper:
        problems.append("Role is a SUPERUSER.")
    if role.rolcreatedb:
        problems.append("Role can CREATE DATABASE.")
    if role.rolcreaterole:
        problems.append("Role can CREATE ROLE.")
    if role.rolbypassrls:
        problems.append("Role bypasses row-level security.")

    writable = (
        conn.execute(
            text(
                """
            SELECT n.nspname || '.' || c.relname AS table_name
            FROM pg_class c
            JOIN pg_namespace n ON n.oid = c.relnamespace
            WHERE c.relkind IN ('r', 'p', 'v', 'm')
              AND n.nspname NOT IN ('pg_catalog', 'information_schema')
              AND n.nspname NOT LIKE 'pg_toast%'
              AND (has_table_privilege(c.oid, 'INSERT')
                   OR has_table_privilege(c.oid, 'UPDATE')
                   OR has_table_privilege(c.oid, 'DELETE')
                   OR has_table_privilege(c.oid, 'TRUNCATE'))
            LIMIT 5
            """
            )
        )
        .scalars()
        .all()
    )
    if writable:
        problems.append("Role has write privileges on: " + ", ".join(writable))

    can_create = (
        conn.execute(
            text(
                "SELECT n.nspname FROM pg_namespace n "
                "WHERE has_schema_privilege(n.oid, 'CREATE') "
                "AND n.nspname NOT LIKE 'pg_%' AND n.nspname <> 'information_schema' LIMIT 5"
            )
        )
        .scalars()
        .all()
    )
    if can_create:
        problems.append("Role can CREATE objects in schema(s): " + ", ".join(can_create))
    return problems


def _short_error(exc: Exception) -> str:
    msg = str(getattr(exc, "orig", None) or exc)
    return msg.strip().splitlines()[0][:300] if msg else exc.__class__.__name__
