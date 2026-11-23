"""Database schema inspection.

Reads tables, views, columns, data types, primary keys, foreign keys and
comments through SQLAlchemy's inspector, keeping only objects the connected
(read-only) role can actually SELECT from. The result is turned into DDL and
plain-English descriptions for the RAG knowledge base, and into the approved
table list used by the SQL validator.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from sqlalchemy import inspect, text
from sqlalchemy.engine import Engine

logger = logging.getLogger(__name__)

SYSTEM_SCHEMAS = {"pg_catalog", "information_schema", "pg_toast"}
# Columns whose values we never sample into the knowledge base (likely PII / free text).
PII_NAME_PATTERN = re.compile(
    r"(email|phone|first_?name|last_?name|full_?name|surname|user_?name|address|street|ssn|password|"
    r"token|secret|birth|card|iban|ip_addr|description|comment|note)",
    re.IGNORECASE,
)
TEXT_TYPES = ("TEXT", "VARCHAR", "CHAR", "CHARACTER", "CITEXT", "ENUM")


@dataclass
class ColumnInfo:
    name: str
    data_type: str
    nullable: bool = True
    default: str | None = None
    comment: str | None = None
    is_primary_key: bool = False
    sample_values: list[str] = field(default_factory=list)


@dataclass
class ForeignKeyInfo:
    columns: list[str]
    referred_schema: str
    referred_table: str
    referred_columns: list[str]
    name: str | None = None


@dataclass
class TableInfo:
    schema: str
    name: str
    columns: list[ColumnInfo] = field(default_factory=list)
    primary_key: list[str] = field(default_factory=list)
    foreign_keys: list[ForeignKeyInfo] = field(default_factory=list)
    comment: str | None = None
    kind: str = "table"  # "table" | "view"
    row_estimate: int | None = None

    @property
    def full_name(self) -> str:
        return f"{self.schema}.{self.name}"

    def column(self, name: str) -> ColumnInfo | None:
        return next((c for c in self.columns if c.name == name), None)

    def to_ddl(self) -> str:
        """Render a CREATE TABLE statement with comments, used as RAG context."""
        lines: list[str] = []
        if self.comment:
            lines.append(f"-- {self.comment}")
        keyword = "VIEW" if self.kind == "view" else "TABLE"
        lines.append(f"CREATE {keyword} {self.full_name} (")
        body: list[tuple[str, str | None]] = []  # (definition, trailing comment)
        for col in self.columns:
            definition = f"    {col.name} {col.data_type}" + ("" if col.nullable else " NOT NULL")
            body.append((definition, col.comment))
        if self.primary_key:
            body.append((f"    PRIMARY KEY ({', '.join(self.primary_key)})", None))
        for fk in self.foreign_keys:
            body.append(
                (
                    f"    FOREIGN KEY ({', '.join(fk.columns)}) REFERENCES "
                    f"{fk.referred_schema}.{fk.referred_table} ({', '.join(fk.referred_columns)})",
                    None,
                )
            )
        for i, (definition, comment) in enumerate(body):
            sep = "," if i < len(body) - 1 else ""
            lines.append(f"{definition}{sep}" + (f"  -- {comment}" if comment else ""))
        lines.append(");")
        return "\n".join(lines)

    def to_description(self) -> str:
        """Plain-English description of the table, including relationships and known values."""
        out = [f"Table {self.full_name}" + (f": {self.comment}" if self.comment else ".")]
        if self.primary_key:
            out.append(f"Primary key: {', '.join(self.primary_key)}.")
        for fk in self.foreign_keys:
            out.append(
                f"{self.name}.{'/'.join(fk.columns)} references "
                f"{fk.referred_table}.{'/'.join(fk.referred_columns)} "
                f"(join: {self.name}.{fk.columns[0]} = {fk.referred_table}.{fk.referred_columns[0]})."
            )
        for col in self.columns:
            if col.sample_values:
                values = ", ".join(f"'{v}'" for v in col.sample_values)
                out.append(f"Column {self.name}.{col.name} only contains these values: {values}.")
        return "\n".join(out)


@dataclass
class SchemaInfo:
    schema: str
    tables: dict[str, TableInfo] = field(default_factory=dict)

    @property
    def table_names(self) -> list[str]:
        return sorted(self.tables)

    @property
    def qualified_table_names(self) -> list[str]:
        return sorted(t.full_name for t in self.tables.values())

    def relationships(self) -> list[str]:
        rels = []
        for table in self.tables.values():
            for fk in table.foreign_keys:
                rels.append(
                    f"{table.full_name}({', '.join(fk.columns)}) -> "
                    f"{fk.referred_schema}.{fk.referred_table}({', '.join(fk.referred_columns)})"
                )
        return sorted(rels)

    def relationships_doc(self) -> str:
        rels = self.relationships()
        if not rels:
            return ""
        return "Foreign-key relationships (use these for JOIN conditions):\n" + "\n".join(f"- {r}" for r in rels)


def list_schemas(engine: Engine) -> list[str]:
    """Schemas the current role can use (excluding system schemas)."""
    with engine.connect() as conn:
        rows = (
            conn.execute(
                text(
                    "SELECT nspname FROM pg_namespace "
                    "WHERE has_schema_privilege(oid, 'USAGE') "
                    "AND nspname NOT LIKE 'pg\\_%' AND nspname <> 'information_schema' "
                    "ORDER BY nspname"
                )
            )
            .scalars()
            .all()
        )
    return list(rows)


def inspect_schema(
    engine: Engine,
    schema: str = "public",
    include_tables: list[str] | None = None,
    sample_values: bool = True,
    max_distinct_values: int = 15,
) -> SchemaInfo:
    """Inspect ``schema`` and return metadata for every table the role can SELECT.

    ``include_tables`` optionally restricts the result to an allow-list
    (bare names or ``schema.table``).
    """
    if schema in SYSTEM_SCHEMAS:
        raise ValueError(f"Refusing to inspect system schema {schema!r}")

    wanted = _normalise_allow_list(include_tables, schema)
    inspector = inspect(engine)
    info = SchemaInfo(schema=schema)

    candidates = [(n, "table") for n in inspector.get_table_names(schema=schema)]
    candidates += [(n, "view") for n in inspector.get_view_names(schema=schema)]
    try:
        candidates += [(n, "view") for n in inspector.get_materialized_view_names(schema=schema)]
    except NotImplementedError:  # pragma: no cover
        pass

    with engine.connect() as conn:
        readable = set(
            conn.execute(
                text(
                    "SELECT c.relname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace "
                    "WHERE n.nspname = :schema AND has_table_privilege(c.oid, 'SELECT')"
                ),
                {"schema": schema},
            ).scalars()
        )
        estimates = dict(
            conn.execute(
                text(
                    "SELECT c.relname, c.reltuples::bigint FROM pg_class c "
                    "JOIN pg_namespace n ON n.oid = c.relnamespace WHERE n.nspname = :schema"
                ),
                {"schema": schema},
            ).all()
        )

    for name, kind in sorted(candidates):
        if name not in readable:
            continue
        if wanted is not None and name not in wanted:
            continue
        pk = inspector.get_pk_constraint(name, schema=schema) or {}
        pk_cols = list(pk.get("constrained_columns") or [])
        columns = [
            ColumnInfo(
                name=c["name"],
                data_type=str(c["type"]),
                nullable=bool(c.get("nullable", True)),
                default=str(c["default"]) if c.get("default") is not None else None,
                comment=c.get("comment"),
                is_primary_key=c["name"] in pk_cols,
            )
            for c in inspector.get_columns(name, schema=schema)
        ]
        fks = [
            ForeignKeyInfo(
                columns=list(fk["constrained_columns"]),
                referred_schema=fk.get("referred_schema") or schema,
                referred_table=fk["referred_table"],
                referred_columns=list(fk["referred_columns"]),
                name=fk.get("name"),
            )
            for fk in inspector.get_foreign_keys(name, schema=schema)
        ]
        try:
            comment = (inspector.get_table_comment(name, schema=schema) or {}).get("text")
        except NotImplementedError:  # pragma: no cover
            comment = None
        estimate = estimates.get(name)
        info.tables[name] = TableInfo(
            schema=schema,
            name=name,
            columns=columns,
            primary_key=pk_cols,
            foreign_keys=fks,
            comment=comment,
            kind=kind,
            row_estimate=int(estimate) if estimate is not None and estimate >= 0 else None,
        )

    if sample_values:
        _sample_categorical_values(engine, info, max_distinct_values)
    return info


def _normalise_allow_list(include_tables: list[str] | None, schema: str) -> set[str] | None:
    if not include_tables:
        return None
    names = set()
    for item in include_tables:
        item = item.strip()
        if "." in item:
            item_schema, item_table = item.split(".", 1)
            if item_schema != schema:
                continue
            names.add(item_table)
        else:
            names.add(item)
    return names


def _sample_categorical_values(engine: Engine, info: SchemaInfo, max_distinct: int) -> None:
    """Record the distinct values of low-cardinality text columns (e.g. status codes).

    Knowing that ``orders.status`` holds 'delivered' (not 'Delivered' or 'complete')
    is one of the biggest accuracy wins for Text-to-SQL. Columns that look like
    PII or free text are skipped, and each query runs under the read-only
    connection with a short timeout.
    """
    preparer = engine.dialect.identifier_preparer
    with engine.connect() as conn:
        conn.execute(text("SET statement_timeout = '3s'"))
        for table in info.tables.values():
            if table.row_estimate is not None and table.row_estimate > 5_000_000:
                continue  # too big to scan casually
            for col in table.columns:
                if col.is_primary_key or PII_NAME_PATTERN.search(col.name):
                    continue
                if not col.data_type.upper().startswith(TEXT_TYPES):
                    continue
                qualified = f"{preparer.quote_schema(table.schema)}.{preparer.quote(table.name)}"
                column = preparer.quote(col.name)
                sql = (
                    f"SELECT {column}::text FROM {qualified} WHERE {column} IS NOT NULL "
                    f"GROUP BY {column} ORDER BY count(*) DESC LIMIT {max_distinct + 1}"
                )
                try:
                    values = [v for v in conn.execute(text(sql)).scalars().all()]
                except Exception as exc:  # noqa: BLE001
                    logger.info("Skipping value sampling for %s.%s: %s", table.name, col.name, exc)
                    conn.rollback()
                    conn.execute(text("SET statement_timeout = '3s'"))
                    continue
                if 0 < len(values) <= max_distinct and all(len(v) <= 60 for v in values):
                    col.sample_values = sorted(values)
        conn.rollback()
