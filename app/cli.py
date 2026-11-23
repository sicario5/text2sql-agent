"""Command-line interface.

python -m app.cli check                 # connection + read-only audit
python -m app.cli inspect               # print the schema the agent sees
python -m app.cli train [--reset]       # (re)build the RAG knowledge base
python -m app.cli ask "question" ...    # ask one or more questions (follow-ups share context)
python -m app.cli validate "SQL"        # run the safety validator on a SQL string
"""

from __future__ import annotations

import argparse
import logging
import sys

from .config import Settings
from .db import check_connection, create_db_engine
from .schema_inspector import inspect_schema


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="text2sql", description="Text-to-SQL RAG agent")
    parser.add_argument("--schema", help="database schema (default: DB_SCHEMA)")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("check", help="test the database connection and audit the role")
    sub.add_parser("inspect", help="show tables, columns and relationships")
    train_p = sub.add_parser("train", help="train the knowledge base")
    train_p.add_argument("--reset", action="store_true", help="clear the knowledge base first")
    ask_p = sub.add_parser("ask", help="ask questions; later questions are follow-ups")
    ask_p.add_argument("questions", nargs="+")
    val_p = sub.add_parser("validate", help="validate a SQL statement")
    val_p.add_argument("sql")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(levelname)s %(message)s")
    settings = Settings.from_env()
    schema = args.schema or settings.db_schema

    if args.command == "check":
        engine = create_db_engine(settings.database_url, settings.query_timeout_seconds)
        status = check_connection(engine)
        if not status.ok:
            print(f"✗ Connection failed: {status.error}")
            return 1
        print(f"✓ Connected to {status.database} as {status.user} (PostgreSQL {status.server_version})")
        if status.warnings:
            print("✗ Role is NOT read-only:")
            for w in status.warnings:
                print(f"   - {w}")
            return 2
        print("✓ Role is read-only")
        return 0

    if args.command == "inspect":
        engine = create_db_engine(settings.database_url, settings.query_timeout_seconds)
        info = inspect_schema(engine, schema, settings.allowed_tables or None, settings.sample_column_values)
        for table in info.tables.values():
            print(table.to_ddl())
            print()
        print(info.relationships_doc())
        return 0

    if args.command == "validate":
        from .services import connect
        from .sql_validator import SQLValidator

        engine, _ = connect(settings)
        info = inspect_schema(engine, schema, settings.allowed_tables or None, sample_values=False)
        result = SQLValidator(info.qualified_table_names, schema, settings.max_rows).validate(args.sql)
        if result.is_valid:
            print("✓ VALID" + (" (row limit added)" if result.limit_applied else ""))
            print(result.sql)
            return 0
        print("✗ REJECTED: " + result.error_message)
        return 1

    from .services import build_services

    services = build_services(settings, schema=schema)

    if args.command == "train":
        counts = services.train(reset=args.reset)
        print(f"✓ Trained knowledge base for {services.status.database}.{schema}: {counts}")
        return 0

    if args.command == "ask":
        history = []
        for question in args.questions:
            print(f"\n❓ {question}")
            response = services.agent.ask(question, history)
            if response.standalone_question != question:
                print(f"   interpreted as: {response.standalone_question}")
            if response.sql:
                print("\n" + response.sql + "\n")
            if response.ok:
                print(response.df.head(20).to_string(index=False))
                print(
                    f"\n{response.row_count} rows · {response.execution_ms:.0f} ms · chart: {response.chart.chart_type}"
                )
                print(f"\n💬 {response.summary}")
                history.append(response.to_turn())
            else:
                print(f"✗ {response.status}: {response.error}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
