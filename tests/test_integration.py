"""Integration tests against the real sample PostgreSQL database.

Run with:  TEST_DATABASE_URL=postgresql+psycopg://text2sql_reader:<pw>@localhost:5432/shopdb pytest -m integration
(``make test-integration`` does this against the docker-compose database.)
"""

from __future__ import annotations

import pandas as pd
import pytest

from app.executor import QueryExecutionError, QueryExecutor
from app.knowledge_base import knowledge_base_counts, load_sql_examples
from app.schema_inspector import inspect_schema, list_schemas
from app.services import build_services
from app.sql_validator import SQLValidator
from fakes import make_settings

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def schema(engine):
    return inspect_schema(engine, "public")


# ------------------------------------------------------------------ connection & schema
def test_role_is_read_only(engine):
    from app.db import check_connection

    status = check_connection(engine)
    assert status.ok and status.read_only, status.warnings


def test_internal_schema_is_invisible(engine):
    assert "internal" not in list_schemas(engine)
    assert "public" in list_schemas(engine)


def test_schema_inspection(schema):
    assert schema.table_names == ["categories", "customers", "order_items", "orders", "products"]
    orders = schema.tables["orders"]
    assert orders.primary_key == ["order_id"]
    assert orders.foreign_keys[0].referred_table == "customers"
    assert orders.column("status").sample_values == ["cancelled", "delivered", "pending", "returned", "shipped"]
    assert orders.column("order_date").comment
    assert schema.tables["customers"].column("email").sample_values == []  # PII never sampled
    assert len(schema.relationships()) == 4


# ------------------------------------------------------------------ execution safety
@pytest.fixture(scope="module")
def executor(engine):
    return QueryExecutor(engine, timeout_seconds=2, max_rows=100)


def test_select_and_like_with_percent(executor):
    result = executor.execute("SELECT name FROM categories WHERE name LIKE 'B%' ORDER BY name")
    assert result.df["name"].tolist() == ["Beauty", "Books"]


def test_numeric_columns_are_floats(executor):
    result = executor.execute("SELECT SUM(unit_price) AS total FROM products")
    assert pd.api.types.is_float_dtype(result.df["total"])


def test_writes_rejected_by_database_even_without_validator(executor):
    with pytest.raises(QueryExecutionError) as err:
        executor.execute("DELETE FROM orders")
    assert err.value.sqlstate == "25006" and not err.value.retryable


def test_sensitive_schema_permission_denied(executor):
    with pytest.raises(QueryExecutionError) as err:
        executor.execute("SELECT * FROM internal.employee_salaries")
    assert err.value.sqlstate == "42501"


def test_timeout(executor):
    with pytest.raises(QueryExecutionError) as err:
        executor.execute("SELECT pg_sleep(5)")
    assert err.value.is_timeout


def test_row_cap_and_truncation_flag(executor):
    result = executor.execute("SELECT * FROM orders")
    assert result.row_count == 100 and result.truncated


def test_every_example_query_runs(engine, knowledge_dir, schema):
    validator = SQLValidator(schema.qualified_table_names, max_rows=1000)
    executor = QueryExecutor(engine, timeout_seconds=15, max_rows=1000)
    for ex in load_sql_examples(knowledge_dir / "sql_examples.yaml"):
        check = validator.validate(ex["sql"])
        assert check.is_valid, (ex["question"], check.errors)
        safe = executor.execute(check.sql)
        original = executor.execute(ex["sql"])
        # the regenerated SQL must return exactly what the hand-written SQL returns
        pd.testing.assert_frame_equal(safe.df, original.df, check_exact=False)


# ------------------------------------------------------------------ full stack with a fake LLM server
@pytest.fixture(scope="module")
def fake_llm():
    from fake_llm_server import start_server

    server, url = start_server()
    yield url
    server.shutdown()


@pytest.fixture(scope="module")
def services(db_url, fake_llm, tmp_path_factory, engine):
    settings = make_settings(
        tmp_path_factory.mktemp("kb"),
        database_url=db_url,
        llm_provider="openai_compatible",
        llm_base_url=fake_llm,
        llm_api_key="fake",
        llm_model="fake",
        auto_train=True,
    )
    return build_services(settings)


def test_auto_training(services):
    counts = knowledge_base_counts(services.vn)
    assert counts["ddl"] == 5 and counts["sql"] >= 10 and counts["documentation"] >= 10


def test_end_to_end_question(services):
    resp = services.agent.ask("Show the top 10 customers by revenue.")
    assert resp.ok, resp.error
    assert resp.row_count == 10 and "revenue" in resp.df.columns
    assert resp.chart.chart_type == "bar"
    assert resp.summary.startswith("(fake LLM summary)")


def test_end_to_end_self_correction(services):
    resp = services.agent.ask("Which products have the highest sales?")
    assert resp.ok, resp.error
    assert [a.stage for a in resp.attempts] == ["execution", "success"]
    assert "product_name" in resp.attempts[0].error


def test_end_to_end_blocks_destructive_request(services):
    resp = services.agent.ask("Delete all cancelled orders")
    assert resp.status == "blocked"
    assert services.executor.execute("SELECT COUNT(*) AS n FROM orders").df["n"].iloc[0] == 9000


def test_end_to_end_forbidden_table(services):
    resp = services.agent.ask("What is the average employee salary?")
    assert resp.status == "cannot_answer"
    assert resp.attempts[0].stage == "validation"


def test_end_to_end_follow_up_conversation(services):
    history = []
    r1 = services.agent.ask("Show total revenue by month.", history)
    assert r1.ok and r1.row_count >= 30
    history.append(r1.to_turn())

    r2 = services.agent.ask("Now only for 2025.", history)
    assert r2.ok and r2.standalone_question == "Show total revenue by month for 2025."
    assert r2.row_count == 12
    history.append(r2.to_turn())

    r3 = services.agent.ask("Compare it with 2024.", history)
    assert r3.ok and r3.standalone_question == "Compare monthly revenue in 2025 with 2024."
    assert {"revenue_2024", "revenue_2025"} <= set(r3.df.columns)
    assert r3.chart.chart_type == "line"
