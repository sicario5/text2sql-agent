"""Tests for the agent loop: retries, blocking, follow-ups and error handling."""

from __future__ import annotations

import datetime as dt

import pandas as pd
import pytest

from app.agent import TextToSQLAgent
from app.knowledge_base import build_training_plan, train
from app.llm import LLMError
from app.sql_generator import SQLGenerator, Turn
from fakes import APPROVED, FakeExecutor, db_error, sql_block


@pytest.fixture
def trained_vn(vn, schema_info, knowledge_dir, validator):
    train(vn, build_training_plan(schema_info, knowledge_dir, validator))
    return vn


def make_agent(vn, validator, executor, schema_info, retries=2, summary=False):
    generator = SQLGenerator(vn, APPROVED, max_rows=1000, today=dt.date(2026, 11, 22))
    return TextToSQLAgent(
        vn, generator, validator, executor, schema_info=schema_info, max_retries=retries, summary_enabled=summary
    )


def test_success_path(trained_vn, validator, schema_info):
    df = pd.DataFrame({"total_revenue": [1234.5]})
    executor = FakeExecutor([df])
    trained_vn.responses = [sql_block("SELECT SUM(unit_price) AS total_revenue FROM order_items")]
    resp = make_agent(trained_vn, validator, executor, schema_info).ask("Total revenue?")
    assert resp.ok and resp.status == "success"
    assert resp.row_count == 1 and resp.chart.chart_type == "kpi"
    assert len(resp.attempts) == 1
    assert executor.executed and "LIMIT 1000" in executor.executed[0]  # validated SQL was executed
    assert resp.context_counts["ddl"] > 0 and resp.context_counts["examples"] > 0


def test_retry_after_validation_error(trained_vn, validator, schema_info):
    executor = FakeExecutor([pd.DataFrame({"n": [5]})])
    trained_vn.responses = [
        sql_block("SELECT COUNT(*) AS n FROM sales"),  # hallucinated table
        sql_block("SELECT COUNT(*) AS n FROM orders"),
    ]
    resp = make_agent(trained_vn, validator, executor, schema_info).ask("How many orders?")
    assert resp.ok
    assert [a.stage for a in resp.attempts] == ["validation", "success"]
    repair_prompt = trained_vn.prompts[-1][-1]["content"]
    assert "public.sales" in repair_prompt and "Approved tables are" in repair_prompt


def test_retry_after_execution_error(trained_vn, validator, schema_info):
    executor = FakeExecutor([db_error('column "revenue" does not exist'), pd.DataFrame({"x": [1, 2]})])
    trained_vn.responses = [sql_block("SELECT revenue FROM orders"), sql_block("SELECT order_id AS x FROM orders")]
    resp = make_agent(trained_vn, validator, executor, schema_info).ask("q")
    assert resp.ok
    assert [a.stage for a in resp.attempts] == ["execution", "success"]
    assert 'column "revenue" does not exist' in trained_vn.prompts[-1][-1]["content"]


def test_gives_up_after_max_retries(trained_vn, validator, schema_info):
    executor = FakeExecutor([db_error("boom")] * 5)
    trained_vn.responses = [sql_block("SELECT order_id FROM orders")] * 3
    resp = make_agent(trained_vn, validator, executor, schema_info, retries=2).ask("q")
    assert resp.status == "error"
    assert len(resp.attempts) == 3 and len(executor.executed) == 3
    assert "after 3 attempts" in resp.error


def test_zero_retries(trained_vn, validator, schema_info):
    executor = FakeExecutor([db_error("boom")])
    trained_vn.responses = [sql_block("SELECT order_id FROM orders")]
    resp = make_agent(trained_vn, validator, executor, schema_info, retries=0).ask("q")
    assert resp.status == "error" and len(resp.attempts) == 1


def test_destructive_sql_is_blocked_without_execution_or_retry(trained_vn, validator, schema_info):
    executor = FakeExecutor()
    trained_vn.responses = [sql_block("DELETE FROM orders")]
    resp = make_agent(trained_vn, validator, executor, schema_info).ask("Delete all orders")
    assert resp.status == "blocked"
    assert executor.executed == []
    assert len(trained_vn.prompts) == 1  # no repair attempt for security violations
    assert "safety layer" in resp.error


def test_stacked_statement_is_blocked(trained_vn, validator, schema_info):
    executor = FakeExecutor()
    trained_vn.responses = [sql_block("SELECT 1; DROP TABLE orders")]
    resp = make_agent(trained_vn, validator, executor, schema_info).ask("q")
    assert resp.status == "blocked" and executor.executed == []


def test_non_retryable_db_error(trained_vn, validator, schema_info):
    executor = FakeExecutor([db_error("permission denied for table orders", "42501")])
    trained_vn.responses = [sql_block("SELECT * FROM orders")]
    resp = make_agent(trained_vn, validator, executor, schema_info).ask("q")
    assert resp.status == "blocked" and len(executor.executed) == 1


def test_cannot_answer(trained_vn, validator, schema_info):
    trained_vn.responses = ["CANNOT_ANSWER: the database has no employee data"]
    resp = make_agent(trained_vn, validator, FakeExecutor(), schema_info).ask("Average salary?")
    assert resp.status == "cannot_answer" and "employee" in resp.error


def test_follow_up_uses_rewritten_question(trained_vn, validator, schema_info):
    executor = FakeExecutor([pd.DataFrame({"month": pd.to_datetime(["2025-01-01", "2025-02-01"]), "revenue": [1, 2]})])
    trained_vn.responses = [
        "Show total revenue by month for 2025.",  # rewrite
        sql_block("SELECT date_trunc('month', order_date) AS month, 1 AS revenue FROM orders GROUP BY 1"),
    ]
    history = [Turn("Show total revenue by month.", "Show total revenue by month.", "SELECT 1", ["month"], 35)]
    resp = make_agent(trained_vn, validator, executor, schema_info).ask("Now only for 2025.", history)
    assert resp.ok
    assert resp.standalone_question == "Show total revenue by month for 2025."
    sql_prompt = trained_vn.prompts[1]
    assert sql_prompt[-1]["content"] == "Show total revenue by month for 2025."
    assert any(m["role"] == "assistant" and "SELECT 1" in m["content"] for m in sql_prompt)
    assert resp.chart.chart_type == "line"
    turn = resp.to_turn()
    assert turn.sql and turn.row_count == 2


def test_summary_generated(trained_vn, validator, schema_info):
    executor = FakeExecutor([pd.DataFrame({"total": [10.0]})])
    trained_vn.responses = [sql_block("SELECT 10 AS total FROM orders"), "Total revenue was 10.00."]
    resp = make_agent(trained_vn, validator, executor, schema_info, summary=True).ask("Total?")
    assert resp.summary == "Total revenue was 10.00."


def test_summary_failure_falls_back(trained_vn, validator, schema_info):
    executor = FakeExecutor([pd.DataFrame({"total": [10.0]})])
    trained_vn.responses = [sql_block("SELECT 10 AS total FROM orders"), LLMError("rate limited")]
    resp = make_agent(trained_vn, validator, executor, schema_info, summary=True).ask("Total?")
    assert resp.ok and "total: 10" in resp.summary


def test_llm_failure_is_reported(trained_vn, validator, schema_info):
    trained_vn.responses = [LLMError("invalid api key")]
    resp = make_agent(trained_vn, validator, FakeExecutor(), schema_info).ask("q")
    assert resp.status == "error" and "invalid api key" in resp.error


def test_empty_question(trained_vn, validator, schema_info):
    resp = make_agent(trained_vn, validator, FakeExecutor(), schema_info).ask("   ")
    assert resp.status == "error" and trained_vn.prompts == []


def test_unapproved_tables_never_reach_prompt(vn, schema_info, knowledge_dir):
    from app.sql_validator import SQLValidator

    full = SQLValidator(APPROVED)
    train(vn, build_training_plan(schema_info, knowledge_dir, full))
    narrow = SQLValidator(["public.orders"])
    generator = SQLGenerator(vn, ["public.orders"], today=dt.date(2026, 11, 22))
    agent = TextToSQLAgent(vn, generator, narrow, FakeExecutor(), schema_info=schema_info, summary_enabled=False)
    vn.responses = [sql_block("SELECT COUNT(*) FROM orders")]
    agent.ask("How many customers per segment?")
    system = vn.prompts[0][0]["content"]
    assert "CREATE TABLE public.orders" in system
    assert "CREATE TABLE public.customers" not in system
