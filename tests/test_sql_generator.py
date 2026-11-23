"""Tests for prompt building, SQL extraction, follow-up rewriting and repair prompts."""

from __future__ import annotations

import datetime as dt

import pytest

from app.knowledge_base import RetrievedContext
from app.sql_generator import SQLGenerator, Turn, extract_sql, parse_cannot_answer
from fakes import APPROVED, sql_block

TODAY = dt.date(2026, 11, 22)


@pytest.fixture
def context(schema_info):
    return RetrievedContext(
        ddl=[schema_info.tables["orders"].to_ddl(), schema_info.tables["order_items"].to_ddl()],
        documentation=["Revenue (net sales):\nSUM(oi.quantity * oi.unit_price * (1 - oi.discount))"],
        examples=[{"question": "How many orders are there?", "sql": "SELECT COUNT(*) FROM orders"}],
    )


@pytest.fixture
def generator(vn):
    return SQLGenerator(vn, allowed_tables=APPROVED, max_rows=500, history_turns=3, today=TODAY)


# ------------------------------------------------------------------ extraction
@pytest.mark.parametrize(
    "response,expected",
    [
        ("```sql\nSELECT 1\n```", "SELECT 1"),
        ("```SQL\nSELECT 1;\n```", "SELECT 1"),
        ("```\nSELECT 1\n```", "SELECT 1"),
        ("```postgresql\nWITH x AS (SELECT 1) SELECT * FROM x;\n```", "WITH x AS (SELECT 1) SELECT * FROM x"),
        ("Here is the query:\n```sql\nSELECT a FROM t\n```\nIt returns a.", "SELECT a FROM t"),
        ("SELECT * FROM orders;", "SELECT * FROM orders"),
        ("Sure! SELECT * FROM orders; This lists orders.", "SELECT * FROM orders"),
        ("First:\n```sql\nSELECT 1\n```\nFinal:\n```sql\nSELECT 2\n```", "SELECT 2"),
        ("I am not sure what you mean.", None),
        ("```sql\nDELETE FROM orders\n```", "DELETE FROM orders"),
        ("DROP TABLE orders;", "DROP TABLE orders"),
        ("", None),
    ],
)
def test_extract_sql(response, expected):
    assert extract_sql(response) == expected


def test_parse_cannot_answer():
    assert parse_cannot_answer("CANNOT_ANSWER: no salary data") == "no salary data"
    assert parse_cannot_answer("cannot_answer: read-only") == "read-only"
    assert parse_cannot_answer("```sql\nSELECT 1\n```") is None


# ------------------------------------------------------------------ prompt
def test_prompt_contains_rules_date_tables_and_context(generator, context):
    prompt = generator.build_prompt("What is total revenue?", context, [])
    system = prompt[0]["content"]
    assert prompt[0]["role"] == "system"
    assert "2026-11-22" in system and "Sunday" in system
    for table in APPROVED:
        assert table in system
    assert "CREATE TABLE public.orders" in system
    assert "Revenue (net sales)" in system
    assert "How many orders are there?" in system
    assert "LIMIT of at most 500" in system
    assert prompt[-1] == {"role": "user", "content": "What is total revenue?"}


def test_prompt_includes_recent_history_as_turns(generator, context):
    history = [
        Turn(
            "Show total revenue by month.",
            "Show total revenue by month.",
            "SELECT month, revenue FROM x",
            ["month", "revenue"],
            35,
        ),
        Turn("broken", "broken", None),  # failed turns are skipped
    ]
    prompt = generator.build_prompt("Now only for 2025.", context, history)
    roles = [m["role"] for m in prompt]
    assert roles == ["system", "user", "assistant", "user"]
    assert prompt[1]["content"] == "Show total revenue by month."
    assert "SELECT month, revenue FROM x" in prompt[2]["content"]
    assert "35 rows" in prompt[2]["content"]


def test_history_is_capped(generator, context):
    history = [Turn(f"q{i}", f"q{i}", f"SELECT {i}") for i in range(10)]
    prompt = generator.build_prompt("next", context, history)
    users = [m["content"] for m in prompt if m["role"] == "user"]
    assert users == ["q7", "q8", "q9", "next"]


def test_generate_returns_sql(generator, context, vn):
    vn.responses = [sql_block("SELECT COUNT(*) FROM orders")]
    result = generator.generate("How many orders?", context)
    assert result.sql == "SELECT COUNT(*) FROM orders"
    assert not result.cannot_answer


def test_generate_cannot_answer(generator, context, vn):
    vn.responses = ["CANNOT_ANSWER: there is no weather data"]
    result = generator.generate("What's the weather?", context)
    assert result.cannot_answer and "weather" in result.cannot_answer_reason
    assert result.sql is None


def test_generate_without_sql_is_cannot_answer(generator, context, vn):
    vn.responses = ["Hmm, interesting question."]
    assert generator.generate("?", context).cannot_answer


def test_repair_prompt_contains_error_and_failed_sql(generator, context, vn):
    vn.responses = [sql_block("SELECT order_id FROM orders")]
    result = generator.repair("q", context, [], "SELECT nope FROM orders", 'column "nope" does not exist')
    sent = vn.prompts[-1]
    assert sent[-2]["role"] == "assistant" and "SELECT nope FROM orders" in sent[-2]["content"]
    assert 'column "nope" does not exist' in sent[-1]["content"]
    assert result.sql == "SELECT order_id FROM orders"


# ------------------------------------------------------------------ follow-ups
def test_rewrite_without_history_makes_no_llm_call(generator, vn):
    assert generator.rewrite_question("Top products?", []) == "Top products?"
    assert vn.prompts == []


def test_rewrite_follow_up(generator, vn):
    vn.responses = ["Show total revenue by month for 2025."]
    history = [Turn("Show total revenue by month.", "Show total revenue by month.", "SELECT 1", ["m"], 3)]
    assert generator.rewrite_question("Now only for 2025.", history) == "Show total revenue by month for 2025."
    sent = vn.prompts[-1][-1]["content"]
    assert "Show total revenue by month." in sent and "Latest question: Now only for 2025." in sent


def test_rewrite_falls_back_on_llm_error(generator, vn):
    vn.responses = [RuntimeError("provider down")]
    history = [Turn("a", "a", "SELECT 1")]
    assert generator.rewrite_question("and b?", history) == "and b?"


def test_rewrite_ignores_sql_output(generator, vn):
    vn.responses = ["SELECT * FROM orders"]
    history = [Turn("a", "a", "SELECT 1")]
    assert generator.rewrite_question("and b?", history) == "and b?"
