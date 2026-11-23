"""Tests for result normalisation, summaries and configuration."""

from __future__ import annotations

import datetime as dt
import decimal

import pandas as pd
import pytest

from app.config import Settings
from app.executor import QueryResult, _dedupe_columns
from app.llm import HashingEmbeddingFunction
from app.results import fallback_summary, format_value, normalize_dataframe, summarize_result


def test_normalize_converts_decimal_and_dates():
    df = pd.DataFrame(
        {
            "revenue": [decimal.Decimal("10.50"), decimal.Decimal("3.25"), None],
            "day": [dt.date(2025, 1, 1), dt.date(2025, 1, 2), None],
            "name": ["a", "b", "c"],
        }
    )
    out = normalize_dataframe(df)
    assert pd.api.types.is_float_dtype(out["revenue"]) and out["revenue"].iloc[0] == 10.5
    assert pd.api.types.is_datetime64_any_dtype(out["day"])
    assert out["name"].tolist() == ["a", "b", "c"]


def test_duplicate_column_names_are_renamed():
    assert _dedupe_columns(["id", "name", "id", "id"]) == ["id", "name", "id_1", "id_2"]


def _result(df, truncated=False):
    return QueryResult(sql="SELECT 1", df=df, row_count=len(df), truncated=truncated, execution_ms=1.0)


def test_fallback_summary_single_row():
    text = fallback_summary(_result(pd.DataFrame({"total_revenue": [1234567.891]})))
    assert "total revenue: 1,234,567.89" in text


def test_fallback_summary_many_rows():
    text = fallback_summary(_result(pd.DataFrame({"a": range(5), "b": range(5)}), truncated=True))
    assert "5 rows" in text and "limited" in text


def test_fallback_summary_empty():
    assert "no rows" in fallback_summary(_result(pd.DataFrame({"a": []})))


def test_summarize_sends_limited_rows(vn):
    vn.responses = ["Books lead with 1,000.00 in revenue."]
    df = pd.DataFrame({"category": [f"c{i}" for i in range(100)], "revenue": range(100)})
    text = summarize_result(vn, "Revenue by category?", "SELECT ...", _result(df), max_rows=10)
    assert text.startswith("Books lead")
    sent = vn.prompts[-1][-1]["content"]
    assert "first 10 of 100" in sent and "c9" in sent and "c10" not in sent


def test_format_value():
    assert format_value(1234) == "1,234"
    assert format_value(1234.5) == "1,234.50"
    assert format_value(None) == "—"
    assert format_value(pd.Timestamp("2025-03-01")) == "2025-03-01"


def test_hashing_embeddings_are_normalised_and_similar_for_related_text():
    ef = HashingEmbeddingFunction()
    a, b, c = ef(["top customers by revenue", "customers ranked by revenue", "weather in paris"])

    def dot(x, y):
        return sum(i * j for i, j in zip(x, y, strict=True))

    assert abs(dot(a, a) - 1) < 1e-9
    assert dot(a, b) > dot(a, c)


# ------------------------------------------------------------------ config
def test_settings_from_env_parts(monkeypatch, tmp_path):
    for var in ["DATABASE_URL", "LLM_API_KEY", "OPENAI_API_KEY"]:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("DB_HOST", "db.example")
    monkeypatch.setenv("DB_NAME", "shop")
    monkeypatch.setenv("DB_READONLY_USER", "reader")
    monkeypatch.setenv("DB_READONLY_PASSWORD", "p@ss:w/rd#1")
    monkeypatch.setenv("LLM_PROVIDER", "anthropic")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("MAX_ROWS", "250")
    monkeypatch.setenv("ALLOWED_TABLES", "orders, customers")
    s = Settings.from_env(tmp_path / "missing.env")
    from sqlalchemy.engine import make_url

    url = make_url(s.database_url)
    assert url.password == "p@ss:w/rd#1" and url.host == "db.example" and url.database == "shop"
    assert "p@ss" not in s.safe_database_url
    assert s.llm_api_key == "sk-test" and s.max_rows == 250
    assert s.allowed_tables == ["orders", "customers"]


def test_invalid_provider_rejected(monkeypatch, tmp_path):
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@h/d")
    monkeypatch.setenv("LLM_PROVIDER", "nonsense")
    with pytest.raises(ValueError):
        Settings.from_env(tmp_path / "missing.env")


def test_missing_database_rejected(monkeypatch, tmp_path):
    for var in ["DATABASE_URL", "DB_HOST", "DB_NAME"]:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(ValueError):
        Settings.from_env(tmp_path / "missing.env")
