"""Test doubles: a scripted LLM, a fake executor and a sample schema."""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd

from app.config import Settings
from app.executor import QueryExecutionError, QueryResult
from app.llm import _ChatMixin, create_vanna
from app.schema_inspector import ColumnInfo, ForeignKeyInfo, SchemaInfo, TableInfo

APPROVED = ["public.categories", "public.customers", "public.order_items", "public.orders", "public.products"]


class ScriptedChat(_ChatMixin):
    """LLM adapter that returns queued responses (or calls ``handler``) and records prompts."""

    def _init_llm(self, settings: Settings) -> None:
        self.responses: list = []
        self.prompts: list[list[dict]] = []
        self.handler: Callable[[list[dict]], str] | None = None

    def submit_prompt(self, prompt, **kwargs) -> str:
        self.prompts.append(prompt)
        if self.handler is not None:
            return self.handler(prompt)
        if not self.responses:
            raise AssertionError("Unexpected LLM call:\n" + prompt[-1]["content"][:500])
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


def make_settings(tmp_path, **overrides) -> Settings:
    base = dict(
        database_url="postgresql+psycopg://reader:pw@localhost/testdb",
        embedding_provider="hashing",
        vector_db_path=str(tmp_path / "chroma"),
        llm_provider="openai",
        llm_model="test-model",
        auto_train=False,
    )
    base.update(overrides)
    return Settings(**base)


def make_vanna(tmp_path, settings: Settings | None = None):
    settings = settings or make_settings(tmp_path)
    return create_vanna(settings, "testdb", "public", adapter=ScriptedChat)


def sql_block(sql: str) -> str:
    return f"```sql\n{sql}\n```"


class FakeExecutor:
    """Returns queued DataFrames or raises queued QueryExecutionErrors."""

    def __init__(self, outcomes: list | None = None, max_rows: int = 1000) -> None:
        self.outcomes = list(outcomes or [])
        self.executed: list[str] = []
        self.max_rows = max_rows

    def execute(self, sql: str) -> QueryResult:
        self.executed.append(sql)
        outcome = self.outcomes.pop(0) if self.outcomes else pd.DataFrame({"value": [1]})
        if isinstance(outcome, Exception):
            raise outcome
        return QueryResult(sql=sql, df=outcome, row_count=len(outcome), truncated=False, execution_ms=12.5)


def db_error(message: str, sqlstate: str = "42703") -> QueryExecutionError:
    return QueryExecutionError(message, sqlstate=sqlstate)


def sample_schema_info() -> SchemaInfo:
    """Mirror of the sample database schema (no DB needed)."""

    def col(name, dtype, pk=False, comment=None, values=None):
        return ColumnInfo(
            name=name, data_type=dtype, nullable=not pk, is_primary_key=pk, comment=comment, sample_values=values or []
        )

    tables = {
        "categories": TableInfo(
            "public",
            "categories",
            [col("category_id", "INTEGER", pk=True), col("name", "TEXT"), col("description", "TEXT")],
            primary_key=["category_id"],
            comment="Product categories",
        ),
        "products": TableInfo(
            "public",
            "products",
            [
                col("product_id", "INTEGER", pk=True),
                col("sku", "TEXT"),
                col("name", "TEXT"),
                col("category_id", "INTEGER"),
                col("unit_price", "NUMERIC(10, 2)"),
                col("unit_cost", "NUMERIC(10, 2)"),
                col("is_active", "BOOLEAN"),
                col("created_at", "DATE"),
            ],
            primary_key=["product_id"],
            foreign_keys=[ForeignKeyInfo(["category_id"], "public", "categories", ["category_id"])],
            comment="Product catalog",
        ),
        "customers": TableInfo(
            "public",
            "customers",
            [
                col("customer_id", "INTEGER", pk=True),
                col("first_name", "TEXT"),
                col("last_name", "TEXT"),
                col("email", "TEXT"),
                col("city", "TEXT"),
                col("state", "TEXT"),
                col("segment", "TEXT", values=["Consumer", "Corporate", "Small Business"]),
                col("signup_date", "DATE"),
            ],
            primary_key=["customer_id"],
            comment="Customers",
        ),
        "orders": TableInfo(
            "public",
            "orders",
            [
                col("order_id", "INTEGER", pk=True),
                col("customer_id", "INTEGER"),
                col("order_date", "TIMESTAMP"),
                col(
                    "status",
                    "TEXT",
                    comment="Order status",
                    values=["cancelled", "delivered", "pending", "returned", "shipped"],
                ),
                col("sales_channel", "TEXT", values=["mobile", "store", "web"]),
                col("shipping_cost", "NUMERIC(10, 2)"),
            ],
            primary_key=["order_id"],
            foreign_keys=[ForeignKeyInfo(["customer_id"], "public", "customers", ["customer_id"])],
            comment="Order headers",
        ),
        "order_items": TableInfo(
            "public",
            "order_items",
            [
                col("order_item_id", "INTEGER", pk=True),
                col("order_id", "INTEGER"),
                col("product_id", "INTEGER"),
                col("quantity", "INTEGER"),
                col("unit_price", "NUMERIC(10, 2)"),
                col("discount", "NUMERIC(4, 3)"),
            ],
            primary_key=["order_item_id"],
            foreign_keys=[
                ForeignKeyInfo(["order_id"], "public", "orders", ["order_id"]),
                ForeignKeyInfo(["product_id"], "public", "products", ["product_id"]),
            ],
            comment="Order line items",
        ),
    }
    return SchemaInfo(schema="public", tables=tables)
