"""Tests for schema -> knowledge base ingestion, training and retrieval."""

from __future__ import annotations

import pytest

from app.knowledge_base import (
    add_verified_example,
    build_training_plan,
    expand_with_related_tables,
    knowledge_base_counts,
    load_business_documents,
    load_sql_examples,
    retrieve_context,
    train,
)


def test_business_documents_split_by_section(knowledge_dir):
    docs = load_business_documents(knowledge_dir / "business_definitions.md")
    assert len(docs) >= 8
    revenue = next(d for d in docs if d.startswith("Revenue"))
    assert "cancelled" in revenue and "returned" in revenue


def test_sql_examples_load(knowledge_dir):
    examples = load_sql_examples(knowledge_dir / "sql_examples.yaml")
    questions = [e["question"] for e in examples]
    assert "Show the top 10 customers by revenue." in questions
    assert all(e["sql"].strip() for e in examples)


def test_all_shipped_examples_pass_validation(knowledge_dir, validator):
    for ex in load_sql_examples(knowledge_dir / "sql_examples.yaml"):
        result = validator.validate(ex["sql"])
        assert result.is_valid, (ex["question"], result.errors)


def test_training_plan_contents(schema_info, knowledge_dir, validator):
    plan = build_training_plan(schema_info, knowledge_dir, validator)
    assert len(plan.ddl) == 5
    assert any("FOREIGN KEY (customer_id) REFERENCES public.customers" in d for d in plan.ddl)
    docs = "\n".join(plan.documentation)
    assert "Foreign-key relationships" in docs
    assert "'delivered'" in docs  # sampled categorical values
    assert plan.examples and not plan.skipped_examples


def test_invalid_examples_are_skipped(schema_info, tmp_path, validator):
    (tmp_path / "sql_examples.yaml").write_text(
        "- question: good\n  sql: SELECT COUNT(*) FROM orders\n"
        "- question: bad\n  sql: SELECT * FROM internal.employee_salaries\n"
        "- question: evil\n  sql: DELETE FROM orders\n"
    )
    plan = build_training_plan(schema_info, tmp_path, validator)
    assert [e["question"] for e in plan.examples] == ["good"]
    assert {q for q, _ in plan.skipped_examples} == {"bad", "evil"}


def test_train_is_idempotent_and_reset_works(vn, schema_info, knowledge_dir, validator):
    plan = build_training_plan(schema_info, knowledge_dir, validator)
    train(vn, plan)
    first = knowledge_base_counts(vn)
    train(vn, plan)
    assert knowledge_base_counts(vn) == first
    assert first["ddl"] == 5 and first["sql"] == len(plan.examples)
    vn.add_documentation("stale note")
    train(vn, plan, reset=True)
    assert knowledge_base_counts(vn) == first


def test_retrieval_finds_relevant_items(vn, schema_info, knowledge_dir, validator):
    train(vn, build_training_plan(schema_info, knowledge_dir, validator))
    ctx = retrieve_context(vn, "top customers by revenue", schema_info)
    assert any("customers" in d for d in ctx.ddl)
    assert any("Revenue" in d for d in ctx.documentation)
    assert any("customer" in e["question"].lower() for e in ctx.examples)


def test_fk_expansion_adds_join_partners(schema_info):
    ddl = [schema_info.tables["order_items"].to_ddl()]
    expanded = expand_with_related_tables(ddl, schema_info)
    names = " ".join(expanded)
    assert "CREATE TABLE public.orders" in names and "CREATE TABLE public.products" in names


def test_retrieval_respects_allowed_tables(vn, schema_info, knowledge_dir, validator):
    train(vn, build_training_plan(schema_info, knowledge_dir, validator))
    ctx = retrieve_context(vn, "revenue by customer", schema_info, allowed_tables=["public.orders"])
    assert ctx.ddl and all("public.orders" in d.split("\n")[1] or "CREATE TABLE public.orders" in d for d in ctx.ddl)


def test_add_verified_example(vn, validator):
    add_verified_example(
        vn, validator, "How many delivered orders?", "SELECT COUNT(*) FROM orders WHERE status = 'delivered'"
    )
    assert knowledge_base_counts(vn)["sql"] == 1
    with pytest.raises(ValueError):
        add_verified_example(vn, validator, "evil", "DROP TABLE orders")
