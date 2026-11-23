"""Tests for the SQL safety layer."""

from __future__ import annotations

import pytest

from app.sql_validator import ErrorCode, SQLValidator

# ------------------------------------------------------------------ allowed queries
VALID_QUERIES = [
    "SELECT * FROM orders",
    "select order_id, status from orders where status = 'delivered'",
    "SELECT o.order_id FROM public.orders AS o",
    'SELECT o.order_id FROM "orders" o',
    "SELECT c.name, SUM(oi.quantity * oi.unit_price) AS revenue FROM order_items oi "
    "JOIN products p ON p.product_id = oi.product_id JOIN categories c ON c.category_id = p.category_id "
    "GROUP BY c.name ORDER BY revenue DESC LIMIT 5",
    "WITH monthly AS (SELECT date_trunc('month', order_date)::date AS month, COUNT(*) AS n FROM orders GROUP BY 1) "
    "SELECT * FROM monthly ORDER BY month",
    "SELECT status FROM orders UNION SELECT segment FROM customers",
    "SELECT customer_id, RANK() OVER (ORDER BY COUNT(*) DESC) FROM orders GROUP BY customer_id",
    "SELECT g::date AS day FROM generate_series(CURRENT_DATE - 30, CURRENT_DATE, INTERVAL '1 day') AS g",
    "SELECT email FROM customers WHERE email LIKE '%@example.com' AND first_name ILIKE 'a%'",
    "SELECT COUNT(*) FILTER (WHERE status = 'returned') FROM orders",
    "SELECT * FROM orders WHERE status = 'DROP TABLE orders; --'",  # keyword inside a string literal
    "SELECT $$delete everything$$ AS note FROM orders",  # dollar-quoted string
    "SELECT c.customer_id FROM customers c WHERE NOT EXISTS "
    "(SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id AND o.order_date >= CURRENT_DATE - 90)",
    "SELECT * FROM orders;",  # trailing semicolon is fine
    "/* leading comment */ SELECT 1",
    "SELECT * FROM orders -- trailing comment",
    "(SELECT order_id FROM orders)",
    "SELECT EXTRACT(YEAR FROM order_date)::int AS year, COUNT(*) FROM orders GROUP BY 1",
]


@pytest.mark.parametrize("sql", VALID_QUERIES)
def test_valid_queries_pass(validator, sql):
    result = validator.validate(sql)
    assert result.is_valid, result.errors
    assert result.sql


@pytest.mark.parametrize("sql", ["/* hidden */ SELECT 1", "SELECT * FROM orders -- note", "SELECT 1 /* x */ + 1"])
def test_comments_stripped_from_executed_sql(validator, sql):
    result = validator.validate(sql)
    assert result.is_valid
    assert "--" not in result.sql and "/*" not in result.sql


# ------------------------------------------------------------------ destructive statements
DESTRUCTIVE = [
    "INSERT INTO orders (customer_id) VALUES (1)",
    "UPDATE orders SET status = 'cancelled'",
    "DELETE FROM orders",
    "DROP TABLE orders",
    "ALTER TABLE orders ADD COLUMN x int",
    "TRUNCATE orders",
    "CREATE TABLE x AS SELECT * FROM orders",
    "GRANT ALL ON orders TO public",
    "REVOKE SELECT ON orders FROM public",
    "MERGE INTO orders USING customers ON true WHEN MATCHED THEN DELETE",
    "COPY orders TO '/tmp/orders.csv'",
    "VACUUM orders",
    "DO $$ BEGIN DELETE FROM orders; END $$",
    "CALL some_procedure()",
    "SET statement_timeout = 0",
    "BEGIN",
    "LOCK TABLE orders",
    "EXPLAIN ANALYZE DELETE FROM orders",
    "PREPARE p AS DELETE FROM orders",
    "REFRESH MATERIALIZED VIEW mv",
    "REINDEX TABLE orders",
]


@pytest.mark.parametrize("sql", DESTRUCTIVE)
def test_destructive_statements_blocked(validator, sql):
    result = validator.validate(sql)
    assert not result.is_valid
    assert result.is_security_violation, result.errors


def test_lowercase_and_mixed_case_keywords_blocked(validator):
    for sql in ["delete from orders", "DeLeTe FrOm orders", "drop table customers"]:
        assert not validator.validate(sql).is_valid


# ------------------------------------------------------------------ injection patterns
INJECTION = [
    "SELECT * FROM orders; DROP TABLE orders",
    "SELECT * FROM orders; DELETE FROM orders;",
    "SELECT 1; SELECT 2",
    "SELECT * FROM customers WHERE email = '' OR 1=1; UPDATE customers SET email = 'x'",
    "WITH d AS (DELETE FROM orders RETURNING *) SELECT * FROM d",  # data-modifying CTE
    "WITH u AS (UPDATE orders SET status = 'x' RETURNING *) SELECT count(*) FROM u",
    "SELECT * INTO backup_orders FROM orders",  # creates a table
    "SELECT * FROM orders FOR UPDATE",  # takes row locks
    "SELECT * FROM orders FOR SHARE",
]


@pytest.mark.parametrize("sql", INJECTION)
def test_injection_patterns_blocked(validator, sql):
    result = validator.validate(sql)
    assert not result.is_valid
    assert result.is_security_violation


def test_multiple_statements_error_code(validator):
    result = validator.validate("SELECT 1; SELECT 2")
    assert result.issues[0].code == ErrorCode.MULTIPLE_STATEMENTS


# ------------------------------------------------------------------ dangerous functions
DANGEROUS_FUNCTIONS = [
    "SELECT pg_sleep(30)",
    "SELECT pg_catalog.pg_sleep(30)",
    "SELECT pg_read_file('/etc/passwd')",
    "SELECT pg_terminate_backend(123)",
    "SELECT set_config('statement_timeout', '0', false)",
    "SELECT current_setting('data_directory')",
    "SELECT nextval('orders_order_id_seq')",
    "SELECT setval('orders_order_id_seq', 1)",
    "SELECT query_to_xml('DELETE FROM orders RETURNING 1', true, false, '')",
    "SELECT dblink_exec('dbname=x', 'DROP TABLE orders')",
    "SELECT lo_import('/etc/passwd')",
    "SELECT version()",
    "SELECT current_user",
    "SELECT current_database()",
    "SELECT * FROM orders WHERE pg_sleep(1) IS NOT NULL",
]


@pytest.mark.parametrize("sql", DANGEROUS_FUNCTIONS)
def test_dangerous_functions_blocked(validator, sql):
    result = validator.validate(sql)
    assert not result.is_valid
    assert result.issues[0].code == ErrorCode.FORBIDDEN_FUNCTION


# ------------------------------------------------------------------ table allow-list
@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM internal.employee_salaries",
        "SELECT * FROM employee_salaries",
        "SELECT * FROM orders o JOIN secret_table s ON s.id = o.order_id",
        "SELECT * FROM other_schema.orders",
        "SELECT * FROM otherdb.public.orders",
        'SELECT * FROM "Orders"',  # quoted identifiers are case-sensitive in Postgres
    ],
)
def test_unapproved_tables_rejected(validator, sql):
    result = validator.validate(sql)
    assert not result.is_valid
    assert result.issues[0].code == ErrorCode.DISALLOWED_TABLE
    assert not result.is_security_violation  # fixable by the LLM -> eligible for retry


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM pg_catalog.pg_user",
        "SELECT * FROM pg_user",
        "SELECT * FROM pg_stat_activity",
        "SELECT * FROM information_schema.tables",
        "SELECT usename FROM pg_shadow",
    ],
)
def test_system_catalogs_rejected(validator, sql):
    result = validator.validate(sql)
    assert not result.is_valid
    assert result.issues[0].code == ErrorCode.SYSTEM_CATALOG


def test_cte_names_are_not_treated_as_tables(validator):
    sql = "WITH secret_table AS (SELECT order_id FROM orders) SELECT * FROM secret_table"
    assert validator.validate(sql).is_valid


def test_cte_cannot_hide_unapproved_table(validator):
    sql = "WITH x AS (SELECT * FROM internal.employee_salaries) SELECT * FROM x"
    assert not validator.validate(sql).is_valid


def test_subquery_tables_are_checked(validator):
    sql = "SELECT * FROM orders WHERE customer_id IN (SELECT id FROM hidden_users)"
    assert not validator.validate(sql).is_valid


def test_referenced_tables_reported(validator):
    result = validator.validate("SELECT * FROM orders o JOIN customers c ON c.customer_id = o.customer_id")
    assert result.tables == ["public.customers", "public.orders"]


def test_allow_list_accepts_bare_and_qualified_names():
    v = SQLValidator(["orders", "public.customers"])
    assert v.validate("SELECT * FROM public.orders").is_valid
    assert v.validate("SELECT * FROM customers").is_valid


def test_non_default_schema():
    v = SQLValidator(["sales.orders"], default_schema="sales")
    assert v.validate("SELECT * FROM orders").is_valid
    assert not v.validate("SELECT * FROM public.orders").is_valid


# ------------------------------------------------------------------ row limits
def test_limit_added_when_missing(validator):
    result = validator.validate("SELECT * FROM orders")
    assert result.limit_applied
    assert result.sql.rstrip().endswith("LIMIT 1000")


def test_small_limit_kept(validator):
    result = validator.validate("SELECT * FROM orders LIMIT 10")
    assert not result.limit_applied
    assert "LIMIT 10" in result.sql


def test_large_limit_reduced(validator):
    result = validator.validate("SELECT * FROM orders LIMIT 500000")
    assert result.limit_applied
    assert "LIMIT 1000" in result.sql and "500000" not in result.sql


def test_limit_all_replaced(validator):
    assert "LIMIT 1000" in validator.validate("SELECT * FROM orders LIMIT ALL").sql


def test_fetch_first_respected(validator):
    small = validator.validate("SELECT * FROM orders FETCH FIRST 5 ROWS ONLY")
    assert not small.limit_applied
    big = validator.validate("SELECT * FROM orders FETCH FIRST 99999 ROWS ONLY")
    assert big.limit_applied and "99999" not in big.sql


def test_limit_applies_to_union(validator):
    result = validator.validate("SELECT status FROM orders UNION SELECT segment FROM customers")
    assert result.sql.rstrip().endswith("LIMIT 1000")


def test_offset_preserved(validator):
    result = validator.validate("SELECT * FROM orders ORDER BY order_id LIMIT 10 OFFSET 20")
    assert "OFFSET 20" in result.sql


def test_inner_limit_does_not_count(validator):
    result = validator.validate("SELECT * FROM (SELECT * FROM orders LIMIT 5) t")
    assert result.limit_applied


# ------------------------------------------------------------------ malformed input
@pytest.mark.parametrize("sql", ["", "   ", None])
def test_empty_sql(validator, sql):
    result = validator.validate(sql)
    assert not result.is_valid
    assert result.issues[0].code == ErrorCode.EMPTY


def test_syntax_error(validator):
    result = validator.validate("SELECT FROM WHERE")
    assert not result.is_valid
    assert result.issues[0].code == ErrorCode.PARSE_ERROR


def test_too_long(validator):
    result = validator.validate("SELECT " + "1 + " * 6000 + "1")
    assert result.issues[0].code == ErrorCode.TOO_LONG


def test_nul_byte(validator):
    assert not validator.validate("SELECT 1\x00; DROP TABLE orders").is_valid


def test_natural_language_is_not_sql(validator):
    assert not validator.validate("Sorry, I cannot answer that question.").is_valid
