# Example questions and generated SQL

Results below come from the sample database (seeded so orders run up to 2026-11-22), executed through the full pipeline: RAG retrieval → SQL → safety validator → read-only execution → chart recommendation.

> The SQL shown is the **validated SQL that was executed**. The safety layer re-generates every query from its parsed AST, so formatting is normalised and a `LIMIT` is added when missing. These question/SQL pairs also ship as few-shot training examples in `knowledge/sql_examples.yaml`. With a live LLM, SQL for new questions is generated from the retrieved schema, business definitions and these examples, so it will differ in style.

## 1. Core questions

### What were our total sales last month?

```sql
SELECT
  ROUND(SUM(oi.quantity * oi.unit_price * (
    1 - oi.discount
  )), 2) AS total_revenue
FROM orders AS o
JOIN order_items AS oi
  ON oi.order_id = o.order_id
WHERE
  NOT o.status IN ('cancelled', 'returned')
  AND o.order_date >= DATE_TRUNC('MONTH', CURRENT_DATE) - INTERVAL '1 MONTH'
  AND o.order_date < DATE_TRUNC('MONTH', CURRENT_DATE)
LIMIT 1000
```

**1 rows** · 5 ms · recommended chart: **kpi**

|   total_revenue |
|----------------:|
|       80,419.07 |

### Show the top 10 customers by revenue.

```sql
SELECT
  c.customer_id,
  c.first_name || ' ' || c.last_name AS customer_name,
  c.segment,
  ROUND(SUM(oi.quantity * oi.unit_price * (
    1 - oi.discount
  )), 2) AS revenue
FROM customers AS c
JOIN orders AS o
  ON o.customer_id = c.customer_id
JOIN order_items AS oi
  ON oi.order_id = o.order_id
WHERE
  NOT o.status IN ('cancelled', 'returned')
GROUP BY
  c.customer_id,
  customer_name,
  c.segment
ORDER BY
  revenue DESC
LIMIT 10
```

**10 rows** · 20 ms · recommended chart: **bar**

|   customer_id | customer_name   | segment        |   revenue |
|--------------:|:----------------|:---------------|----------:|
|           254 | Lucas Muller    | Consumer       |  9,938.27 |
|           304 | Linda Jones     | Small Business |  9,601.13 |
|           132 | Barbara Johnson | Consumer       |  8,741.06 |
|           228 | Sarah Garcia    | Small Business |  8,684.92 |
|            21 | Thomas Williams | Consumer       |  8,549.00 |

*(first 5 rows)*

### Which products have the highest sales?

```sql
SELECT
  p.name AS product_name,
  cat.name AS category,
  SUM(oi.quantity) AS units_sold,
  ROUND(SUM(oi.quantity * oi.unit_price * (
    1 - oi.discount
  )), 2) AS revenue
FROM order_items AS oi
JOIN orders AS o
  ON o.order_id = oi.order_id
JOIN products AS p
  ON p.product_id = oi.product_id
JOIN categories AS cat
  ON cat.category_id = p.category_id
WHERE
  NOT o.status IN ('cancelled', 'returned')
GROUP BY
  p.product_id,
  p.name,
  cat.name
ORDER BY
  revenue DESC
LIMIT 10
```

**10 rows** · 19 ms · recommended chart: **bar** · self-corrected after 1 failed attempt(s)

| product_name                  | category    |   units_sold |    revenue |
|:------------------------------|:------------|-------------:|-----------:|
| Smart Smartwatch              | Electronics |          537 | 130,602.37 |
| Ultra 4K Monitor              | Electronics |          477 | 122,856.10 |
| Essential Webcam              | Electronics |          482 | 117,102.27 |
| Smart Tablet                  | Electronics |          491 | 113,509.43 |
| Essential Mechanical Keyboard | Electronics |          447 |  89,833.60 |

*(first 5 rows)*

<details><summary>Self-correction: first attempt failed</summary>

PostgreSQL error fed back to the LLM: `column p.product_name does not exist
LINE 2:   p.product_name AS product_name,
          ^`

```sql
SELECT
  p.product_name AS product_name,
  cat.name AS category,
  SUM(oi.quantity) AS units_sold,
  ROUND(SUM(oi.quantity * oi.unit_price * (
    1 - oi.discount
  )), 2) AS revenue
FROM order_items AS oi
JOIN orders AS o
  ON o.order_id = oi.order_id
JOIN products AS p
  ON p.product_id = oi.product_id
JOIN categories AS cat
  ON cat.category_id = p.category_id
WHERE
  NOT o.status IN ('cancelled', 'returned')
GROUP BY
  p.product_id,
  p.name,
  cat.name
ORDER BY
  revenue DESC
LIMIT 10
```
</details>

### Compare monthly revenue for the last year.

```sql
SELECT
  CAST(DATE_TRUNC('MONTH', o.order_date) AS DATE) AS month,
  ROUND(SUM(oi.quantity * oi.unit_price * (
    1 - oi.discount
  )), 2) AS revenue
FROM orders AS o
JOIN order_items AS oi
  ON oi.order_id = o.order_id
WHERE
  NOT o.status IN ('cancelled', 'returned')
  AND o.order_date >= DATE_TRUNC('MONTH', CURRENT_DATE) - INTERVAL '12 MONTHS'
  AND o.order_date < DATE_TRUNC('MONTH', CURRENT_DATE)
GROUP BY
  1
ORDER BY
  1
LIMIT 1000
```

**12 rows** · 9 ms · recommended chart: **line**

| month      |   revenue |
|:-----------|----------:|
| 2025-11-01 | 67,268.20 |
| 2025-12-01 | 72,807.76 |
| 2026-01-01 | 70,144.27 |
| 2026-02-01 | 61,729.32 |
| 2026-03-01 | 68,270.53 |

*(first 5 rows)*

### Find customers who haven't placed an order in the last 90 days.

```sql
SELECT
  c.customer_id,
  c.first_name || ' ' || c.last_name AS customer_name,
  c.email,
  CAST(MAX(o.order_date) AS DATE) AS last_order_date
FROM customers AS c
LEFT JOIN orders AS o
  ON o.customer_id = c.customer_id
WHERE
  NOT EXISTS(
    SELECT
      1
    FROM orders AS o2
    WHERE
      o2.customer_id = c.customer_id
      AND o2.order_date >= CURRENT_DATE - INTERVAL '90 DAYS'
  )
GROUP BY
  c.customer_id,
  customer_name,
  c.email
ORDER BY
  last_order_date DESC NULLS LAST
LIMIT 1000
```

**104 rows** · 3 ms · recommended chart: **table**

|   customer_id | customer_name   | email                          | last_order_date   |
|--------------:|:----------------|:-------------------------------|:------------------|
|           563 | Lucas Davis     | lucas.davis396@example.com     | 2026-08-23        |
|            94 | Elizabeth Patel | elizabeth.patel440@example.com | 2026-08-23        |
|           367 | Priya Rossi     | priya.rossi42@example.com      | 2026-08-22        |
|           554 | Fatima Thomas   | fatima.thomas36@example.com    | 2026-08-21        |
|           140 | Lucas Anderson  | lucas.anderson348@example.com  | 2026-08-21        |

*(first 5 rows)*

### What is the revenue by product category?

```sql
SELECT
  cat.name AS category,
  ROUND(SUM(oi.quantity * oi.unit_price * (
    1 - oi.discount
  )), 2) AS revenue
FROM order_items AS oi
JOIN orders AS o
  ON o.order_id = oi.order_id
JOIN products AS p
  ON p.product_id = oi.product_id
JOIN categories AS cat
  ON cat.category_id = p.category_id
WHERE
  NOT o.status IN ('cancelled', 'returned')
GROUP BY
  cat.name
ORDER BY
  revenue DESC
LIMIT 1000
```

**8 rows** · 18 ms · recommended chart: **bar**

| category       |    revenue |
|:---------------|-----------:|
| Electronics    | 729,744.13 |
| Sports         | 352,088.92 |
| Home & Kitchen | 252,196.08 |
| Clothing       | 247,802.64 |
| Toys           | 174,960.88 |

*(first 5 rows)*

### What share of orders comes from each sales channel?

```sql
SELECT
  o.sales_channel,
  COUNT(*) AS orders,
  ROUND(CAST(100.0 * COUNT(*) / SUM(COUNT(*)) OVER () AS DECIMAL), 1) AS pct_of_orders
FROM orders AS o
WHERE
  NOT o.status IN ('cancelled', 'returned')
GROUP BY
  o.sales_channel
ORDER BY
  orders DESC
LIMIT 1000
```

**3 rows** · 3 ms · recommended chart: **pie**

| sales_channel   |   orders |   pct_of_orders |
|:----------------|---------:|----------------:|
| web             |     4441 |           55.50 |
| mobile          |     2398 |           30.00 |
| store           |     1159 |           14.50 |

### What is the gross margin by category?

```sql
SELECT
  cat.name AS category,
  ROUND(SUM(oi.quantity * oi.unit_price * (
    1 - oi.discount
  )), 2) AS revenue,
  ROUND(
    SUM(oi.quantity * (
      oi.unit_price * (
        1 - oi.discount
      ) - p.unit_cost
    )),
    2
  ) AS gross_profit,
  ROUND(
    100 * SUM(oi.quantity * (
      oi.unit_price * (
        1 - oi.discount
      ) - p.unit_cost
    )) / NULLIF(SUM(oi.quantity * oi.unit_price * (
      1 - oi.discount
    )), 0),
    1
  ) AS gross_margin_pct
FROM order_items AS oi
JOIN orders AS o
  ON o.order_id = oi.order_id
JOIN products AS p
  ON p.product_id = oi.product_id
JOIN categories AS cat
  ON cat.category_id = p.category_id
WHERE
  NOT o.status IN ('cancelled', 'returned')
GROUP BY
  cat.name
ORDER BY
  gross_margin_pct DESC
LIMIT 1000
```

**8 rows** · 27 ms · recommended chart: **bar**

| category       |    revenue |   gross_profit |   gross_margin_pct |
|:---------------|-----------:|---------------:|-------------------:|
| Beauty         | 123,773.11 |      54,114.42 |              43.70 |
| Books          |  91,352.05 |      39,929.12 |              43.70 |
| Home & Kitchen | 252,196.08 |     103,432.19 |              41.00 |
| Toys           | 174,960.88 |      68,351.08 |              39.10 |
| Sports         | 352,088.92 |     137,099.40 |              38.90 |

*(first 5 rows)*

## 2. Follow-up conversation

Follow-ups are rewritten into standalone questions using the session history, and the previous turns (question + SQL) are included in the prompt.

**User:** Show total revenue by month.  
**Interpreted as:** Show total revenue by month.

```sql
SELECT
  CAST(DATE_TRUNC('MONTH', o.order_date) AS DATE) AS month,
  ROUND(SUM(oi.quantity * oi.unit_price * (
    1 - oi.discount
  )), 2) AS revenue
FROM orders AS o
JOIN order_items AS oi
  ON oi.order_id = o.order_id
WHERE
  NOT o.status IN ('cancelled', 'returned')
GROUP BY
  1
ORDER BY
  1
LIMIT 1000
```

35 rows · chart: **line** (x=`month`, series=['revenue'])

**User:** Now only for 2025.  
**Interpreted as:** Show total revenue by month for 2025.

```sql
SELECT
  CAST(DATE_TRUNC('MONTH', o.order_date) AS DATE) AS month,
  ROUND(SUM(oi.quantity * oi.unit_price * (
    1 - oi.discount
  )), 2) AS revenue
FROM orders AS o
JOIN order_items AS oi
  ON oi.order_id = o.order_id
WHERE
  NOT o.status IN ('cancelled', 'returned')
  AND o.order_date >= CAST('2025-01-01' AS DATE)
  AND o.order_date < CAST('2026-01-01' AS DATE)
GROUP BY
  1
ORDER BY
  1
LIMIT 1000
```

12 rows · chart: **line** (x=`month`, series=['revenue'])

**User:** Compare it with 2024.  
**Interpreted as:** Compare monthly revenue in 2025 with 2024.

```sql
SELECT
  CAST(EXTRACT(MONTH FROM o.order_date) AS INT) AS month_num,
  ROUND(
    SUM(
      CASE
        WHEN EXTRACT(YEAR FROM o.order_date) = 2024
        THEN oi.quantity * oi.unit_price * (
          1 - oi.discount
        )
      END
    ),
    2
  ) AS revenue_2024,
  ROUND(
    SUM(
      CASE
        WHEN EXTRACT(YEAR FROM o.order_date) = 2025
        THEN oi.quantity * oi.unit_price * (
          1 - oi.discount
        )
      END
    ),
    2
  ) AS revenue_2025
FROM orders AS o
JOIN order_items AS oi
  ON oi.order_id = o.order_id
WHERE
  NOT o.status IN ('cancelled', 'returned')
  AND o.order_date >= CAST('2024-01-01' AS DATE)
  AND o.order_date < CAST('2026-01-01' AS DATE)
GROUP BY
  1
ORDER BY
  1
LIMIT 1000
```

12 rows · chart: **line** (x=`month_num`, series=['revenue_2024', 'revenue_2025'])

|   month_num |   revenue_2024 |   revenue_2025 |
|------------:|---------------:|---------------:|
|        1.00 |      21,527.13 |      60,705.06 |
|        2.00 |      31,737.95 |      58,359.20 |
|        3.00 |      39,526.36 |      58,446.36 |
|        4.00 |      46,750.63 |      60,272.34 |
|        5.00 |      56,603.97 |      70,876.92 |
|        6.00 |      45,272.04 |      66,212.12 |
|        7.00 |      58,249.68 |      65,706.98 |
|        8.00 |      48,458.06 |      72,464.40 |
|        9.00 |      46,438.06 |      68,320.41 |
|       10.00 |      59,644.63 |      56,382.60 |
|       11.00 |      50,580.45 |      67,268.20 |
|       12.00 |      45,260.87 |      72,807.76 |

## 3. Safety layer in action

Verdicts from `SQLValidator` for queries a malicious prompt or a confused model might produce (`python -m app.cli validate "<sql>"`):

| SQL | Verdict |
|---|---|
| `DELETE FROM orders` | ⛔ Forbidden keyword 'DELETE': only read-only SELECT queries are allowed. |
| `SELECT * FROM orders; DROP TABLE orders` | ⛔ Forbidden keyword 'DROP': only read-only SELECT queries are allowed. |
| `WITH d AS (DELETE FROM orders RETURNING *) SELECT * FROM d` | ⛔ Forbidden keyword 'DELETE': only read-only SELECT queries are allowed. |
| `SELECT * INTO backup FROM orders` | ⛔ Forbidden keyword 'INTO': only read-only SELECT queries are allowed. |
| `SELECT * FROM orders FOR UPDATE` | ⛔ Forbidden keyword 'UPDATE': only read-only SELECT queries are allowed. |
| `SELECT pg_sleep(60)` | ⛔ Function 'pg_sleep' is not allowed. |
| `SELECT query_to_xml('DELETE FROM orders', true, false, '')` | ⛔ Function 'query_to_xml' is not allowed. |
| `SELECT * FROM internal.employee_salaries` | ⛔ Table 'internal.employee_salaries' does not exist or is not approved for querying. |
| `SELECT * FROM pg_catalog.pg_user` | ⛔ Access to system catalog 'pg_catalog.pg_user' is not allowed. |
| `SELECT * FROM orders WHERE status = 'DROP TABLE x'` | ✅ allowed (LIMIT 1000 added) |
| `SELECT * FROM orders` | ✅ allowed (LIMIT 1000 added) |

Even if a statement got past the validator, the database would still refuse it: the role is read-only, every transaction is opened `READ ONLY`, and statements run under `statement_timeout`.

End-to-end behaviour of the agent:

- **"Delete all cancelled orders"** → status `blocked`: Query blocked by the safety layer: Forbidden keyword 'DELETE': only read-only SELECT queries are allowed.
- **"What is the average employee salary?"** → status `cannot_answer`: there is no salary data in the approved tables.
