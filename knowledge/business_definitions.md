# Business definitions

Each `##` section below is stored as one document in the RAG knowledge base and
retrieved when it is relevant to a question. Edit this file to teach the agent
your own metrics, then click **Train / refresh knowledge base** in the sidebar
(or run `python -m app.cli train`).

## Revenue (net sales)
Revenue always means net product revenue:
`SUM(oi.quantity * oi.unit_price * (1 - oi.discount))` from `order_items oi`
joined to `orders o ON o.order_id = oi.order_id`, counting only orders whose
`o.status NOT IN ('cancelled', 'returned')`.
Shipping (`orders.shipping_cost`) is NOT revenue. "Sales" means revenue unless
the question asks for units or order counts. Round money to 2 decimals.

## Gross sales and discounts
Gross sales = `SUM(oi.quantity * oi.unit_price)` before discounts, for orders
that are not cancelled or returned. Discount amount =
`SUM(oi.quantity * oi.unit_price * oi.discount)`. `order_items.discount` is a
fraction (0.10 = 10%).

## Orders and average order value (AOV)
A valid order is an order with `status NOT IN ('cancelled', 'returned')`.
"Number of orders" counts valid orders: `COUNT(DISTINCT o.order_id)`.
Average order value (AOV) = revenue / number of valid orders.

## Units sold
Units sold = `SUM(oi.quantity)` over valid orders (status not cancelled or returned).
"Best-selling" or "highest sales" products are ranked by revenue unless the
question says units or quantity.

## Profit and gross margin
Gross profit per line = `oi.quantity * (oi.unit_price * (1 - oi.discount) - p.unit_cost)`
where `p` is `products` joined on `p.product_id = oi.product_id`.
Gross margin % = gross profit / revenue * 100. Valid orders only.

## Customers: active, inactive and new
- Customer name = `c.first_name || ' ' || c.last_name`.
- An active customer placed at least one order in the last 90 days:
  `o.order_date >= CURRENT_DATE - INTERVAL '90 days'`.
- An inactive (lapsed) customer has NOT placed any order in the last 90 days.
  Use `NOT EXISTS (SELECT 1 FROM orders o WHERE o.customer_id = c.customer_id
  AND o.order_date >= CURRENT_DATE - INTERVAL '90 days')`. Customers who never
  ordered are also inactive.
- New customers in a period are customers whose `signup_date` falls in that period.
- Top customers by revenue are ranked by their revenue (see Revenue definition).

## Time periods
`orders.order_date` is a TIMESTAMP. Always use half-open ranges (`>=` start AND `<` end).
- "last month" = the previous full calendar month:
  `o.order_date >= date_trunc('month', CURRENT_DATE) - INTERVAL '1 month' AND o.order_date < date_trunc('month', CURRENT_DATE)`.
- "this month" / "month to date" = `o.order_date >= date_trunc('month', CURRENT_DATE)`.
- "last year" (as a period) = the previous calendar year; "this year" = year to date.
- "the last 12 months" / "monthly ... for the last year" = the 12 most recent complete months:
  `o.order_date >= date_trunc('month', CURRENT_DATE) - INTERVAL '12 months' AND o.order_date < date_trunc('month', CURRENT_DATE)`.
- "last N days" = `o.order_date >= CURRENT_DATE - INTERVAL 'N days'`.
- Group by month with `date_trunc('month', o.order_date)::date AS month`, ordered by month.
- A specific year such as 2025: `o.order_date >= DATE '2025-01-01' AND o.order_date < DATE '2026-01-01'`.

## Year-over-year comparisons
To compare years month by month, return one row per month number with one
column per year, e.g.
`EXTRACT(MONTH FROM o.order_date)::int AS month_num`,
`SUM(CASE WHEN EXTRACT(YEAR FROM o.order_date) = 2024 THEN <revenue expr> END) AS revenue_2024`, and
`SUM(CASE WHEN EXTRACT(YEAR FROM o.order_date) = 2025 THEN <revenue expr> END) AS revenue_2025`.

## Order status and fulfilment
Order status values: 'pending', 'shipped', 'delivered', 'cancelled', 'returned'.
Return rate % = returned orders / (delivered + returned orders) * 100.
Cancellation rate % = cancelled orders / all orders * 100.

## Sales channels and segments
`orders.sales_channel` is one of 'web', 'mobile', 'store'.
`customers.segment` is one of 'Consumer', 'Small Business', 'Corporate'.
Store orders have zero shipping cost.

## Products and categories
Each product belongs to one category (`products.category_id -> categories.category_id`).
Category name is `categories.name`; product name is `products.name`.
Discontinued products have `products.is_active = FALSE`. Historical revenue
should still include discontinued products unless the question says "active products".
