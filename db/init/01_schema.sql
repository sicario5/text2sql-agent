-- =====================================================================
-- Sample e-commerce schema for the Text-to-SQL agent
-- Runs automatically on first start of the postgres container
-- (files in /docker-entrypoint-initdb.d are executed in name order).
-- =====================================================================

-- ---------------------------------------------------------------------
-- public schema: the analytics data the agent is allowed to query
-- ---------------------------------------------------------------------

CREATE TABLE categories (
    category_id  SERIAL PRIMARY KEY,
    name         TEXT NOT NULL UNIQUE,
    description  TEXT
);

CREATE TABLE products (
    product_id   SERIAL PRIMARY KEY,
    sku          TEXT NOT NULL UNIQUE,
    name         TEXT NOT NULL,
    category_id  INT  NOT NULL REFERENCES categories (category_id),
    unit_price   NUMERIC(10, 2) NOT NULL CHECK (unit_price >= 0),
    unit_cost    NUMERIC(10, 2) NOT NULL CHECK (unit_cost >= 0),
    is_active    BOOLEAN NOT NULL DEFAULT TRUE,
    created_at   DATE NOT NULL
);

CREATE TABLE customers (
    customer_id  SERIAL PRIMARY KEY,
    first_name   TEXT NOT NULL,
    last_name    TEXT NOT NULL,
    email        TEXT NOT NULL UNIQUE,
    city         TEXT,
    state        TEXT,
    country      TEXT NOT NULL DEFAULT 'USA',
    segment      TEXT NOT NULL CHECK (segment IN ('Consumer', 'Corporate', 'Small Business')),
    signup_date  DATE NOT NULL
);

CREATE TABLE orders (
    order_id       SERIAL PRIMARY KEY,
    customer_id    INT NOT NULL REFERENCES customers (customer_id),
    order_date     TIMESTAMP NOT NULL,
    status         TEXT NOT NULL CHECK (status IN ('pending', 'shipped', 'delivered', 'cancelled', 'returned')),
    sales_channel  TEXT NOT NULL CHECK (sales_channel IN ('web', 'mobile', 'store')),
    shipping_cost  NUMERIC(10, 2) NOT NULL DEFAULT 0
);

CREATE TABLE order_items (
    order_item_id  SERIAL PRIMARY KEY,
    order_id       INT NOT NULL REFERENCES orders (order_id) ON DELETE CASCADE,
    product_id     INT NOT NULL REFERENCES products (product_id),
    quantity       INT NOT NULL CHECK (quantity > 0),
    unit_price     NUMERIC(10, 2) NOT NULL CHECK (unit_price >= 0),
    discount       NUMERIC(4, 3) NOT NULL DEFAULT 0 CHECK (discount >= 0 AND discount < 1)
);

CREATE INDEX idx_products_category    ON products (category_id);
CREATE INDEX idx_orders_customer      ON orders (customer_id);
CREATE INDEX idx_orders_order_date    ON orders (order_date);
CREATE INDEX idx_orders_status        ON orders (status);
CREATE INDEX idx_order_items_order    ON order_items (order_id);
CREATE INDEX idx_order_items_product  ON order_items (product_id);

-- Comments are picked up by the schema inspector and fed into the RAG knowledge base.
COMMENT ON TABLE categories IS 'Product categories (e.g. Electronics, Books).';
COMMENT ON TABLE products IS 'Product catalog. One row per sellable product (SKU).';
COMMENT ON COLUMN products.unit_price IS 'Current list price in USD.';
COMMENT ON COLUMN products.unit_cost IS 'Current unit cost in USD, used for margin calculations.';
COMMENT ON COLUMN products.is_active IS 'FALSE when the product is discontinued.';

COMMENT ON TABLE customers IS 'Customers who have signed up. One row per customer.';
COMMENT ON COLUMN customers.segment IS 'Customer segment: Consumer, Corporate or Small Business.';
COMMENT ON COLUMN customers.signup_date IS 'Date the customer created their account.';

COMMENT ON TABLE orders IS 'Order headers. One row per order placed by a customer.';
COMMENT ON COLUMN orders.order_date IS 'Timestamp when the order was placed.';
COMMENT ON COLUMN orders.status IS 'pending, shipped, delivered, cancelled or returned. Revenue excludes cancelled and returned orders.';
COMMENT ON COLUMN orders.sales_channel IS 'Channel the order was placed through: web, mobile or store.';
COMMENT ON COLUMN orders.shipping_cost IS 'Shipping charged to the customer in USD. Not part of product revenue.';

COMMENT ON TABLE order_items IS 'Order line items. One row per product within an order.';
COMMENT ON COLUMN order_items.unit_price IS 'Price per unit in USD at the time of the order (before discount).';
COMMENT ON COLUMN order_items.discount IS 'Fractional discount applied to the line, e.g. 0.100 = 10% off.';

-- ---------------------------------------------------------------------
-- internal schema: sensitive data the agent must NEVER see.
-- The read-only role gets no privileges here (see 03_roles.sh), and the
-- SQL validator rejects any table that is not on the approved list.
-- ---------------------------------------------------------------------
CREATE SCHEMA internal;

CREATE TABLE internal.employee_salaries (
    employee_id  SERIAL PRIMARY KEY,
    full_name    TEXT NOT NULL,
    salary_usd   NUMERIC(12, 2) NOT NULL,
    ssn_last4    CHAR(4) NOT NULL
);
