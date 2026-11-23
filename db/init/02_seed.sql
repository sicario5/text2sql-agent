-- =====================================================================
-- Deterministic sample data.
-- Orders run from 2024-01-01 up to the day the database is initialised,
-- so relative questions ("last month", "last 90 days") always have data.
-- Re-initialise the database (make reset-db) to move the window forward.
-- =====================================================================

SELECT setseed(0.42);

-- ---------------------------------------------------------------- categories
INSERT INTO categories (name, description) VALUES
    ('Electronics',     'Consumer electronics and accessories'),
    ('Home & Kitchen',  'Cookware, appliances and home goods'),
    ('Books',           'Printed books and e-readers'),
    ('Clothing',        'Apparel for men, women and kids'),
    ('Sports',          'Sports and outdoor equipment'),
    ('Beauty',          'Skincare, haircare and cosmetics'),
    ('Toys',            'Toys and games'),
    ('Office Supplies', 'Stationery and office equipment');

-- ---------------------------------------------------------------- products
WITH catalog (category, base_price, nouns) AS (
    VALUES
    ('Electronics',     120.0, ARRAY['Headphones','Smartwatch','Bluetooth Speaker','Tablet','Webcam','Power Bank','Mechanical Keyboard','4K Monitor']),
    ('Home & Kitchen',   60.0, ARRAY['Air Fryer','Chef Knife','Coffee Maker','Blender','Cookware Set','Toaster','Vacuum Cleaner','Bed Sheets']),
    ('Books',            18.0, ARRAY['Cookbook','Mystery Novel','Sci-Fi Novel','Biography','Travel Guide','Kids Picture Book','Business Book','Poetry Collection']),
    ('Clothing',         40.0, ARRAY['T-Shirt','Jeans','Hoodie','Running Shoes','Rain Jacket','Sweater','Sneakers','Dress']),
    ('Sports',           55.0, ARRAY['Yoga Mat','Dumbbell Set','Tennis Racket','Cycling Helmet','Camping Tent','Football','Water Bottle','Hiking Backpack']),
    ('Beauty',           25.0, ARRAY['Face Serum','Shampoo','Moisturizer','Lipstick','Sunscreen','Hair Dryer','Perfume','Nail Kit']),
    ('Toys',             30.0, ARRAY['Building Blocks','Puzzle','Board Game','Plush Bear','RC Car','Doll House','Art Kit','Science Kit']),
    ('Office Supplies',  22.0, ARRAY['Notebook','Desk Lamp','Ergonomic Chair','Gel Pens','Stapler','Planner','Desk Organizer','Whiteboard'])
),
expanded AS (
    SELECT c.category_id,
           cat.category,
           g AS n,
           cat.nouns[g] AS noun,
           cat.base_price,
           (ARRAY['Classic','Pro','Eco','Ultra','Essential','Premium','Compact','Smart'])[1 + floor(random() * 8)::int] AS adjective,
           0.5 + random() * 1.8 AS price_factor,
           0.45 + random() * 0.25 AS cost_ratio
    FROM catalog cat
    JOIN categories c ON c.name = cat.category
    CROSS JOIN generate_series(1, 8) AS g
)
INSERT INTO products (sku, name, category_id, unit_price, unit_cost, is_active, created_at)
SELECT upper(left(regexp_replace(category, '[^A-Za-z]', '', 'g'), 3)) || '-' || lpad((category_id * 100 + n)::text, 5, '0'),
       adjective || ' ' || noun,
       category_id,
       round((base_price * price_factor)::numeric, 2),
       round((base_price * price_factor * cost_ratio)::numeric, 2),
       random() > 0.08,
       DATE '2023-01-01' + (random() * 300)::int
FROM expanded
ORDER BY category_id, n;

-- ---------------------------------------------------------------- customers
WITH names AS (
    SELECT ARRAY['James','Mary','Robert','Patricia','John','Jennifer','Michael','Linda','David','Elizabeth',
                 'William','Barbara','Richard','Susan','Joseph','Jessica','Thomas','Sarah','Priya','Arjun',
                 'Wei','Mei','Carlos','Sofia','Ahmed','Fatima','Kenji','Yuki','Olga','Lucas'] AS first_names,
           ARRAY['Smith','Johnson','Williams','Brown','Jones','Garcia','Miller','Davis','Rodriguez','Martinez',
                 'Hernandez','Lopez','Gonzalez','Wilson','Anderson','Thomas','Taylor','Moore','Patel','Sharma',
                 'Chen','Wang','Kim','Nguyen','Silva','Khan','Tanaka','Ivanova','Rossi','Muller'] AS last_names,
           ARRAY['New York|NY','Los Angeles|CA','Chicago|IL','Houston|TX','Phoenix|AZ','Philadelphia|PA',
                 'San Antonio|TX','San Diego|CA','Dallas|TX','Austin|TX','Seattle|WA','Denver|CO',
                 'Boston|MA','Miami|FL','Atlanta|GA','Portland|OR'] AS cities
),
raw AS (
    SELECT g AS n,
           first_names[1 + floor(random() * 30)::int] AS first_name,
           last_names[1 + floor(random() * 30)::int]  AS last_name,
           cities[1 + floor(random() * 16)::int]      AS city_state,
           random() AS seg_r,
           DATE '2023-01-01' + floor(power(random(), 1.6) * (CURRENT_DATE - DATE '2023-01-01' - 7))::int AS signup_date
    FROM names, generate_series(1, 600) AS g
)
INSERT INTO customers (first_name, last_name, email, city, state, country, segment, signup_date)
SELECT first_name,
       last_name,
       lower(first_name || '.' || last_name || n || '@example.com'),
       split_part(city_state, '|', 1),
       split_part(city_state, '|', 2),
       'USA',
       CASE WHEN seg_r < 0.62 THEN 'Consumer' WHEN seg_r < 0.85 THEN 'Small Business' ELSE 'Corporate' END,
       signup_date
FROM raw
ORDER BY signup_date, n;  -- customer_id increases with signup_date (used below)

-- ---------------------------------------------------------------- orders
-- Build orders in a temp table first so every random() is evaluated per row.
CREATE TEMP TABLE tmp_orders AS
SELECT g AS n,
       random() AS customer_r,
       -- power(random(), 0.8) skews dates towards the present => visible growth trend
       TIMESTAMP '2024-01-01 08:00'
         + power(random(), 0.8) * ((CURRENT_DATE - DATE '2024-01-01') * INTERVAL '1 day')
         + floor(random() * 14) * INTERVAL '1 hour' AS order_date,
       random() AS status_r,
       random() AS channel_r
FROM generate_series(1, 9000) AS g;

-- Pick a customer who had already signed up when the order was placed
-- (customer ids 1..k are exactly the customers who signed up by that date).
INSERT INTO orders (customer_id, order_date, status, sales_channel, shipping_cost)
SELECT GREATEST(1, ceil(customer_r * (SELECT count(*) FROM customers c WHERE c.signup_date <= t.order_date::date)))::int,
       LEAST(order_date, now()::timestamp - INTERVAL '1 hour'),
       CASE
           WHEN order_date > CURRENT_DATE - 3 THEN CASE WHEN status_r < 0.6 THEN 'pending' ELSE 'shipped' END
           WHEN order_date > CURRENT_DATE - 10 THEN CASE WHEN status_r < 0.15 THEN 'pending' WHEN status_r < 0.75 THEN 'shipped' ELSE 'delivered' END
           WHEN status_r < 0.06 THEN 'cancelled'
           WHEN status_r < 0.11 THEN 'returned'
           ELSE 'delivered'
       END,
       CASE WHEN channel_r < 0.55 THEN 'web' WHEN channel_r < 0.85 THEN 'mobile' ELSE 'store' END,
       CASE WHEN channel_r >= 0.85 THEN 0 ELSE round((4.99 + random() * 10)::numeric, 2) END
FROM tmp_orders t
ORDER BY order_date;

-- ---------------------------------------------------------------- order items
CREATE TEMP TABLE tmp_order_sizes AS
SELECT order_id, 1 + floor(random() * 4)::int AS n_items FROM orders;

CREATE TEMP TABLE tmp_items AS
SELECT o.order_id,
       1 + floor(random() * (SELECT count(*) FROM products))::int AS product_id,
       CASE WHEN random() < 0.7 THEN 1 ELSE 2 + floor(random() * 3)::int END AS quantity,
       random() AS discount_r
FROM tmp_order_sizes o
CROSS JOIN LATERAL generate_series(1, o.n_items) AS s;

INSERT INTO order_items (order_id, product_id, quantity, unit_price, discount)
SELECT i.order_id,
       i.product_id,
       i.quantity,
       p.unit_price,
       CASE WHEN i.discount_r < 0.70 THEN 0
            WHEN i.discount_r < 0.82 THEN 0.05
            WHEN i.discount_r < 0.92 THEN 0.10
            WHEN i.discount_r < 0.97 THEN 0.15
            ELSE 0.20 END
FROM tmp_items i
JOIN products p ON p.product_id = i.product_id;

-- Guard: nobody orders before they sign up.
UPDATE customers c
SET signup_date = LEAST(c.signup_date, f.first_order::date)
FROM (SELECT customer_id, min(order_date) AS first_order FROM orders GROUP BY customer_id) f
WHERE f.customer_id = c.customer_id;

DROP TABLE tmp_items;
DROP TABLE tmp_order_sizes;
DROP TABLE tmp_orders;

-- ---------------------------------------------------------------- sensitive data
INSERT INTO internal.employee_salaries (full_name, salary_usd, ssn_last4) VALUES
    ('Alice Admin',   185000, '1234'),
    ('Bob Builder',    92000, '5678'),
    ('Carol Finance', 143000, '9012');

ANALYZE;
