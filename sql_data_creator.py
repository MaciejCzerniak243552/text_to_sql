#!/usr/bin/env python3
"""
Seed a realistic dummy retail dataset into existing MySQL tables:

- Clears rows only (keeps table structure): returns -> order_items -> orders
- Creates:
    * 1000 orders
    * ~3000-4500 order_items (multiple items per order)
    * ~150-250 returns (minority of orders; some partial/multi returns)
- Dates span 2024-01-01 through 2025-12-31
- Ensures:
    * product_name <= 50 chars
    * product_name/category/price match a catalog (with small price jitter)
    * orders.total_amount equals sum(order_items.price * quantity) (rounded)
    * returns.refund_amount <= related order total (and partial refunds possible)
    * returns.processed_date >= order_date (clamped within 2024-2025)

Usage:
  export DB_URL="mysql+pymysql://USER:PASSWORD@HOST:3306/boas"
  python seed_realistic_2024_2025.py
"""

import os
import random
from datetime import date, timedelta
from typing import Dict, List, Tuple

from sqlalchemy import create_engine, text


# =========================
# CONFIG
# =========================
DB_URL = os.getenv("DB_URL")  # e.g. mysql+pymysql://boas:***@mysqltest...:3306/boas

N_ORDERS = 1000

# Basket size distribution (realistic: mostly 1–4 items, few larger)
ITEMS_PER_ORDER_WEIGHTS = {
    1: 0.25,
    2: 0.30,
    3: 0.22,
    4: 0.13,
    5: 0.06,
    6: 0.03,
    7: 0.01,
}

# Returns behavior
RETURN_ORDER_RATE = 0.18         # ~18% of orders have at least one return row
MULTI_RETURN_RATE = 0.15         # among returned orders, ~15% have 2 return rows
MAX_RETURN_LAG_DAYS = 45         # returns processed within 45 days after order date (clamped)

# Date span: 2024-2025
YEAR_START = date(2024, 1, 1)
YEAR_END = date(2025, 12, 31)

# Small price noise to simulate discounts/price changes while keeping product mapping
PRICE_JITTER_PCT = 0.08  # +/- 8% around base price

# Order status distribution (roughly realistic)
STATUS_WEIGHTS = [
    ("Delivered", 0.72),
    ("Shipped", 0.12),
    ("Processing", 0.08),
    ("Cancelled", 0.03),
    ("Returned", 0.05),
]


# =========================
# PRODUCT CATALOG
# product_name MUST be <= 50 chars
# =========================
CATALOG: List[Tuple[str, str, float]] = [
    ("Blue Shirt", "Apparel", 29.99),
    ("Red Hoodie", "Apparel", 59.99),
    ("Black Jeans", "Apparel", 74.99),
    ("Winter Jacket", "Apparel", 129.99),
    ("Sports Socks (Pack)", "Apparel", 12.99),
    ("White Sneakers", "Footwear", 89.99),
    ("Running Shoes", "Footwear", 109.99),
    ("Dress Shoes", "Footwear", 139.99),
    ("Leather Belt", "Accessories", 24.99),
    ("Baseball Cap", "Accessories", 19.99),
    ("Sunglasses", "Accessories", 49.99),
    ("Travel Backpack", "Accessories", 79.99),
]

# Validate name length constraint
for name, _, _ in CATALOG:
    if len(name) > 50:
        raise ValueError(f"Product name too long (>50): {name}")


RETURN_REASONS = [
    "Defective",
    "Changed mind",
    "Too small",
    "Too large",
    "Not as described",
    "Arrived late",
]


# =========================
# HELPERS
# =========================
def money(x: float) -> float:
    """Round to 2 decimals, avoid float noise."""
    return float(f"{x:.2f}")


def weighted_choice(weight_map: Dict[int, float]) -> int:
    items = list(weight_map.items())
    values = [k for k, _ in items]
    weights = [w for _, w in items]
    return random.choices(values, weights=weights, k=1)[0]


def weighted_status() -> str:
    values = [s for s, _ in STATUS_WEIGHTS]
    weights = [w for _, w in STATUS_WEIGHTS]
    return random.choices(values, weights=weights, k=1)[0]


def rand_date(start: date, end: date) -> date:
    span = (end - start).days
    return start + timedelta(days=random.randint(0, span))


def clamp(d: date, start: date, end: date) -> date:
    if d < start:
        return start
    if d > end:
        return end
    return d


def jitter_price(base: float) -> float:
    """Jitter base price by +/- PRICE_JITTER_PCT and round."""
    factor = 1.0 + random.uniform(-PRICE_JITTER_PCT, PRICE_JITTER_PCT)
    return money(base * factor)


# =========================
# DATA GENERATION
# =========================
def build_dataset(n_orders: int):
    """
    Returns:
      orders_rows: list[dict] for INSERT into orders (no order_id)
      items_rows: list[dict] for INSERT into order_items (no id)
      returns_rows: list[dict] for INSERT into returns (no return_id)
    """
    orders_rows: List[Dict] = []
    items_rows: List[Dict] = []
    returns_rows: List[Dict] = []

    # We'll reference order_id as 1..n_orders after resetting AUTO_INCREMENT.
    for i in range(n_orders):
        order_id = i + 1
        order_date = rand_date(YEAR_START, YEAR_END)

        # customer_id in your schema is decimal(14,0), so keep it numeric and large
        customer_id = random.randint(10000000000000, 99999999999999)

        # Basket size and line items
        basket_size = weighted_choice(ITEMS_PER_ORDER_WEIGHTS)

        # Pick distinct-ish products (allow repeats occasionally)
        chosen_products = [random.choice(CATALOG) for _ in range(basket_size)]

        order_total = 0.0
        for (product_name, category, base_price) in chosen_products:
            unit_price = jitter_price(base_price)
            qty = random.randint(1, 4)

            line_total = unit_price * qty
            order_total += line_total

            items_rows.append(
                {
                    "order_id": order_id,
                    "product_name": product_name,
                    "category": category,
                    "price": unit_price,
                    "quantity": qty,
                }
            )

        order_total = money(order_total)

        # Status: if cancelled, total still exists (demo), but no returns by default.
        status = weighted_status()

        orders_rows.append(
            {
                "customer_id": customer_id,
                "order_date": order_date.isoformat(),
                "total_amount": order_total,
                "status": status,
            }
        )

        # Returns: only for some orders, and generally not for Cancelled.
        if status != "Cancelled" and random.random() < RETURN_ORDER_RATE:
            n_return_rows = 2 if random.random() < MULTI_RETURN_RATE else 1

            # Split refund across 1 or 2 rows (partial refunds)
            # total refund is 20%..100% of order total
            total_refund = money(random.uniform(0.2 * order_total, order_total))

            # If two returns, split it into two parts
            if n_return_rows == 2:
                part1 = money(total_refund * random.uniform(0.3, 0.7))
                part2 = money(total_refund - part1)
                refund_parts = [part1, part2]
            else:
                refund_parts = [total_refund]

            for refund_amount in refund_parts:
                lag = random.randint(0, MAX_RETURN_LAG_DAYS)
                processed_date = clamp(order_date + timedelta(days=lag), YEAR_START, YEAR_END)

                returns_rows.append(
                    {
                        "order_id": order_id,
                        "reason": random.choice(RETURN_REASONS),
                        "refund_amount": min(refund_amount, order_total),
                        "processed_date": processed_date.isoformat(),
                    }
                )

    return orders_rows, items_rows, returns_rows


# =========================
# DB OPS
# =========================
def clear_tables_keep_structure(conn) -> None:
    """
    Clear rows only (keep schema). Child tables first.
    Reset AUTO_INCREMENT so order_id becomes 1..N for deterministic linking.
    """
    conn.execute(text("DELETE FROM returns;"))
    conn.execute(text("DELETE FROM order_items;"))
    conn.execute(text("DELETE FROM orders;"))

    # Reset autoincrement counters (MySQL). If privileges block this, comment out.
    conn.execute(text("ALTER TABLE orders AUTO_INCREMENT = 1;"))
    conn.execute(text("ALTER TABLE order_items AUTO_INCREMENT = 1;"))
    conn.execute(text("ALTER TABLE returns AUTO_INCREMENT = 1;"))


def main() -> None:
    if not DB_URL:
        raise SystemExit(
            "DB_URL is not set.\n"
            'Example: export DB_URL="mysql+pymysql://user:pass@host:3306/boas"\n'
        )

    engine = create_engine(DB_URL, pool_pre_ping=True)

    orders_rows, items_rows, returns_rows = build_dataset(N_ORDERS)

    with engine.begin() as conn:
        clear_tables_keep_structure(conn)

        # Insert orders (order_id auto_increment => do NOT insert order_id)
        conn.execute(
            text(
                """
                INSERT INTO orders (customer_id, order_date, total_amount, status)
                VALUES (:customer_id, :order_date, :total_amount, :status)
                """
            ),
            orders_rows,
        )

        # Insert order_items (id auto_increment => do NOT insert id)
        conn.execute(
            text(
                """
                INSERT INTO order_items (order_id, product_name, category, price, quantity)
                VALUES (:order_id, :product_name, :category, :price, :quantity)
                """
            ),
            items_rows,
        )

        # Insert returns (return_id auto_increment => do NOT insert return_id)
        if returns_rows:
            conn.execute(
                text(
                    """
                    INSERT INTO returns (order_id, reason, refund_amount, processed_date)
                    VALUES (:order_id, :reason, :refund_amount, :processed_date)
                    """
                ),
                returns_rows,
            )

    print("✅ Done seeding dataset.")
    print(f"✅ Orders:      {len(orders_rows)}")
    print(f"✅ Order items: {len(items_rows)}")
    print(f"✅ Returns:     {len(returns_rows)}")
    print("✅ Date range:  2024-01-01 .. 2025-12-31")


if __name__ == "__main__":
    main()