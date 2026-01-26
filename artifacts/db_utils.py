from typing import Any, Dict, List, Tuple

import streamlit as st
from sqlalchemy import create_engine, inspect, text


# Cache the database engine between reruns.
@st.cache_resource(show_spinner=False)
def get_engine(db_url: str):
    """Create and cache a SQLAlchemy engine."""
    return create_engine(db_url, pool_pre_ping=True)


# Cache schema discovery to reduce DB roundtrips.
@st.cache_data(show_spinner=False)
def extract_schema(db_url: str) -> Tuple[Dict[str, List[str]], str]:
    """Inspect tables/columns and return (schema map, enriched schema details)."""
    engine = get_engine(db_url)
    inspector = inspect(engine)
    schema: Dict[str, List[str]] = {}
    schema_lines: List[str] = []

    # Capture table names and their columns for the prompt and safety checks.
    for table in inspector.get_table_names():
        columns = inspector.get_columns(table)
        column_names: List[str] = []
        column_details: List[str] = []
        for col in columns:
            name = col["name"]
            column_names.append(name)
            col_type = str(col.get("type", "")).upper()
            pk = " PK" if col.get("primary_key") else ""
            column_details.append(f"{name} {col_type}{pk}".strip())
        schema[table] = column_names
        if column_details:
            schema_lines.append(f"{table}: {', '.join(column_details)}")

    relationships: List[str] = []
    if "order_items" in schema and "orders" in schema:
        relationships.append("order_items.order_id -> orders.order_id")
    if "returns" in schema and "orders" in schema:
        relationships.append("returns.order_id -> orders.order_id")

    if relationships:
        schema_lines.append("Relationships: " + "; ".join(relationships))

    date_hints: List[str] = []
    if "orders" in schema and "order_date" in schema["orders"]:
        date_hints.append("orders.order_date (sales time axis)")
    if "returns" in schema and "processed_date" in schema["returns"]:
        date_hints.append("returns.processed_date (refund time axis)")
    if date_hints:
        schema_lines.append("Date hints: " + "; ".join(date_hints))

    schema_details = "\n".join(schema_lines)
    return schema, schema_details


# Execute SQL and return rows as dictionaries.
def run_query(db_url: str, sql: str) -> List[Dict[str, Any]]:
    """Execute SQL against the DB and return row dicts."""
    engine = get_engine(db_url)
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        return [dict(row) for row in result.mappings().all()]
