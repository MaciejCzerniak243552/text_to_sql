from typing import Any, Dict, List

import streamlit as st
from sqlalchemy import create_engine, inspect, text


# Cache the database engine between reruns to avoid reconnect overhead.
@st.cache_resource(show_spinner=False)
def get_engine(db_url: str):
    """Create and cache a SQLAlchemy engine for the supplied database URL."""
    # Use pool_pre_ping to keep long-lived connections healthy across reruns.
    return create_engine(db_url, pool_pre_ping=True)


# Cache schema discovery to reduce DB roundtrips.
@st.cache_data(show_spinner=False)
def extract_schema(db_url: str) -> Dict[str, List[str]]:
    """Inspect the database and return a dict of table names to column names."""
    # Build the engine and inspector so we can introspect table metadata.
    engine = get_engine(db_url)
    inspector = inspect(engine)
    schema: Dict[str, List[str]] = {}

    # Capture table names and their columns for the prompt and safety checks.
    for table in inspector.get_table_names():
        # Fetch columns for each table and retain only their names.
        columns = inspector.get_columns(table)
        schema[table] = [col["name"] for col in columns]

    # Return the full schema map for downstream prompt and validation logic.
    return schema


# Execute SQL and return rows as dictionaries.
def run_query(db_url: str, sql: str) -> List[Dict[str, Any]]:
    """Execute SQL against the database and return rows as dictionaries."""
    # Reuse the cached engine for consistent connection pooling.
    engine = get_engine(db_url)
    # Open a connection, execute the SQL, and materialize rows as dicts.
    with engine.connect() as conn:
        result = conn.execute(text(sql))
        return [dict(row) for row in result.mappings().all()]
