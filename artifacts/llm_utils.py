"""LLM prompt templates and helpers for SQL generation and answer wording."""

import json
import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Prompt templates for SQL generation, repair, and answer phrasing.
SQL_TEMPLATE = """
Translate the question into a single MySQL query using ONLY the given schema.
Requirements:
- Output ONE SQL statement only (no markdown, no commentary, no labels like SQL:).
- Must start with SELECT or WITH.
- Use only tables and columns present in the schema.
- Avoid numeric formatting (no FORMAT(), no currency symbols). CONCAT/DATE_FORMAT allowed only
  for time-bucket labels / x keys.
- Do not guess columns (e.g., no discount unless it exists in the schema).

Time bucketing:
- Any time-bucket expression used in SELECT must also appear in GROUP BY.
- If grouping by week/month/quarter, include year with the bucket (never WEEK/MONTH/QUARTER alone).

Join rules (when relevant tables exist):
- Use these keys exactly: order_items.order_id = orders.order_id.
- Use these keys exactly: returns.order_id = orders.order_id.
- Do not use orders.id unless it exists in the schema.

Revenue rules (when revenue is requested):
- Use order_items price * quantity for gross revenue.
- Use returns for refunds when needed; net = gross - refunds.
- Avoid double counting refunds by aggregating returns per order_id first.
- If the user asks for "revenue" without qualifiers, return gross revenue only
  and name the column gross_revenue.

Schema:
{schema_details}

Question: {query}
"""

CHART_SQL_TEMPLATE = """
Translate the question into a single MySQL query suitable for charting.
Requirements:
- Output ONE SQL statement only (no markdown, no commentary, no labels like SQL:).
- Must start with SELECT or WITH.
- Output exactly one x column and 1-3 numeric y columns. No extra columns.
- Alias x as x and numeric columns clearly.
- ORDER BY x ASC unless sort explicitly requests DESC.
- Avoid numeric formatting (no FORMAT(), no currency symbols). CONCAT/DATE_FORMAT allowed only
  for time-bucket labels / x keys.
- Use only tables and columns present in the schema.
- Do not guess columns (e.g., no discount unless it exists in the schema).

Time bucketing:
- Any time-bucket expression used in SELECT must also appear in GROUP BY.
- Construct x by grain:
  day: DATE(date_col)
  week: YEARWEEK(date_col, 1)
  month: DATE_FORMAT(date_col, '%Y-%m-01')
  quarter: CONCAT(YEAR(date_col), '-Q', QUARTER(date_col))
  year: YEAR(date_col)
- If grouping by week/month/quarter, include year with the bucket.

Join rules (when relevant tables exist):
- Use these keys exactly: order_items.order_id = orders.order_id.
- Use these keys exactly: returns.order_id = orders.order_id.
- Do not use orders.id unless it exists in the schema.

Revenue rules (when revenue is requested):
- Use order_items price * quantity for gross revenue.
- Use returns for refunds when needed; net = gross - refunds.
- Avoid double counting refunds by aggregating returns per order_id first.
- If the user asks for "revenue" without qualifiers, return gross revenue only
  and name the column gross_revenue.

Schema:
{schema_details}

Chart intent: chart_type={chart_type}, grain={grain}, x={x}, y={y}, series={series}, sort={sort}
Question: {query}
"""

FIX_TEMPLATE = """
The following SQL failed to execute. Fix it for MySQL.
Schema: {schema_details}
User question: {query}
SQL:
```sql
{sql}
```
Error: {error}
Return ONLY the corrected SQL query. Output must start with SELECT or WITH.
Use only tables and columns present in the schema.
Avoid numeric formatting (no FORMAT(), no currency symbols). CONCAT/DATE_FORMAT allowed only
for time-bucket labels / x keys.
Join rules (when relevant tables exist):
- Use these keys exactly: order_items.order_id = orders.order_id.
- Use these keys exactly: returns.order_id = orders.order_id.
- Do not use orders.id unless it exists in the schema.
If subtracting refunds, aggregate returns per order_id first to avoid double counting.
If grouping by week/month/quarter, include year with the bucket.
"""

FIX_CHART_TEMPLATE = """
The following SQL failed to execute. Fix it for MySQL.
Schema: {schema_details}
User question: {query}
SQL:
```sql
{sql}
```
Error: {error}
Return ONLY the corrected SQL query. Output must start with SELECT or WITH.
Chart requirements:
- Exactly one x column (aliased as x) and 1-3 numeric y columns.
- Group by the x bucket and ORDER BY x ASC (unless sort says DESC).
Use only tables and columns present in the schema.
Avoid numeric formatting (no FORMAT(), no currency symbols). CONCAT/DATE_FORMAT allowed only
for time-bucket labels / x keys.
No extra columns besides x and the 1-3 numeric y columns.
Join rules (when relevant tables exist):
- Use these keys exactly: order_items.order_id = orders.order_id.
- Use these keys exactly: returns.order_id = orders.order_id.
- Do not use orders.id unless it exists in the schema.
If subtracting refunds, aggregate returns per order_id first to avoid double counting.

Chart intent: chart_type={chart_type}, grain={grain}, x={x}, y={y}, series={series}, sort={sort}
"""

ANSWER_TEMPLATE = """
You are a data assistant. answer="Here are results:" or if results are empty, say no data was found.
Do not mention SQL or the database.

Question: {query}
Results: {results}
Answer:
"""

CODE_BLOCK_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def is_chart_intent(intent: Optional[Dict[str, Any]]) -> bool:
    """Return True when chart intent is explicitly requested."""
    return bool(intent and intent.get("requested"))


def clean_text(text_value: str) -> str:
    """Remove <think>...</think> tags some models emit."""
    cleaned_text = re.sub(r"<think>.*?</think>", "", text_value, flags=re.DOTALL)
    return cleaned_text.strip()


def strip_code_fences(text_value: str) -> str:
    """Drop markdown code fences from model output."""
    # Only strip fences if the output looks like a code block.
    if text_value.strip().startswith("```"):
        text_value = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", text_value.strip(), flags=re.DOTALL)
    return text_value.strip()


def clean_sql(text_value: str) -> str:
    """Extract SQL from a code block and normalize it for execution."""
    cleaned = clean_text(text_value)
    match = CODE_BLOCK_RE.search(cleaned)
    # Prefer the content inside a ```sql``` block if present.
    if match:
        cleaned = match.group(1)
    else:
        cleaned = strip_code_fences(cleaned)
    cleaned = re.sub(r"^\s*sql\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip().rstrip(";")


def make_json_safe(value: Any) -> Any:
    """Convert common DB types to JSON-safe representations."""
    # Convert datetime/date objects to ISO strings.
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    # Convert Decimal to float so JSON can serialize it.
    if isinstance(value, Decimal):
        return float(value)
    # Decode bytes to a string with replacement for invalid bytes.
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    # Recurse into dictionaries.
    if isinstance(value, dict):
        return {key: make_json_safe(val) for key, val in value.items()}
    # Recurse into lists/tuples.
    if isinstance(value, (list, tuple)):
        return [make_json_safe(item) for item in value]
    return value


def to_sql_query(
    query: str,
    schema_details: str,
    model: OllamaLLM,
    intent: Optional[Dict[str, Any]] = None,
) -> str:
    """Generate SQL using the base or chart prompt."""
    template = CHART_SQL_TEMPLATE if is_chart_intent(intent) else SQL_TEMPLATE
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    payload = {
        "query": query,
        "schema_details": schema_details,
        "chart_type": (intent or {}).get("chart_type", "auto"),
        "grain": (intent or {}).get("grain", "auto"),
        "x": (intent or {}).get("x", "auto"),
        "y": (intent or {}).get("y", "auto"),
        "series": (intent or {}).get("series", "auto"),
        "sort": (intent or {}).get("sort", "auto"),
    }
    return clean_sql(chain.invoke(payload))


def fix_sql_query(
    query: str,
    schema_details: str,
    sql: str,
    error: str,
    model: OllamaLLM,
    intent: Optional[Dict[str, Any]] = None,
) -> str:
    """Repair SQL by feeding the DB error back to the model."""
    template = FIX_CHART_TEMPLATE if is_chart_intent(intent) else FIX_TEMPLATE
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    payload = {
        "query": query,
        "schema_details": schema_details,
        "sql": sql,
        "error": error,
        "chart_type": (intent or {}).get("chart_type", "auto"),
        "grain": (intent or {}).get("grain", "auto"),
        "x": (intent or {}).get("x", "auto"),
        "y": (intent or {}).get("y", "auto"),
        "series": (intent or {}).get("series", "auto"),
        "sort": (intent or {}).get("sort", "auto"),
    }
    return clean_sql(chain.invoke(payload))


def generate_answer(query: str, rows: List[Dict[str, Any]], model: OllamaLLM) -> str:
    """Return a minimal answer so the table/plot carry the detail."""
    if rows:
        return "Here are the results."
    return "No data was found."


def infer_range_answer(rows: List[Dict[str, Any]]) -> Optional[str]:
    """Return a plain-language range answer for common min/max patterns."""
    # Only handle single-row summaries with dict-shaped data.
    if len(rows) != 1 or not isinstance(rows[0], dict):
        return None
    row = rows[0]
    key_map = {key.lower(): key for key in row.keys()}
    pairs = [
        ("earliest_date", "latest_date"),
        ("min_date", "max_date"),
        ("start_date", "end_date"),
        ("min", "max"),
    ]
    # Scan common min/max key pairs.
    for start_key, end_key in pairs:
        # Use the first matching pair we find.
        if start_key in key_map and end_key in key_map:
            start_val = make_json_safe(row[key_map[start_key]])
            end_val = make_json_safe(row[key_map[end_key]])
            return (
                f"The range is from {start_val} to {end_val}. "
                "These are the earliest and latest dates in the data."
            )
    return None
