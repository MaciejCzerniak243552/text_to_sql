import json
import re
from datetime import date, datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama.llms import OllamaLLM

# Prompt templates for SQL generation, repair, and answer phrasing.
SQL_TEMPLATE = """
You are a SQL expert. Given the following schema: {schema_details}, translate the user's
natural language question into a valid MySQL query.
Requirements:
- Return ONLY the SQL query (no code fences, no extra text).
- Do not format currency or numeric measures (avoid FORMAT or currency symbols).
- String labels for time buckets are allowed when used as the x axis.
- Use only tables and columns present in the schema (do not invent tables like date calendars).
Time bucketing rules:
- Any time-bucketing expression used in SELECT must also appear in GROUP BY.
- If grouping by quarter, always include year in SELECT and GROUP BY (never QUARTER alone).
Default time axis:
- Use orders.order_date for revenue time series unless the user explicitly asks for refund processing time.

Revenue rules:
- Use order_items to calculate revenue from item price and quantity.
- Use returns to account for refunds where relevant.
- Net revenue is gross minus refunds.
- Use the most appropriate date columns from the tables in the schema.
Defaults:
- If the user says "revenue" without qualifiers, treat it as gross revenue.
- If refunds are missing for a day, treat refunds as zero (use COALESCE).
Ranking rules:
- For questions like "which ... had the most/least" or "top/bottom", order by the metric and LIMIT 1
  (or LIMIT N if the user specifies N).
Join rules:
- order_items joins orders on order_items.order_id = orders.order_id.
- returns joins orders on returns.order_id = orders.order_id.
Refund aggregation rule:
- When subtracting refunds, aggregate returns per order_id first (e.g., a CTE refunds_by_order) to avoid double counting.
Default:
- If revenue is requested without a grain, default to daily using orders.order_date.

User question: {query}
"""

CHART_SQL_TEMPLATE = """
You are a SQL expert. Given the following schema: {schema_details}, translate the user's
natural language question into a valid MySQL query for charting.
Requirements:
- Return ONLY the SQL query (no code fences, no extra text).
- Output must be exactly one x column and 1-3 numeric y columns. No extra columns.
- Alias the x column as x, and numeric columns clearly (e.g., gross_revenue, refunds, net_revenue).
- For time series, group to one row per bucket based on grain (day/week/month/quarter/year).
- Include ORDER BY x ASC unless sort explicitly requests DESC.
- Do not format currency or numeric measures (avoid FORMAT or currency symbols).
- String labels for time buckets are allowed when used as the x axis.
- Use only tables and columns present in the schema (do not invent tables like date calendars).
Time bucketing rules:
- Any time-bucketing expression used in SELECT must also appear in GROUP BY.
- If grouping by quarter, always include year in SELECT and GROUP BY (never QUARTER alone).
Construct x deterministically by grain:
- day: DATE(date_col)
- week: YEARWEEK(date_col, 1)
- month: DATE_FORMAT(date_col, '%Y-%m-01')
- quarter: CONCAT(YEAR(date_col), '-Q', QUARTER(date_col))  (must include year)
- year: YEAR(date_col)
X uniqueness:
- x must uniquely identify the bucket (e.g., quarter must include year).
Default time axis:
- Use orders.order_date for revenue time series unless the user explicitly asks for refund processing time.

Revenue rules:
- Use order_items to calculate revenue from item price and quantity.
- Use returns to account for refunds where relevant.
- Net revenue is gross minus refunds.
- Use the most appropriate date columns from the tables in the schema.
Defaults:
- If the user says "revenue" without qualifiers, return only gross_revenue (plus the x column).
- Include gross_revenue or refunds only if explicitly requested.
- If refunds are missing for a day, treat refunds as zero (use COALESCE).
Ranking rules:
- For questions like "which ... had the most/least" or "top/bottom", order by the metric and LIMIT 1
  (or LIMIT N if the user specifies N).
Join rules:
- order_items joins orders on order_items.order_id = orders.order_id.
- returns joins orders on returns.order_id = orders.order_id.
Refund aggregation rule:
- When subtracting refunds, aggregate returns per order_id first (e.g., a CTE refunds_by_order) to avoid double counting.
Default:
- If revenue is requested without a grain, default to daily using orders.order_date.
- If asked for daily revenue in Feb 2025, filter by orders.order_date for Feb 2025
  and group by DATE(orders.order_date) AS x.
Example (quarterly gross revenue by sale date):
SELECT CONCAT(YEAR(o.order_date), '-Q', QUARTER(o.order_date)) AS x,
       SUM(oi.price*oi.quantity) AS gross_revenue
FROM orders o
JOIN order_items oi ON oi.order_id = o.order_id
WHERE o.order_date >= '2024-01-01' AND o.order_date < '2026-01-01'
GROUP BY YEAR(o.order_date), QUARTER(o.order_date)
ORDER BY YEAR(o.order_date), QUARTER(o.order_date);

Chart intent: chart_type={chart_type}, grain={grain}, x={x}, y={y}, series={series}, sort={sort}
User question: {query}
"""

FIX_TEMPLATE = """
You are a SQL expert. The following SQL failed to execute.
Schema: {schema_details}
User question: {query}
SQL:
```sql
{sql}
```
Error: {error}
Fix the query for MySQL. Return ONLY the corrected SQL query (no code fences, no extra text).
Keep numeric/date columns unformatted (no FORMAT/CONCAT).
Use only tables and columns present in the schema.
If the user asked for revenue without qualifiers, return gross_revenue only.
If the question is about "most/least" or "top/bottom", return only the top row (LIMIT 1).
"""

FIX_CHART_TEMPLATE = """
You are a SQL expert. The following SQL failed to execute.
Schema: {schema_details}
User question: {query}
SQL:
```sql
{sql}
```
Error: {error}
Fix the query for MySQL. Return ONLY the corrected SQL query (no code fences, no extra text).
Chart requirements:
- Exactly one x column (aliased as x) and 1-3 numeric y columns.
- Group by the x bucket and ORDER BY x ASC (unless sort says DESC).
- Keep numeric/date columns unformatted.
Use only tables and columns present in the schema.
If the user asked for revenue without qualifiers, return x and gross_revenue only.
If the question is about "most/least" or "top/bottom", return only the top row (LIMIT 1).

Chart intent: chart_type={chart_type}, grain={grain}, x={x}, y={y}, series={series}, sort={sort}
"""

ANSWER_TEMPLATE = """
You are a data assistant. Given the user question and SQL results (JSON array of
objects), answer in 1-2 sentences of plain language. The first sentence should be the
direct answer. The second sentence (if needed) should briefly explain how to interpret
the result. If results are empty, say that no data was found. If results are a single-row
summary, state the values directly (e.g., "The range is from X to Y. These are the earliest
and latest dates.").
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
    """Summarize results into a user-facing answer."""
    prompt = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)
    chain = prompt | model
    results_json = json.dumps(make_json_safe(rows), ensure_ascii=True)
    return clean_text(chain.invoke({"query": query, "results": results_json}))


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
