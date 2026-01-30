"""Non-UI helpers for the chat workflow (intent parsing, retries, clarity checks)."""

from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

from langchain_ollama.llms import OllamaLLM

from artifacts.config_utils import get_setting
from artifacts.db_utils import run_query
from artifacts.llm_utils import fix_sql_query, to_sql_query
from artifacts.sql_safety import ensure_limit, sql_safety_reason


def parse_chart_intent(user_prompt: str, show_chart_toggle: bool) -> Dict[str, Any]:
    """Parse a deterministic chart intent structure from the prompt."""
    text = user_prompt.lower()
    requested = show_chart_toggle or bool(
        re.search(
            r"\b(plot|chart|graph|visuali[sz]e|trend|line|bar|histogram|scatter|pie|donut|draw|diagram|distribution|correlation)\b",
            text,
        )
    )

    chart_type = "auto"
    if re.search(r"\b(scatter|correlation)\b", text):
        chart_type = "scatter"
    elif re.search(r"\b(pie|donut)\b", text):
        chart_type = "pie"
    elif re.search(r"\b(hist|histogram|distribution)\b", text):
        chart_type = "hist"
    elif re.search(r"\b(bar|column)\b", text):
        chart_type = "bar"
    elif re.search(r"\b(line|trend|time series|timeseries)\b", text):
        chart_type = "line"

    grain = "auto"
    if re.search(r"\b(daily|by day|per day)\b", text):
        grain = "day"
    elif re.search(r"\b(weekly|by week|per week)\b", text):
        grain = "week"
    elif re.search(r"\b(monthly|by month|per month)\b", text):
        grain = "month"
    elif re.search(r"\b(quarterly|by quarter|per quarter)\b", text):
        grain = "quarter"
    elif re.search(r"\b(yearly|by year|per year|annual)\b", text):
        grain = "year"

    return {
        "requested": requested,
        "chart_type": chart_type,
        "grain": grain,
        "x": "auto",
        "y": "auto",
        "series": "auto",
        "sort": "auto",
    }


def get_db_display_name(db_url: str) -> str:
    """Return a friendly database name for display."""
    name = get_setting("DB_NAME")
    if name:
        return name
    if db_url.startswith("sqlite"):
        path = db_url.split("///", 1)[-1]
        return os.path.basename(path) or "sqlite"
    parsed = urlsplit(db_url)
    if parsed.path:
        return parsed.path.lstrip("/") or "database"
    return "database"


def needs_clarification(question: str, schema: Dict[str, List[str]]) -> Optional[str]:
    """Return a clarification prompt when the question is too vague."""
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", question.lower())
    if len(tokens) < 3:
        return "Could you clarify your question with more detail?"
    schema_terms = {name.lower() for name in schema.keys()}
    for cols in schema.values():
        schema_terms.update(col.lower() for col in cols)
    mentions_schema = any(tok in schema_terms for tok in tokens)
    generic = {"show", "list", "data", "info", "details", "report", "stats", "summary", "everything", "all", "overview"}
    has_generic = any(tok in generic for tok in tokens)
    has_filter = any(
        tok
        in {
            "count",
            "average",
            "avg",
            "min",
            "max",
            "sum",
            "total",
            "range",
            "between",
            "before",
            "after",
            "since",
            "during",
            "latest",
            "last",
            "top",
            "bottom",
            "most",
            "least",
        }
        for tok in tokens
    ) or bool(re.search(r"\d", question))
    if has_generic and not mentions_schema and not has_filter:
        table_names = sorted(schema.keys())[:3]
        if table_names:
            examples = ", ".join(table_names)
            return (
                "Could you clarify what you want to see? "
                f"For example, ask about {examples}, or specify a metric and time range."
            )
        return "Could you clarify what you want to see? Please specify a metric and any filters."
    return None


def get_last_result(messages: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Return the most recent assistant message that contains rows."""
    for message in reversed(messages):
        if message.get("rows") is not None:
            return message
    return None


def is_followup_plot_request(question: str) -> bool:
    """Detect short follow-up plot requests that refer to prior results."""
    text = question.lower()
    plot_words = r"\b(plot|chart|graph|visuali[sz]e|trend|line|bar|histogram|scatter|pie|draw|diagram)\b"
    refer_words = r"\b(this|that|these|those|above|previous|prior|same)\b|\blast\s+(result|results|query|one)\b"
    return bool(re.search(plot_words, text) and re.search(refer_words, text))


def query_with_retries(
    query: str,
    schema_details: str,
    schema: Dict[str, List[str]],
    db_url: str,
    model: OllamaLLM,
    max_retries: int,
    intent: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate SQL, validate it, execute it, and retry with fixes if needed."""
    sql = to_sql_query(query, schema_details, model, intent=intent)
    seen_sql = set()
    last_error = None

    for attempt in range(max_retries + 1):
        sql = ensure_limit(sql, limit=0, chart_mode=bool(intent and intent.get("requested")))
        safety_error = sql_safety_reason(sql, schema)
        if safety_error:
            return {"sql": sql, "rows": None, "error": safety_error}
        if sql in seen_sql:
            return {"sql": sql, "rows": None, "error": "Generated SQL repeated without fixing the error."}
        seen_sql.add(sql)
        try:
            rows = run_query(db_url, sql)
            return {"sql": sql, "rows": rows, "error": None}
        except Exception as exc:
            last_error = str(exc)
            if attempt >= max_retries:
                break
            sql = fix_sql_query(query, schema_details, sql, last_error, model, intent=intent)

    return {"sql": sql, "rows": None, "error": f"Query failed: {last_error}"}
