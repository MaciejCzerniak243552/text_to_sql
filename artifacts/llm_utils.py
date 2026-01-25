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
natural language question into a valid MySQL query. Return ONLY the SQL code
inside a code block.

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
Fix the query for MySQL. Return ONLY the corrected SQL code inside a code block.
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

# Regex for extracting fenced SQL blocks from model output.
CODE_BLOCK_RE = re.compile(r"```(?:sql)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def clean_text(text_value: str) -> str:
    """Strip hidden model reasoning tags and trim whitespace."""
    # Remove any <think>...</think> tags that should not reach the user.
    cleaned_text = re.sub(r"<think>.*?</think>", "", text_value, flags=re.DOTALL)
    # Normalize leading/trailing whitespace after tag removal.
    return cleaned_text.strip()


def strip_code_fences(text_value: str) -> str:
    """Drop markdown code fences from model output."""
    # Only strip fences if the output looks like a code block.
    if text_value.strip().startswith("```"):
        # Remove opening/closing fences while keeping inner SQL content.
        text_value = re.sub(r"^```[a-zA-Z]*\n|\n```$", "", text_value.strip(), flags=re.DOTALL)
    # Return the cleaned text with surrounding whitespace removed.
    return text_value.strip()


def clean_sql(text_value: str) -> str:
    """Extract SQL from model output and normalize it for execution."""
    # First, remove any hidden tags that are not part of the SQL content.
    cleaned = clean_text(text_value)
    # Look for a fenced SQL code block and prefer its inner content.
    match = CODE_BLOCK_RE.search(cleaned)
    if match:
        cleaned = match.group(1)
    else:
        # Fall back to stripping generic code fences when no SQL block is found.
        cleaned = strip_code_fences(cleaned)
    # Remove a leading "sql:" label some models prepend.
    cleaned = re.sub(r"^\s*sql\s*:\s*", "", cleaned, flags=re.IGNORECASE)
    # Trim whitespace and any trailing semicolon for execution.
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
    # Return primitive types unchanged.
    return value


def to_sql_query(query: str, schema_details: str, model: OllamaLLM) -> str:
    """Generate SQL from a user query and schema details using the base prompt."""
    # Build the prompt and chain it to the LLM.
    prompt = ChatPromptTemplate.from_template(SQL_TEMPLATE)
    chain = prompt | model
    # Invoke the chain and normalize the output to raw SQL.
    return clean_sql(chain.invoke({"query": query, "schema_details": schema_details}))


def fix_sql_query(query: str, schema_details: str, sql: str, error: str, model: OllamaLLM) -> str:
    """Repair SQL by feeding the DB error back to the model."""
    # Build the prompt that includes the failing SQL and DB error details.
    prompt = ChatPromptTemplate.from_template(FIX_TEMPLATE)
    chain = prompt | model
    # Invoke the chain and normalize the corrected SQL.
    return clean_sql(
        chain.invoke({"query": query, "schema_details": schema_details, "sql": sql, "error": error})
    )


def generate_answer(query: str, rows: List[Dict[str, Any]], model: OllamaLLM) -> str:
    """Summarize SQL results into a user-facing answer."""
    # Build the answer prompt for the LLM.
    prompt = ChatPromptTemplate.from_template(ANSWER_TEMPLATE)
    chain = prompt | model
    # Serialize rows into JSON so the model sees concrete values.
    results_json = json.dumps(make_json_safe(rows), ensure_ascii=True)
    # Invoke the model and strip hidden tags from its response.
    return clean_text(chain.invoke({"query": query, "results": results_json}))


def infer_range_answer(rows: List[Dict[str, Any]]) -> Optional[str]:
    """Return a plain-language range answer for common min/max patterns."""
    # Only handle single-row summaries with dict-shaped data.
    if len(rows) != 1 or not isinstance(rows[0], dict):
        return None
    # Build a lowercase key map so we can match case-insensitive column names.
    row = rows[0]
    key_map = {key.lower(): key for key in row.keys()}
    # Define common pairs that represent range outputs.
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
            # Convert values to JSON-safe formats before composing the sentence.
            start_val = make_json_safe(row[key_map[start_key]])
            end_val = make_json_safe(row[key_map[end_key]])
            return (
                f"The range is from {start_val} to {end_val}. "
                "These are the earliest and latest dates in the data."
            )
    # Return None when no recognized range pattern is present.
    return None
