import re
from typing import Dict, List

# Safety guardrails for generated SQL.
ALLOWED_START = re.compile(r"^\s*(select|with)\b", re.IGNORECASE)
DISALLOWED = re.compile(
    r";|--|/\*|\b(insert|update|delete|drop|alter|create|truncate|attach|detach|pragma|grant|revoke)\b"
    r"|\binto\s+outfile\b|\binto\s+dumpfile\b|\bload_file\b|\bsleep\b|\bbenchmark\b",
    re.IGNORECASE,
)
TABLE_REF_RE = re.compile(r"\b(from|join)\s+([`\"\[]?[\w.]+[`\"\]]?)", re.IGNORECASE)
CTE_NAME_RE = re.compile(r"\bwith\s+([a-zA-Z_][\w]*)\s+as\s*\(|,\s*([a-zA-Z_][\w]*)\s+as\s*\(", re.IGNORECASE)
COMMENT_RE = re.compile(r"(--[^\n]*|/\*.*?\*/)", re.DOTALL)
DISALLOWED_SCHEMA_PREFIXES = {"information_schema", "mysql", "performance_schema", "sys", "pg_catalog"}


# Normalize identifiers for allowlist checks.
def normalize_identifier(name: str) -> str:
    """Trim quotes/brackets around identifiers."""
    return name.strip().strip('`"[]')


# Extract table references to validate against the schema.
def extract_table_refs(sql: str) -> List[str]:
    """Find table tokens after FROM/JOIN for allowlist checks."""
    return [match[1] for match in TABLE_REF_RE.findall(sql)]


def extract_cte_names(sql: str) -> List[str]:
    """Extract CTE names from WITH clauses."""
    names: List[str] = []
    for match in CTE_NAME_RE.findall(sql):
        name = match[0] or match[1]
        if name:
            names.append(name)
    return names


def strip_comments(sql: str) -> str:
    """Remove SQL comments for safer parsing."""
    return COMMENT_RE.sub(" ", sql)


# Block reads from system schemas.
def has_disallowed_prefix(ref: str) -> bool:
    """Reject references to system schemas that should be off-limits."""
    parts = [normalize_identifier(part).lower() for part in ref.split(".") if part]
    # Walk schema qualifiers and block system schemas.
    for prefix in parts[:-1]:
        if prefix in DISALLOWED_SCHEMA_PREFIXES:
            return True
    return False


# Match "schema.table" or "table" and keep just the table name.
def normalize_table_name(ref: str) -> str:
    """Strip schema qualifiers and return the table name."""
    parts = [normalize_identifier(part) for part in ref.split(".") if part]
    # No parts means we can't validate the table.
    if not parts:
        return ""
    return parts[-1]


# Accept only safe SELECT/CTE queries that target known tables.
def sql_safety_reason(sql: str, schema: Dict[str, List[str]]) -> str:
    """Return an error string if SQL is unsafe, otherwise an empty string."""
    stripped = strip_comments(sql)
    # Only allow SELECT or CTEs.
    if not ALLOWED_START.match(stripped):
        return "Only SELECT statements are allowed."
    # Block dangerous keywords and constructs.
    if DISALLOWED.search(stripped):
        return "The query includes a disallowed keyword or construct."
    refs = extract_table_refs(stripped)
    # Reject any reference to system schemas.
    for ref in refs:
        if has_disallowed_prefix(ref):
            return "The query references a system schema that is blocked."
    tables = {normalize_table_name(ref) for ref in refs if normalize_table_name(ref)}
    # If we can't detect table names, allow (conservative behavior).
    if not tables:
        return ""
    allowed = {name.lower() for name in schema.keys()}
    allowed.update(name.lower() for name in extract_cte_names(stripped))
    for table in tables:
        if table.lower() not in allowed:
            return f"Unknown table referenced: {table}"
    return ""


def is_safe_sql(sql: str, schema: Dict[str, List[str]]) -> bool:
    """Validate that SQL is read-only and targets only known tables."""
    return sql_safety_reason(sql, schema) == ""


# Default safety cap on result size.
def ensure_limit(sql: str, limit: int = 200, chart_mode: bool = False) -> str:
    """Add a LIMIT if the query doesn't already specify one."""
    if chart_mode:
        # Ensure deterministic ordering for time series output.
        if not re.search(r"\border\s+by\b", sql, flags=re.IGNORECASE):
            limit_match = re.search(r"\b(limit|fetch)\b", sql, flags=re.IGNORECASE)
            if limit_match:
                sql = f"{sql[:limit_match.start()]} ORDER BY 1 {sql[limit_match.start():]}"
            else:
                sql = f"{sql} ORDER BY 1"
        return sql
    # Respect existing LIMIT/FETCH clauses.
    if re.search(r"\blimit\b|\bfetch\b", sql, flags=re.IGNORECASE):
        return sql
    return f"{sql} LIMIT {limit}"
