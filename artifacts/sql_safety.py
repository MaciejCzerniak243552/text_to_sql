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
DISALLOWED_SCHEMA_PREFIXES = {"information_schema", "mysql", "performance_schema", "sys", "pg_catalog"}


# Normalize identifiers for allowlist checks.
def normalize_identifier(name: str) -> str:
    """Trim quotes/brackets around identifiers for consistent comparison."""
    # Remove whitespace and wrapping characters that vary between SQL dialects.
    return name.strip().strip('`"[]')


# Extract table references to validate against the schema.
def extract_table_refs(sql: str) -> List[str]:
    """Find table tokens after FROM/JOIN for allowlist checks."""
    # Return the raw table references so later checks can normalize them.
    return [match[1] for match in TABLE_REF_RE.findall(sql)]


# Block reads from system schemas.
def has_disallowed_prefix(ref: str) -> bool:
    """Reject references to system schemas that should be off-limits."""
    # Normalize each schema qualifier and lower-case for comparison.
    parts = [normalize_identifier(part).lower() for part in ref.split(".") if part]
    # Walk schema qualifiers and block system schemas.
    for prefix in parts[:-1]:
        if prefix in DISALLOWED_SCHEMA_PREFIXES:
            return True
    # No forbidden prefixes were found.
    return False


# Match "schema.table" or "table" and keep just the table name.
def normalize_table_name(ref: str) -> str:
    """Strip schema qualifiers and return the table name."""
    # Break the reference into its dot-separated parts.
    parts = [normalize_identifier(part) for part in ref.split(".") if part]
    # No parts means we cannot validate the table name.
    if not parts:
        return ""
    # Return only the final segment, which is the table name.
    return parts[-1]


# Accept only safe SELECT/CTE queries that target known tables.
def is_safe_sql(sql: str, schema: Dict[str, List[str]]) -> bool:
    """Validate that SQL is read-only and targets only known tables."""
    # Only allow SELECT statements or WITH (CTE) statements.
    if not ALLOWED_START.match(sql):
        return False
    # Block dangerous keywords and constructs.
    if DISALLOWED.search(sql):
        return False
    # Extract table references so we can validate against the schema.
    refs = extract_table_refs(sql)
    # Reject any reference to system schemas.
    for ref in refs:
        if has_disallowed_prefix(ref):
            return False
    # Normalize table names from the references.
    tables = {normalize_table_name(ref) for ref in refs if normalize_table_name(ref)}
    # If we cannot detect table names, allow (conservative behavior).
    if not tables:
        return True
    # Compare against the known schema tables in a case-insensitive way.
    allowed = {name.lower() for name in schema.keys()}
    return all(table.lower() in allowed for table in tables)


# Default safety cap on result size.
def ensure_limit(sql: str, limit: int = 200) -> str:
    """Add a LIMIT if the query doesn't already specify one."""
    # Respect existing LIMIT/FETCH clauses and avoid double-appending.
    if re.search(r"\blimit\b|\bfetch\b", sql, flags=re.IGNORECASE):
        return sql
    # Append a LIMIT clause to keep result sizes manageable.
    return f"{sql} LIMIT {limit}"
