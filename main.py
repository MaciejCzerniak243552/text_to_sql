import json
import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

import streamlit as st
from langchain_ollama.llms import OllamaLLM

from artifacts.config_utils import build_db_url, get_setting, load_dotenv_file
from artifacts.db_utils import extract_schema, run_query
from artifacts.llm_utils import fix_sql_query, generate_answer, infer_range_answer, to_sql_query
from artifacts.sql_safety import ensure_limit, is_safe_sql
from artifacts.viz_utils import charts_available, render_results

load_dotenv_file(os.getenv("DOTENV_PATH", ".env"))

# ─────────────────────────────────────────────────────────────────────────────
# Page Configuration
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Query Assistant",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# Professional Dark Theme
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Import clean professional font */
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;600;700&display=swap');
    
    /* Root variables */
    :root {
        --bg-primary: #0e1117;
        --bg-secondary: #1a1f2e;
        --bg-tertiary: #252b3b;
        --text-primary: #fafafa;
        --text-secondary: #a0aec0;
        --accent: #3b82f6;
        --accent-hover: #2563eb;
        --success: #10b981;
        --error: #ef4444;
        --border: #2d3748;
    }
    
    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Source Sans Pro', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Hide Streamlit branding */
    #MainMenu, footer, header[data-testid="stHeader"] {
        display: none !important;
    }
    
    /* Main container */
    .stApp {
        background: var(--bg-primary);
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: var(--bg-secondary);
        border-right: 1px solid var(--border);
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: var(--text-secondary);
    }
    
    /* Sidebar header */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--text-primary) !important;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background: var(--bg-tertiary) !important;
        border-radius: 6px;
        font-weight: 500;
    }
    
    .streamlit-expanderContent {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-top: none;
        border-radius: 0 0 6px 6px;
    }
    
    /* Chat messages */
    .stChatMessage {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border);
        border-radius: 8px;
        padding: 1rem;
        margin-bottom: 0.75rem;
    }
    
    /* Chat input */
    .stChatInput > div {
        background: var(--bg-secondary) !important;
        border: 1px solid var(--border) !important;
        border-radius: 8px;
    }
    
    .stChatInput input {
        color: var(--text-primary) !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: var(--bg-tertiary);
        color: var(--text-primary);
        border: 1px solid var(--border);
        border-radius: 6px;
        font-weight: 500;
        transition: all 0.2s ease;
    }
    
    .stButton > button:hover {
        background: var(--accent);
        border-color: var(--accent);
    }
    
    /* Toggle */
    .stCheckbox label span {
        color: var(--text-secondary) !important;
    }
    
    /* Dataframe */
    .stDataFrame {
        border: 1px solid var(--border);
        border-radius: 8px;
        overflow: hidden;
    }
    
    /* Code blocks */
    .stCodeBlock {
        background: var(--bg-tertiary) !important;
        border: 1px solid var(--border);
        border-radius: 6px;
    }
    
    /* Headers */
    h1, h2, h3 {
        color: var(--text-primary) !important;
        font-weight: 700;
        letter-spacing: -0.03em;
    }
    
    /* Spinner */
    .stSpinner > div {
        border-color: var(--accent) transparent transparent transparent !important;
    }
    
    /* Info/Error boxes */
    .stAlert {
        background: var(--bg-secondary);
        border: 1px solid var(--border);
        border-radius: 6px;
    }
    
    /* Custom header styling */
    .main-header {
        padding: 1.5rem 0 1rem 0;
        border-bottom: 1px solid var(--border);
        margin-bottom: 1.5rem;
    }
    
    .main-title {
        font-size: 1.75rem;
        font-weight: 700;
        color: var(--text-primary);
        margin: 0;
        letter-spacing: -0.03em;
    }
    
    .main-subtitle {
        font-size: 0.9rem;
        color: var(--text-secondary);
        margin-top: 0.25rem;
    }
    
    /* Status indicator */
    .status-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        background: var(--success);
        border-radius: 50%;
        margin-right: 6px;
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.5; }
    }
    
    /* Clean scrollbar */
    ::-webkit-scrollbar {
        width: 6px;
        height: 6px;
    }
    
    ::-webkit-scrollbar-track {
        background: var(--bg-primary);
    }
    
    ::-webkit-scrollbar-thumb {
        background: var(--border);
        border-radius: 3px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: var(--text-secondary);
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ─────────────────────────────────────────────────────────────────────────────
def wants_chart(question: str) -> bool:
    """Detect if user wants a visualization."""
    return bool(re.search(
        r"\b(plot|chart|graph|visuali[sz]e|trend|line|bar|histogram|scatter|pie|diagram)\b",
        question, re.I
    ))


def get_db_display_name(db_url: str) -> str:
    """Return a friendly database name."""
    name = get_setting("DB_NAME")
    if name:
        return name
    if db_url.startswith("sqlite"):
        path = db_url.split("///", 1)[-1]
        return os.path.basename(path) or "SQLite"
    parsed = urlsplit(db_url)
    return parsed.path.lstrip("/") if parsed.path else "Database"


def needs_clarification(question: str, schema: Dict[str, List[str]]) -> Optional[str]:
    """Check if question needs more detail."""
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", question.lower())
    if len(tokens) < 3:
        return "Could you provide more detail about what you'd like to know?"
    
    schema_terms = {name.lower() for name in schema.keys()}
    for cols in schema.values():
        schema_terms.update(col.lower() for col in cols)
    
    mentions_schema = any(tok in schema_terms for tok in tokens)
    generic = {"show", "list", "data", "info", "details", "report", "stats", "summary", "everything", "all", "overview"}
    has_generic = any(tok in generic for tok in tokens)
    has_filter = any(
        tok in {"count", "average", "avg", "min", "max", "sum", "total", "range", "between", 
                "before", "after", "since", "during", "latest", "last", "top", "bottom", "most", "least"}
        for tok in tokens
    ) or bool(re.search(r"\d", question))
    
    if has_generic and not mentions_schema and not has_filter:
        tables = sorted(schema.keys())[:3]
        if tables:
            return f"Could you be more specific? Try asking about: {', '.join(tables)}"
        return "Could you specify what data you're looking for?"
    return None


def render_sql(sql: str, index: int) -> None:
    """Display SQL in expander."""
    with st.expander("View SQL", expanded=False):
        st.code(sql, language="sql")


def query_with_retries(
    query: str,
    schema_details: str,
    schema: Dict[str, List[str]],
    db_url: str,
    model: OllamaLLM,
    limit: int,
    max_retries: int,
) -> Dict[str, Any]:
    """Generate and execute SQL with retry logic."""
    sql = to_sql_query(query, schema_details, model)
    seen_sql = set()
    last_error = None

    for attempt in range(max_retries + 1):
        sql = ensure_limit(sql, limit=limit)
        
        if not is_safe_sql(sql, schema):
            return {"sql": sql, "rows": None, "error": "Query validation failed."}
        
        if sql in seen_sql:
            return {"sql": sql, "rows": None, "error": "Could not generate a working query."}
        seen_sql.add(sql)
        
        try:
            rows = run_query(db_url, sql)
            return {"sql": sql, "rows": rows, "error": None}
        except Exception as exc:
            last_error = str(exc)
            if attempt >= max_retries:
                break
            sql = fix_sql_query(query, schema_details, sql, last_error, model)

    return {"sql": sql, "rows": None, "error": f"Query failed: {last_error}"}


# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
OLLAMA_MODEL = get_setting("OLLAMA_MODEL")
OLLAMA_BASE_URL = get_setting("OLLAMA_BASE_URL") or get_setting("OLLAMA_HOST")

if not OLLAMA_MODEL:
    st.error("Missing OLLAMA_MODEL configuration.")
    st.stop()

model = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL) if OLLAMA_BASE_URL else OllamaLLM(model=OLLAMA_MODEL)

QUERY_LIMIT = int(get_setting("QUERY_LIMIT", "200"))
SQL_MAX_RETRIES = int(get_setting("SQL_MAX_RETRIES", "2"))

db_url = build_db_url()
if not db_url:
    st.error("Database configuration missing.")
    st.stop()

db_name = get_db_display_name(db_url)

try:
    schema = extract_schema(db_url)
    schema_details = json.dumps(schema, ensure_ascii=True)
except Exception as exc:
    st.error(f"Could not connect to database: {exc}")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### Database Schema")
    st.caption(f"Connected to **{db_name}**")
    
    for table_name in sorted(schema.keys()):
        with st.expander(f"📋 {table_name}"):
            for col in schema[table_name]:
                st.markdown(f"• `{col}`")
    
    st.divider()
    
    # Controls
    show_chart = st.toggle("Show visualizations", value=False, help="Display charts for numeric results")
    
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()
    
    st.divider()
    st.caption(f"Model: `{OLLAMA_MODEL}`")

# ─────────────────────────────────────────────────────────────────────────────
# Main Content
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <div class="main-title">Query Assistant</div>
    <div class="main-subtitle">
        <span class="status-dot"></span>
        Ask questions about your data in natural language
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize chat
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for idx, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            if message.get("rows") is not None:
                render_results(message["rows"], show_chart=message.get("show_chart", False))
            if message.get("error"):
                st.error(message["error"])
            if message.get("sql"):
                render_sql(message["sql"], idx)

# Chat input
if prompt := st.chat_input("What would you like to know?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Check for clarification
    clarification = needs_clarification(prompt, schema)
    if clarification:
        with st.chat_message("assistant"):
            st.markdown(clarification)
        st.session_state.messages.append({"role": "assistant", "content": clarification})
        st.stop()
    
    # Process query
    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            result = query_with_retries(
                prompt, schema_details, schema, db_url, model, QUERY_LIMIT, SQL_MAX_RETRIES
            )
        
        if result["error"]:
            st.error(result["error"])
            st.session_state.messages.append({
                "role": "assistant",
                "content": "I encountered an issue processing that query.",
                "error": result["error"],
                "sql": result.get("sql"),
            })
            st.stop()
        
        sql = result["sql"]
        rows = result["rows"]
        plot_requested = show_chart or wants_chart(prompt)
        
        # Generate answer
        answer_rows = rows[:int(get_setting("ANSWER_MAX_ROWS", "50"))]
        with st.spinner(""):
            try:
                range_answer = infer_range_answer(answer_rows)
                answer = range_answer if range_answer else generate_answer(prompt, answer_rows, model)
            except Exception as exc:
                answer = f"Retrieved {len(rows)} results."
        
        st.markdown(answer)
        render_results(rows, show_chart=plot_requested)
        render_sql(sql, len(st.session_state.messages))
        
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sql": sql,
            "rows": rows,
            "show_chart": plot_requested,
        })
