import base64
import os
import re
from typing import Any, Dict, List, Optional
from urllib.parse import urlsplit

# Text-to-SQL pipeline:
# 1) Build schema context from the database.
# 2) Ask the model for SQL (and fix it on errors).
# 3) Execute the query, summarize the results, and render tables/charts.

import streamlit as st
import streamlit.components.v1 as components
from langchain_ollama.llms import OllamaLLM

from artifacts.config_utils import build_db_url, get_setting, load_dotenv_file
from artifacts.db_utils import extract_schema, run_query
from artifacts.llm_utils import fix_sql_query, generate_answer, infer_range_answer, to_sql_query
from artifacts.sql_safety import ensure_limit, is_safe_sql, sql_safety_reason
from artifacts.viz_utils import charts_available, render_results

# Load environment variables from a local .env file when present.
load_dotenv_file(os.getenv("DOTENV_PATH", ".env"))


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
        tok in {"count", "average", "avg", "min", "max", "sum", "total", "range", "between", "before", "after", "since",
                "during", "latest", "last", "top", "bottom", "most", "least"}
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
    refer_words = r"\b(this|that|these|those|above|previous|last|prior|same)\b"
    return bool(re.search(plot_words, text) and re.search(refer_words, text))


def render_sql_button(sql: str, index: int) -> None:
    """Show SQL without triggering a rerun that cancels in-flight work."""
    with st.expander("Show SQL"):
        st.code(sql, wrap_lines=True, language="sql")

# Run SQL with retry loops and model-based fixes.
def query_with_retries(
    query: str,
    schema_details: str,
    schema: Dict[str, List[str]],
    db_url: str,
    model: OllamaLLM,
    limit: int,
    max_retries: int,
    intent: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Generate SQL, validate it, execute it, and retry with fixes if needed."""
    sql = to_sql_query(query, schema_details, model, intent=intent)
    seen_sql = set()
    last_error = None

    # Retry loop for query generation and fix attempts.
    for attempt in range(max_retries + 1):
        # Enforce limits and safety checks before execution.
        sql = ensure_limit(sql, limit=limit, chart_mode=bool(intent and intent.get("requested")))
        # Stop early if the SQL violates safety rules.
        safety_error = sql_safety_reason(sql, schema)
        if safety_error:
            return {"sql": sql, "rows": None, "error": safety_error}
        # Guard against repeated SQL that never fixes the error.
        if sql in seen_sql:
            return {"sql": sql, "rows": None, "error": "Generated SQL repeated without fixing the error."}
        seen_sql.add(sql)
        try:
            rows = run_query(db_url, sql)
            return {"sql": sql, "rows": rows, "error": None}
        except Exception as exc:
            last_error = str(exc)
            # Stop after the final retry.
            if attempt >= max_retries:
                break
            # Feed the DB error back to the model to repair the query.
            sql = fix_sql_query(query, schema_details, sql, last_error, model, intent=intent)

    return {"sql": sql, "rows": None, "error": f"Query failed: {last_error}"}


DEFAULT_GREETING = {
    "role": "assistant",
    "content": (
        """Hi! Ask questions about your data in plain language.
        Examples:
        • \"How many orders were placed last month?\"
        • \"Calculate daily revenue for February 2025\"
        • \"Top 5 categories by revenue\"
        If you want a chart, say \"plot\", \"chart\", \"line\", or \"bar\",
        or use the \"Plot results\" toggle. You can also follow up with
        \"Now plot those results\".
        If a question is unclear, I may ask for clarification.
        Use the \"Show SQL\" expander to see the query behind each answer."""
    ),
}


st.set_page_config(page_title="Query Assistant", initial_sidebar_state="expanded", layout="wide")
st.markdown(
    """
    <style>
    :root {
        --header-height: 80px;
        --header-offset: 12px;
        --footer-height: 80px;
        --input-height: 90px;
        --sidebar-width: 15vw;
    }
    * {
        box-sizing: border-box;
    }
    html, body, [data-testid="stAppViewContainer"] {
        height: 100%;
        overflow: hidden;
        overflow-x: hidden;
    }
    section.main {
        height: 100vh;
        overflow: hidden;
        overflow-x: hidden;
    }
    section.main > div {
        height: 100%;
        overflow: hidden;
        overflow-x: hidden;
        padding-top: calc(var(--header-height) + var(--header-offset));
        padding-bottom: var(--footer-height);
        box-sizing: border-box;
    }
    [data-testid="collapsedControl"] {
        display: none;
    }
    [data-testid="stSidebarCollapseButton"] {
        display: none;
    }
    button[title="Close sidebar"],
    button[title="Open sidebar"],
    button[aria-label="Close sidebar"],
    button[aria-label="Open sidebar"] {
        display: none;
    }
    [data-testid="stSidebarResizer"] {
        display: none !important;
    }
    section[data-testid="stSidebar"] {
        width: var(--sidebar-width) !important;
        min-width: var(--sidebar-width) !important;
        max-width: var(--sidebar-width) !important;
        flex: 0 0 var(--sidebar-width) !important;
    }
    section[data-testid="stSidebar"] > div:first-child {
        width: var(--sidebar-width) !important;
    }
    .app-header {
        position: fixed;
        top: var(--header-offset);
        left: 0;
        width: 100%;
        z-index: 1000;
        background: var(--background-color, white);
        border-bottom: 1px solid #ddd;
        padding: 10px 20px;
    }
    .app-title {
        font-size: 20px;
        font-weight: 600;
        margin: 0;
    }
    .app-meta {
        color: rgba(0, 0, 0, 0.6);
        font-size: 0.85rem;
    }
    .chat-scroll {
        overflow-y: auto;
        overflow-x: hidden;
        height: calc(100vh - var(--header-height) - var(--header-offset) - var(--footer-height) - var(--input-height));
        padding-right: 8px;
    }
    .fixed-footer {
        position: fixed;
        bottom: 0;
        left: 0;
        width: 100%;
        background-color: white;
        z-index: 1000;
        padding: 10px 20px;
        border-top: 1px solid #ddd;
        text-align: center;
    }
    div[data-testid="stChatInput"] {
        position: sticky;
        bottom: var(--footer-height);
        background: var(--background-color, white);
        padding-top: 8px;
    }
    .stPlotlyChart, .plot-container, .js-plotly-plot {
        max-width: 100% !important;
    }
    div[data-testid="stDataFrame"] {
        max-width: 100% !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Model and retry settings.
OLLAMA_MODEL = get_setting("OLLAMA_MODEL")
OLLAMA_BASE_URL = get_setting("OLLAMA_BASE_URL") or get_setting("OLLAMA_HOST")

if not OLLAMA_MODEL:
    st.error("OLLAMA_MODEL is missing. Set it in .env or .streamlit/secrets.toml.")
    st.stop()

# Use a remote Ollama base URL when configured; otherwise default to local.
if OLLAMA_BASE_URL:
    model = OllamaLLM(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
else:
    model = OllamaLLM(model=OLLAMA_MODEL)

QUERY_LIMIT = int(get_setting("QUERY_LIMIT", "200"))
SQL_MAX_RETRIES = int(get_setting("SQL_MAX_RETRIES", "2"))

# Resolve DB connection info.
db_url = build_db_url()
# Stop early if we cannot build a valid DB URL.
if not db_url:
    st.error(
        "Database settings are missing. Set DB_URL or DB_HOST/DB_USER/DB_PASSWORD/DB_NAME."
    )
    st.stop()

db_display_name = get_db_display_name(db_url)
header_container = st.container()
with header_container:
    st.markdown('<div id="header-anchor"></div>', unsafe_allow_html=True)
    header_left, header_center, header_right = st.columns([1.2, 3, 1.2], vertical_alignment="center")
    with header_left:
        st.markdown('')
    with header_center:
        st.markdown('<div class="app-title">Query Assistant</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="app-meta">Model: {OLLAMA_MODEL} | Database: {db_display_name}</div>',
            unsafe_allow_html=True,
        )
    with header_right:
        show_chart = st.toggle(
            "Plot results",
            value=False,
            key="show_chart",
            help="Force to render a chart for numeric results.",
        )
        if not charts_available():
            st.caption("Charts disabled until pandas and plotly are installed.")
        if st.button("Clear chat", use_container_width=True):
            st.session_state.messages = [DEFAULT_GREETING]
            st.rerun()

st.markdown(
    """
    <div class="fixed-footer">
        <p>(c) 2026 My App</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# Load schema once for the prompt and safety checks.
try:
    schema, schema_details = extract_schema(db_url)
except Exception as exc:
    st.error(f"Could not load schema: {exc}")
    st.stop()

# Sidebar logo at the top.
logo_path = os.path.join(os.path.dirname(__file__), "img", "alternate.png")
try:
    with open(logo_path, "rb") as handle:
        logo_b64 = base64.b64encode(handle.read()).decode("utf-8")
    st.sidebar.markdown(
        f"""
        <style>
        .sidebar-logo {{
            text-align: center;
            margin: 4px 0 12px;
        }}
        .sidebar-logo img {{
            width: 100%;
            height: auto;
            border: 0px solid #4FC3F7;
        }}
        </style>
        <div class="sidebar-logo">
            <a href="https://alternate.nl" target="_blank" rel="noopener noreferrer">
                <img src="data:image/png;base64,{logo_b64}" alt="alternate.nl" />
            </a>
        </div>
        """,
        unsafe_allow_html=True,
    )
except FileNotFoundError:
    pass

# Sidebar schema browser.
st.sidebar.header("Database schema")
st.sidebar.caption("Click a table to see its columns.")
for table_name in sorted(schema.keys()):
    with st.sidebar.expander(table_name):
        st.markdown("\n".join(f"- {col}" for col in schema[table_name]))

chat_container = st.container()
with chat_container:
    st.markdown('<div id="chat-scroll-anchor"></div>', unsafe_allow_html=True)
    # Initialize chat history once.
    if "messages" not in st.session_state:
        st.session_state.messages = [DEFAULT_GREETING]

    # Re-render chat history.
    for idx, message in enumerate(st.session_state.messages):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Only render extra details for assistant messages.
            if message["role"] == "assistant":
                # Show prior rows if present.
                if message.get("rows") is not None:
                    render_results(
                        message["rows"],
                        show_chart=message.get("show_chart", False),
                        intent=message.get("intent"),
                        key_prefix=f"history-{idx}",
                    )
                # Show prior error if present.
                if message.get("error"):
                    st.error(message["error"])
                # Optional SQL reveal.
                if message.get("sql"):
                    render_sql_button(message["sql"], idx)

components.html(
    """
    <script>
    const start = Date.now();
    const timer = setInterval(() => {
        const headerAnchor = window.parent.document.getElementById("header-anchor");
        if (headerAnchor) {
            const block = headerAnchor.closest('div[data-testid="stVerticalBlock"]');
            if (block && !block.classList.contains("app-header")) {
                block.classList.add("app-header");
            }
        }
        const chatAnchor = window.parent.document.getElementById("chat-scroll-anchor");
        if (chatAnchor) {
            const block = chatAnchor.closest('div[data-testid="stVerticalBlock"]');
            if (block && !block.classList.contains("chat-scroll")) {
                block.classList.add("chat-scroll");
            }
        }
        if ((headerAnchor && chatAnchor) || Date.now() - start > 3000) {
            clearInterval(timer);
        }
    }, 50);
    </script>
    """,
    height=0,
)

user_prompt = st.chat_input("Ask about your data")
# Only process when the user submits a question.
if user_prompt:
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    # Render the just-submitted question in the current run.
    with chat_container:
        with st.chat_message("user"):
            st.markdown(user_prompt)
    intent = parse_chart_intent(user_prompt, show_chart)
    last_result = get_last_result(st.session_state.messages)
    if intent["requested"] and last_result and is_followup_plot_request(user_prompt):
        with chat_container:
            with st.chat_message("assistant"):
                st.markdown("Here is the chart for the previous result.")
                render_results(
                    last_result["rows"],
                    show_chart=True,
                    intent=intent,
                    key_prefix=f"followup-{len(st.session_state.messages)}",
                )
                if last_result.get("sql"):
                    render_sql_button(last_result["sql"], len(st.session_state.messages))
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "Here is the chart for the previous result.",
                "sql": last_result.get("sql"),
                "rows": last_result.get("rows"),
                "show_chart": True,
                "intent": intent,
            }
        )
        st.stop()
    clarification = needs_clarification(user_prompt, schema)
    if clarification:
        with chat_container:
            with st.chat_message("assistant"):
                st.markdown(clarification)
        st.session_state.messages.append({"role": "assistant", "content": clarification})
        st.stop()
    with chat_container:
        with st.chat_message("assistant"):
            with st.spinner("Generating SQL..."):
                result = query_with_retries(
                    user_prompt,
                    schema_details,
                    schema,
                    db_url,
                    model,
                    QUERY_LIMIT,
                    SQL_MAX_RETRIES,
                    intent=intent,
                )

            # Surface errors before attempting to answer.
            if result["error"]:
                error = result["error"]
                st.error(error)
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": "I ran into an error while answering that.",
                        "error": error,
                        "sql": result.get("sql"),
                    }
                )
                st.stop()

            sql = result["sql"]
            rows = result["rows"]
            plot_requested = intent["requested"]

            # Summarize results, then show data and optional SQL.
            answer_rows = rows[: int(get_setting("ANSWER_MAX_ROWS", "50"))]
            with st.spinner("Generating answer..."):
                try:
                    range_answer = infer_range_answer(answer_rows)
                    # Prefer a direct range explanation when detected.
                    if range_answer:
                        answer = range_answer
                    else:
                        answer = generate_answer(user_prompt, answer_rows, model)
                except Exception as exc:
                    answer = f"I could not generate an answer: {exc}"
            st.markdown(answer)
            render_results(
                rows,
                show_chart=plot_requested,
                intent=intent,
                key_prefix=f"result-{len(st.session_state.messages)}",
            )
            render_sql_button(sql, len(st.session_state.messages))

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": answer,
                    "sql": sql,
                    "rows": rows,
                    "show_chart": plot_requested,
                    "intent": intent,
                }
            )
