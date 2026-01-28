"""Streamlit chat UI and orchestration for the text-to-SQL pipeline."""

import base64
import os
from typing import Optional

# Text-to-SQL pipeline overview:
# 1) Build schema context from the database.
# 2) Ask the model for SQL (and fix it on errors).
# 3) Execute the query, summarize the results, and render tables/charts.

import streamlit as st
import streamlit.components.v1 as components
from langchain_ollama.llms import OllamaLLM

from artifacts.config_utils import build_db_url, get_setting, load_dotenv_file
from artifacts.chat_utils import (
    get_db_display_name,
    get_last_result,
    is_followup_plot_request,
    needs_clarification,
    parse_chart_intent,
    query_with_retries,
)
from artifacts.db_utils import extract_schema
from artifacts.llm_utils import generate_answer, infer_range_answer
from artifacts.viz_utils import charts_available, render_results

# Load environment variables from a local .env file when present.
load_dotenv_file(os.getenv("DOTENV_PATH", ".env"))


@st.cache_resource(show_spinner=False)
def get_ollama_client(model_name: str, base_url: Optional[str]) -> OllamaLLM:
    """Cache the Ollama client so models stay warm across Streamlit reruns."""
    if base_url:
        return OllamaLLM(model=model_name, base_url=base_url)
    return OllamaLLM(model=model_name)


def render_sql_button(sql: str, index: int) -> None:
    """Show SQL without triggering a rerun that cancels in-flight work."""
    with st.expander("Show SQL"):
        st.code(sql, wrap_lines=True, language="sql")

# Initial assistant message shown when a new chat starts.
DEFAULT_GREETING = {
    "role": "assistant",
    "is_greeting": True,
    "content": (
        """
        <div style="font-size:20px;font-weight:600;margin-bottom:6px;">Welcome to Query Assistant</div>
        <div style="margin-bottom:8px;">Ask questions about your database. Examples:</div>
        <ul style="margin-top:0;margin-bottom:8px;">
          <li>How many orders were placed last month?</li>
          <li>Calculate daily revenue for February 2025</li>
          <li>Top 5 categories by revenue</li>
        </ul>
        <div style="margin-bottom:6px;">
          If you want a chart, say <b>plot</b>, <b>chart</b>, <b>line</b>, or <b>bar</b>,
          or use the <b>Plot results</b> toggle.
        </div>
        <div style="margin-bottom:6px;">
          You can also follow up with "Now plot those results".
          If a question is unclear, I may ask for clarification.
        </div>
        <div>Use the <b>Show SQL</b> expander to see the query behind each answer.</div>
        """
    ),
}

st.set_page_config(page_title="Query Assistant", initial_sidebar_state="expanded", layout="wide")
# Layout + scroll behavior is controlled via CSS so the chat history is the only scroller.
st.markdown(
    """
    <style>
    :root {
        --header-height: 110px;
        --header-offset: 40px;
        --footer-height: 80px;
        --input-height: 90px;
        --sidebar-width: 16vw;
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
        padding: 20px 20px;
    }
    .app-title {
        font-size: 25px;
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
model = get_ollama_client(OLLAMA_MODEL, OLLAMA_BASE_URL)

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
    header_left, header_center, header_right_center, header_right = st.columns([0.9, 3, 0.6, 1.2], vertical_alignment="center")
    with header_left:
        st.markdown('')
    with header_center:
        st.markdown('<div class="app-title">Query Assistant</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div class="app-meta">Model: {OLLAMA_MODEL} | Database: {db_display_name}</div>',
            unsafe_allow_html=True,
        )
    with header_right_center:
        show_chart = st.toggle(
            "Plot results",
            value=False,
            key="show_chart",
            help="Force to render a chart for numeric results.",
        )
    with header_right:
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
            if message.get("is_greeting"):
                st.markdown(message["content"], unsafe_allow_html=True)
            else:
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
        // Attach CSS classes after the DOM is ready (Streamlit rebuilds often).
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
        # Allow "plot that" follow-ups without re-querying the database.
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


