# Query Assistant

A Streamlit chat app that translates natural-language questions into SQL, runs them against a MySQL database, and returns a plain-language answer with optional charts.

## Features
- Natural-language Q&A over your database
- Deterministic chart intent parsing (no extra LLM call)
- Chart-aware SQL generation with explicit revenue/date rules
- Safe, read-only SQL guardrails
- Sidebar schema browser

## Requirements
- Python 3.10+
- A running MySQL database
- Ollama (local or remote) with a supported model

## Install
Create a virtual environment and install dependencies:

```bash
python -m venv .venv
# Windows
.\.venv\Scripts\activate
# macOS/Linux
# source .venv/bin/activate

pip install -U streamlit langchain-ollama sqlalchemy mysqlclient
# Optional (for charts)
pip install -U pandas plotly
```

## Configuration
Create a `.env` file in the project root (do not commit secrets):

```env
# Ollama
OLLAMA_MODEL=qwen3:latest
OLLAMA_BASE_URL=http://localhost:11434

# Database (choose one approach)
# Option A: full URL
DB_URL=mysql+pymysql://user:password@host:3306/database

# Option B: separate parts
DB_DIALECT=mysql+pymysql
DB_HOST=host
DB_PORT=3306
DB_USER=user
DB_PASSWORD=password
DB_NAME=database

# Query behavior
QUERY_LIMIT=200
SQL_MAX_RETRIES=2
ANSWER_MAX_ROWS=50
```

If you prefer Streamlit secrets, put the same keys in `.streamlit/secrets.toml`.

## Run
From the project root:

```bash
streamlit run main.py
```

## Usage
1. Ask a question in plain language.
2. Use **Plot results** or mention “plot/chart/line/bar” to request a chart.
3. Use **Show SQL** under a response to inspect the query.

## Chart Intent Rules (Summary)
The app uses deterministic parsing to decide chart type and grain:
- **Requested** if the toggle is on or the prompt includes plot/chart/graph/trend/etc.
- **Chart type** from keywords (line/bar/scatter/pie/hist); otherwise auto.
- **Grain** from keywords (daily/weekly/monthly/quarterly/yearly); otherwise auto.

## Troubleshooting
- **No charts**: install `pandas` and `plotly`.
- **Model not found**: run `ollama list` and set `OLLAMA_MODEL`.
- **Database errors**: verify DB credentials and permissions.

## Security
This app generates **read-only** SQL and blocks unsafe statements, but you should still use a read-only DB user in production.
