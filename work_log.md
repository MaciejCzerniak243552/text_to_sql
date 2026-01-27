# Work Log

This log summarizes the evolution of the Query Assistant project. It focuses on code, structure, and behavior changes from the start through the current state. Dates are approximate.

## Phase 1 - Initial requirements and foundations
1) Clarified the goal: a Streamlit chat app that translates natural language questions into SQL, runs queries, and returns plain-language answers with optional charts.
2) Added support for environment-based configuration (.env) and guidance on secrets.
3) Introduced database execution in the pipeline (not only SQL generation), with retry and error-handling behavior.
4) Established a UI with chat input, message history, and a "Show SQL" toggle.

## Phase 2 - Refactor into a structured codebase
1) Split the monolithic script into helper modules under `artifacts/` (LLM helpers, DB utilities, safety, visualization, config).
2) Created a clean project layout (main app file plus artifacts, README, requirements, .env).
3) Added a README with setup and run instructions.
4) Added a SQL data seeding script for local testing.

## Phase 3 - Query pipeline improvements
1) Added deterministic chart intent parsing (plot/chart keywords and grain detection).
2) Added chart-aware SQL prompting and answer formatting rules.
3) Improved SQL safety checks (read-only enforcement, schema allow-listing, comment stripping, etc.).
4) Added error feedback loops (pass database errors back to the model to fix SQL).
5) Added range-answer detection for common min/max patterns.

## Phase 4 - Schema and join stability
1) Enhanced schema extraction to include column types, relationships, and date hints.
2) Added guidance for time bucketing (day/week/month/quarter/year).
3) Added rules to avoid incorrect quarter grouping (include year with quarter).
4) Added refund aggregation guidance to avoid double counting.

## Phase 5 - Visualization improvements
1) Added plot rendering support when pandas and plotly are present.
2) Improved x/y selection logic in charts (prefer time series and numeric columns).
3) Added logic to avoid mis-coercing categorical time buckets.
4) Added unique keys to Plotly charts to prevent duplicate element ID errors.

## Phase 6 - UI and layout changes
1) Added a fixed header/footer layout and made the chat area the primary scroller.
2) Added/adjusted sidebar behavior, including logos and schema browser.
3) Added plotting toggles and clarified display of model and database name.
4) Added a custom HTML greeting message to guide user questions.

## Phase 7 - Model strategy changes
1) Explored dual-model setup (SQL model vs answer model) for speed/quality tradeoffs.
2) Added an Ollama client cache via `@st.cache_resource` to reduce cold starts.
3) Switched models multiple times based on stability and latency testing (qwen3:latest, sqlcoder:7b, qwen3:1.7b, qwen3:8b).
4) Noted model/template mismatches: smaller or specialized models (sqlcoder:7b, qwen3:1.7b) were less consistent with the prompt constraints, leading to invented columns (e.g., non-existent fields), incorrect join keys, or chart-shape drift. This influenced the decision to keep qwen3:8b for stability and to strengthen prompt guardrails.

## Phase 8 - Prompt tightening and stabilization
1) Simplified and hardened SQL prompts to reduce hallucinations.
2) Added explicit join rules and "do not guess columns" guardrails.
3) Reduced prompt bloat by removing redundant or conflicting rules.
4) Added a rule: if the user says "revenue" without qualifiers, return gross revenue only and name the column `gross_revenue`.

## Phase 9 - Backend refactor cleanup
1) Moved non-UI helpers out of `main.py` into `artifacts/chat_utils.py`:
   - intent parsing
   - follow-up detection
   - clarification checks
   - last-result lookup
   - query retries
2) Kept UI-only code in `main.py` (layout, rendering, Streamlit widgets).

## Phase 10 - Bug fixes and UX adjustments
1) Fixed follow-up detection to avoid triggering "last month" as "plot last result".
2) Improved greeting formatting via HTML (rendered only for the greeting message).
3) Reworked prompt ordering so join rules appear before revenue rules.
4) Adjusted fix templates to restore critical constraints after earlier tightening.

## Rollbacks and experiments
- Several experiments were tried and rolled back by the user (e.g., overly aggressive schema minimization, model swaps, prompt variants). These are noted here to document the evolution and lessons learned.

## Current state (as of now)
- `main.py` is a Streamlit UI layer with cached Ollama client and a custom HTML greeting.
- Non-UI logic lives in `artifacts/` modules (LLM, DB, chat helpers, safety, viz).
- LLM templates are lean but include strong guardrails against hallucinated columns and incorrect joins.
- The app supports optional charts and a follow-up "plot last result" workflow.

## Suggested next steps
- Re-test main questions with qwen3:8b after the latest prompt adjustments.
- If latency remains high, consider streaming or keep-alive settings for Ollama.
- Review the SQL safety and prompt rules after real user feedback.
