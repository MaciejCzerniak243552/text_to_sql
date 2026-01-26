import re
from typing import Any, Dict, List, Optional

import streamlit as st

# Optional deps for nicer tables/charts.
try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import plotly.express as px
except ImportError:
    px = None


def charts_available() -> bool:
    """Return True when both pandas and plotly are installed."""
    return pd is not None and px is not None


# Detect datetime-like columns for better charts.
def coerce_datetime_columns(df):
    """Convert object columns to datetimes when they look date-like."""
    # Skip conversion if pandas is not available.
    if pd is None:
        return df
    # Inspect each column for datetime-like values.
    for col in df.columns:
        if df[col].dtype == "object":
            parsed = pd.to_datetime(df[col], errors="coerce")
            # Treat column as datetime if most values parse cleanly.
            if parsed.notna().mean() >= 0.8:
                df[col] = parsed
    return df


def coerce_numeric_columns(df):
    """Convert object columns to numeric when they look numeric-like."""
    if pd is None:
        return df
    for col in df.columns:
        if df[col].dtype == "object":
            series = df[col].astype(str).str.strip()
            # Skip columns that contain letters (e.g., "2024-Q1") to preserve bucket labels.
            if series.str.contains(r"[A-Za-z]", regex=True, na=False).mean() >= 0.1:
                continue
            parsed = pd.to_numeric(series, errors="coerce")
            if parsed.notna().mean() >= 0.8:
                df[col] = parsed
                continue
            cleaned = series.str.replace(r"[,\s$]", "", regex=True)
            cleaned = cleaned.str.replace(r"[A-Za-z]", "", regex=True)
            parsed = pd.to_numeric(cleaned, errors="coerce")
            if parsed.notna().mean() >= 0.8:
                df[col] = parsed
    return df


# Pick a reasonable chart (line for time series, bar otherwise).
def build_chart(df, intent: Optional[Dict[str, Any]] = None):
    """Build a plotly chart from the dataframe when possible."""
    # Charts require pandas, plotly, and at least one row.
    if pd is None or px is None or df.empty:
        return None
    df = coerce_datetime_columns(df.copy())
    df = coerce_numeric_columns(df)
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    # No numeric columns means no sensible chart.
    if not numeric_cols:
        return None

    x_col = None
    # Pick the first non-numeric column as the x-axis.
    for col in df.columns:
        if col not in numeric_cols:
            x_col = col
            break
    # Fall back to a synthetic index when all columns are numeric.
    if x_col is None:
        time_candidates = [col for col in df.columns if re.search(r"(date|day|month|year|time)", col, re.I)]
        if time_candidates:
            candidate = time_candidates[0]
            remaining = [col for col in numeric_cols if col != candidate]
            if remaining:
                x_col = candidate
                numeric_cols = remaining
        if x_col is None:
            df = df.reset_index(drop=True)
            df["index"] = df.index
            x_col = "index"

    chart_type = (intent or {}).get("chart_type", "auto")
    grain = (intent or {}).get("grain", "auto")
    requested_x = (intent or {}).get("x", "auto")
    requested_y = (intent or {}).get("y", "auto")

    if requested_x != "auto" and requested_x in df.columns:
        x_col = requested_x
    elif grain != "auto" and grain != "none":
        datetime_candidates = [
            col for col in df.columns if pd.api.types.is_datetime64_any_dtype(df[col])
        ]
        if datetime_candidates:
            x_col = datetime_candidates[0]

    if requested_y != "auto" and requested_y in numeric_cols:
        y_cols = requested_y
    else:
        preferred = ["net_revenue", "gross_revenue", "refunds", "revenue", "total"]
        y_cols = None
        for candidate in preferred:
            for col in numeric_cols:
                if col.lower() == candidate:
                    y_cols = col
                    break
            if y_cols:
                break
        if y_cols is None:
            y_cols = numeric_cols if len(numeric_cols) > 1 else numeric_cols[0]

    if x_col and pd.api.types.is_datetime64_any_dtype(df[x_col]):
        df = df.sort_values(x_col)
    elif x_col and intent and intent.get("grain") in {"quarter", "month", "week"}:
        # Preserve categorical bucket labels for quarters/months/weeks.
        df[x_col] = df[x_col].astype(str)

    if chart_type == "auto":
        if x_col and pd.api.types.is_datetime64_any_dtype(df[x_col]):
            chart_type = "line"
        else:
            chart_type = "bar"

    if chart_type == "scatter":
        if len(numeric_cols) < 2:
            return None
        return px.scatter(df, x=numeric_cols[0], y=numeric_cols[1])
    if chart_type == "hist":
        return px.histogram(df, x=y_cols if isinstance(y_cols, str) else numeric_cols[0])
    if chart_type == "pie":
        if x_col is None:
            x_col = df.columns[0]
        y_value = y_cols if isinstance(y_cols, str) else numeric_cols[0]
        return px.pie(df, names=x_col, values=y_value)
    if chart_type == "line":
        return px.line(df, x=x_col, y=y_cols)
    if chart_type == "bar":
        if isinstance(y_cols, list):
            y_cols = y_cols[0]
        return px.bar(df, x=x_col, y=y_cols)
    return None


# Render tabular data plus an optional chart.
def render_results(
    rows: List[Dict[str, Any]],
    show_chart: bool = False,
    intent: Optional[Dict[str, Any]] = None,
    key_prefix: Optional[str] = None,
) -> None:
    """Display results as a table and an optional chart."""
    # Show rows if we have any.
    if rows:
        # Without pandas, render the list of dicts directly.
        if pd is None:
            st.dataframe(rows, use_container_width=True, key=f"{key_prefix}-table" if key_prefix else None)
            return
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, key=f"{key_prefix}-table" if key_prefix else None)
        # Only render a chart when explicitly requested.
        should_plot = intent.get("requested") if intent else show_chart
        if should_plot:
            fig = build_chart(df, intent=intent)
            if fig is not None:
                st.plotly_chart(fig, use_container_width=True, key=f"{key_prefix}-chart" if key_prefix else None)
            else:
                st.caption(
                    "Chart not available. Provide a time/category column and at least one numeric column."
                )
    else:
        st.info("No rows returned.")
