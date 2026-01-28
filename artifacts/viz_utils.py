from typing import Any, Dict, List

import streamlit as st

try:
    import pandas as pd
except ImportError:
    pd = None

try:
    import plotly.express as px
    import plotly.graph_objects as go
except ImportError:
    px = None
    go = None


# Professional color palette
COLORS = {
    "primary": "#3b82f6",
    "secondary": "#6366f1", 
    "success": "#10b981",
    "warning": "#f59e0b",
    "error": "#ef4444",
    "neutral": "#64748b",
}

CHART_COLORS = [
    "#3b82f6",  # Blue
    "#10b981",  # Green
    "#f59e0b",  # Amber
    "#8b5cf6",  # Purple
    "#ec4899",  # Pink
    "#06b6d4",  # Cyan
    "#f97316",  # Orange
    "#84cc16",  # Lime
]

# Dark theme for Plotly charts
CHART_LAYOUT = {
    "template": "plotly_dark",
    "paper_bgcolor": "rgba(0,0,0,0)",
    "plot_bgcolor": "rgba(26,31,46,0.8)",
    "font": {"family": "Source Sans Pro, sans-serif", "color": "#a0aec0"},
    "title": {"font": {"size": 16, "color": "#fafafa"}},
    "legend": {"bgcolor": "rgba(0,0,0,0)", "font": {"size": 12}},
    "xaxis": {
        "gridcolor": "rgba(45,55,72,0.6)",
        "linecolor": "rgba(45,55,72,0.8)",
        "tickfont": {"size": 11},
    },
    "yaxis": {
        "gridcolor": "rgba(45,55,72,0.6)",
        "linecolor": "rgba(45,55,72,0.8)",
        "tickfont": {"size": 11},
    },
    "margin": {"l": 60, "r": 30, "t": 40, "b": 50},
}


def charts_available() -> bool:
    """Check if charting libraries are available."""
    return pd is not None and px is not None


def coerce_datetime_columns(df):
    """Convert columns that look like dates to datetime."""
    if pd is None:
        return df
    for col in df.columns:
        if df[col].dtype == "object":
            parsed = pd.to_datetime(df[col], errors="coerce")
            if parsed.notna().mean() >= 0.8:
                df[col] = parsed
    return df


def format_value(val, col_name: str = "") -> str:
    """Format values for display."""
    if pd is None:
        return str(val)
    if pd.isna(val):
        return "—"
    
    col_lower = col_name.lower()
    
    # Currency formatting
    if any(term in col_lower for term in ["price", "amount", "total", "revenue", "cost", "value"]):
        try:
            return f"${float(val):,.2f}"
        except (ValueError, TypeError):
            return str(val)
    
    # Percentage formatting
    if "percent" in col_lower or "rate" in col_lower:
        try:
            return f"{float(val):.1f}%"
        except (ValueError, TypeError):
            return str(val)
    
    # Large number formatting
    if isinstance(val, (int, float)) and not isinstance(val, bool):
        if abs(val) >= 1_000_000:
            return f"{val:,.0f}"
        elif abs(val) >= 1_000:
            return f"{val:,.0f}"
        elif isinstance(val, float):
            return f"{val:.2f}"
    
    return str(val)


def build_chart(df):
    """Build a styled Plotly chart."""
    if pd is None or px is None or df.empty:
        return None
    
    df = coerce_datetime_columns(df.copy())
    numeric_cols = df.select_dtypes(include="number").columns.tolist()
    
    if not numeric_cols:
        return None

    # Find x-axis column
    x_col = None
    for col in df.columns:
        if col not in numeric_cols:
            x_col = col
            break
    
    if x_col is None:
        df = df.reset_index(drop=True)
        df["Index"] = df.index
        x_col = "Index"

    # Choose chart type
    is_timeseries = pd.api.types.is_datetime64_any_dtype(df[x_col])
    
    if is_timeseries:
        fig = px.line(
            df, x=x_col, y=numeric_cols,
            color_discrete_sequence=CHART_COLORS
        )
        fig.update_traces(line={"width": 2})
    else:
        y_cols = numeric_cols if len(numeric_cols) > 1 else numeric_cols[0]
        fig = px.bar(
            df, x=x_col, y=y_cols,
            color_discrete_sequence=CHART_COLORS,
            barmode="group"
        )
        fig.update_traces(marker={"line": {"width": 0}})
    
    # Apply professional styling
    fig.update_layout(**CHART_LAYOUT)
    fig.update_layout(
        height=400,
        showlegend=len(numeric_cols) > 1,
        hovermode="x unified",
    )
    
    return fig


def render_results(rows: List[Dict[str, Any]], show_chart: bool = False) -> None:
    """Display results as a styled table and optional chart."""
    if not rows:
        st.info("No results found.")
        return
    
    if pd is None:
        st.dataframe(rows, use_container_width=True)
        return
    
    df = pd.DataFrame(rows)
    
    # Configure column formatting
    column_config = {}
    for col in df.columns:
        col_lower = col.lower()
        
        if any(term in col_lower for term in ["price", "amount", "total", "revenue", "cost", "value"]):
            column_config[col] = st.column_config.NumberColumn(
                col,
                format="$%.2f"
            )
        elif "percent" in col_lower or "rate" in col_lower:
            column_config[col] = st.column_config.NumberColumn(
                col,
                format="%.1f%%"
            )
        elif "date" in col_lower or "time" in col_lower:
            column_config[col] = st.column_config.DatetimeColumn(
                col,
                format="YYYY-MM-DD"
            )
    
    # Display table
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config=column_config
    )
    
    # Display chart
    if show_chart:
        fig = build_chart(df)
        if fig is not None:
            st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
