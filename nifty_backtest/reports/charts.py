"""Plotly chart helpers for backtest reporting."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def equity_curve_figure(equity_df: pd.DataFrame) -> go.Figure:
    """Build an equity curve figure."""

    fig = go.Figure()
    if equity_df is not None and not equity_df.empty and {"datetime", "equity"}.issubset(equity_df.columns):
        fig.add_trace(go.Scatter(x=equity_df["datetime"], y=equity_df["equity"], mode="lines", name="Equity"))
    fig.update_layout(title="Equity Curve", xaxis_title="Time", yaxis_title="Equity")
    return fig


def drawdown_figure(equity_df: pd.DataFrame) -> go.Figure:
    """Build a drawdown figure."""

    fig = go.Figure()
    if equity_df is not None and not equity_df.empty and {"datetime", "equity"}.issubset(equity_df.columns):
        curve = equity_df.copy()
        curve["peak"] = curve["equity"].cummax()
        curve["drawdown_pct"] = ((curve["peak"] - curve["equity"]) / curve["peak"].replace(0, pd.NA)) * 100.0
        fig.add_trace(go.Scatter(x=curve["datetime"], y=curve["drawdown_pct"], mode="lines", name="Drawdown"))
    fig.update_layout(title="Drawdown", xaxis_title="Time", yaxis_title="Drawdown %")
    return fig


def trade_distribution_figure(trades_df: pd.DataFrame) -> go.Figure:
    """Build a trade PnL distribution figure."""

    fig = go.Figure()
    if trades_df is not None and not trades_df.empty:
        pnl_col = "pnl_amount" if "pnl_amount" in trades_df.columns else "pnl_points"
        fig.add_trace(go.Histogram(x=trades_df[pnl_col], nbinsx=30, name="Trade PnL"))
    fig.update_layout(title="Trade Distribution", xaxis_title="PnL", yaxis_title="Count")
    return fig


def save_figure_html(fig: go.Figure, path: str | Path) -> None:
    """Persist a Plotly figure as HTML."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(path))

