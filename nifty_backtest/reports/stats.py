"""Performance statistics for backtest results."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _safe_div(numerator: float, denominator: float) -> float:
    """Divide two numbers safely."""

    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_statistics(trades_df: pd.DataFrame, equity_curve: pd.DataFrame | None = None, starting_capital: float = 0.0) -> dict:
    """Compute a standard set of trading performance metrics."""

    if trades_df is None or trades_df.empty:
        return {
            "total_trades": 0,
            "win_rate_pct": 0.0,
            "loss_rate_pct": 0.0,
            "avg_win_pct": 0.0,
            "avg_loss_pct": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_pct": 0.0,
            "best_trade": None,
            "worst_trade": None,
            "sharpe_ratio": 0.0,
        }

    df = trades_df.copy()
    pnl_col = "pnl_amount" if "pnl_amount" in df.columns else "pnl_points"
    return_col = "pnl_pct" if "pnl_pct" in df.columns else pnl_col
    wins = df[df[pnl_col] > 0]
    losses = df[df[pnl_col] < 0]
    gross_profit = wins[pnl_col].sum()
    gross_loss = abs(losses[pnl_col].sum())
    if gross_loss == 0 and gross_profit > 0:
        profit_factor = float("inf")
    else:
        profit_factor = float(_safe_div(gross_profit, gross_loss))

    if equity_curve is not None and not equity_curve.empty and "equity" in equity_curve.columns:
        curve = equity_curve.copy()
        curve["datetime"] = pd.to_datetime(curve["datetime"], errors="coerce")
        curve = curve.dropna(subset=["datetime"]).sort_values("datetime")
        curve["returns"] = curve["equity"].pct_change().fillna(0.0)
        volatility = curve["returns"].std(ddof=0)
        sharpe = _safe_div(curve["returns"].mean(), volatility) * np.sqrt(252) if volatility else 0.0
        peak = curve["equity"].cummax()
        drawdown = ((peak - curve["equity"]) / peak.replace(0, np.nan)) * 100.0
        max_drawdown = float(drawdown.max()) if not drawdown.empty else 0.0
    else:
        sharpe = 0.0
        max_drawdown = 0.0

    return {
        "total_trades": int(len(df)),
        "win_rate_pct": float((len(wins) / len(df)) * 100.0),
        "loss_rate_pct": float((len(losses) / len(df)) * 100.0),
        "avg_win_pct": float(wins[return_col].mean()) if not wins.empty else 0.0,
        "avg_loss_pct": float(losses[return_col].mean()) if not losses.empty else 0.0,
        "profit_factor": profit_factor,
        "max_drawdown_pct": float(max_drawdown),
        "best_trade": df.loc[df[pnl_col].idxmax()].to_dict() if not df.empty else None,
        "worst_trade": df.loc[df[pnl_col].idxmin()].to_dict() if not df.empty else None,
        "sharpe_ratio": float(sharpe),
    }


def monthly_pnl_breakdown(trades_df: pd.DataFrame) -> pd.DataFrame:
    """Group closed-trade PnL by calendar month."""

    if trades_df is None or trades_df.empty:
        return pd.DataFrame(columns=["month", "pnl_amount"])
    df = trades_df.copy()
    if "exit_time" not in df.columns:
        raise ValueError("Trades dataframe must contain exit_time.")
    df["exit_time"] = pd.to_datetime(df["exit_time"], errors="coerce")
    df = df.dropna(subset=["exit_time"])
    pnl_col = "pnl_amount" if "pnl_amount" in df.columns else "pnl_points"
    df["month"] = df["exit_time"].dt.to_period("M").astype(str)
    return df.groupby("month", as_index=False)[pnl_col].sum().rename(columns={pnl_col: "pnl_amount"})


def signal_hit_frequency(signals_df: pd.DataFrame) -> pd.DataFrame:
    """Count how often each strategy action and strength appears."""

    if signals_df is None or signals_df.empty:
        return pd.DataFrame(columns=["category", "value", "count"])

    records: list[dict] = []
    if "action" in signals_df.columns:
        action_counts = signals_df["action"].fillna("HOLD").value_counts().to_dict()
        records.extend({"category": "action", "value": key, "count": int(value)} for key, value in action_counts.items())
    if "strength" in signals_df.columns:
        strength_counts = signals_df["strength"].fillna("NA").value_counts().to_dict()
        records.extend({"category": "strength", "value": key, "count": int(value)} for key, value in strength_counts.items())
    return pd.DataFrame(records)
