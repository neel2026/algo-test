"""Combined HTML report builder for completed NIFTY backtests."""

from __future__ import annotations

import html
import json
import math
import sys
from datetime import datetime, time as dtime
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[1]


def _load_csv(path: Path) -> pd.DataFrame:
    """Load a CSV file if it exists, otherwise return an empty dataframe."""

    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def _require_files(result_dir: Path) -> None:
    """Ensure the required results files exist before generating a report."""

    required = ["trades.csv", "daily_pnl.csv", "signal_log.csv"]
    missing = [name for name in required if not (result_dir / name).exists()]
    if missing:
        print(f"❌ Results directory is incomplete: missing {', '.join(missing)}")
        sys.exit(1)


def _fmt_inr(value: float) -> str:
    """Format a number as Indian rupees text."""

    sign = "-" if value < 0 else ""
    value = abs(float(value))
    integer = int(value)
    decimals = f"{value:.2f}".split(".")[1]
    s = str(integer)
    if len(s) <= 3:
        grouped = s
    else:
        grouped = s[-3:]
        s = s[:-3]
        while s:
            grouped = f"{s[-2:]},{grouped}"
            s = s[:-2]
    return f"{sign}₹{grouped}.{decimals}"


def _trade_frame(trades: pd.DataFrame) -> pd.DataFrame:
    """Normalize trade dataframe types."""

    if trades.empty:
        return trades.copy()
    df = trades.copy()
    for column in ["entry_time", "partial_exit_time", "full_exit_time", "exit_time"]:
        if column in df.columns:
            df[column] = pd.to_datetime(df[column], errors="coerce")
    df = df.sort_values("entry_time").reset_index(drop=True)
    if "total_pnl" not in df.columns and "pnl_amount" in df.columns:
        df["total_pnl"] = df["pnl_amount"]
    return df


def _daily_curve(daily_pnl: pd.DataFrame, capital: float) -> pd.DataFrame:
    """Normalize daily PnL into equity and drawdown values."""

    if daily_pnl.empty:
        return daily_pnl.copy()
    df = daily_pnl.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    if "daily_pnl" not in df.columns:
        df["daily_pnl"] = 0.0
    if "cumulative_pnl" not in df.columns:
        df["cumulative_pnl"] = df["daily_pnl"].cumsum()
    df["equity"] = capital + df["cumulative_pnl"]
    if "drawdown" not in df.columns:
        peak = df["equity"].cummax()
        df["drawdown"] = peak - df["equity"]
    df["drawdown_pct"] = (df["drawdown"] / capital) * 100.0 if capital else 0.0
    return df


def _summary_metrics(trades: pd.DataFrame, daily_curve: pd.DataFrame, capital: float) -> dict[str, Any]:
    """Compute core metrics used in the report header."""

    if trades.empty:
        return {
            "net_pnl": 0.0,
            "win_rate_pct": 0.0,
            "profit_factor": 0.0,
            "max_dd_pct": 0.0,
            "sharpe_ratio": 0.0,
        }
    pnl_col = "total_pnl" if "total_pnl" in trades.columns else "pnl_amount"
    wins = trades[trades[pnl_col] > 0]
    losses = trades[trades[pnl_col] < 0]
    gross_profit = float(wins[pnl_col].sum())
    gross_loss = float(abs(losses[pnl_col].sum()))
    profit_factor = float("inf") if gross_loss == 0 and gross_profit > 0 else (gross_profit / gross_loss if gross_loss else 0.0)
    win_rate_pct = float(len(wins) / len(trades) * 100.0)
    net_pnl = float(trades[pnl_col].sum())
    sharpe = 0.0
    max_dd_pct = 0.0
    if not daily_curve.empty:
        returns = daily_curve["equity"].pct_change().fillna(0.0)
        volatility = float(returns.std(ddof=0))
        sharpe = float((returns.mean() / volatility) * math.sqrt(252)) if volatility else 0.0
        max_dd_pct = float((daily_curve["drawdown"].max() / capital) * 100.0) if capital else 0.0
    return {
        "net_pnl": net_pnl,
        "win_rate_pct": win_rate_pct,
        "profit_factor": profit_factor,
        "max_dd_pct": max_dd_pct,
        "sharpe_ratio": sharpe,
    }


def _monthly_heatmap(trades: pd.DataFrame) -> go.Figure:
    """Build a 12-column monthly PnL heatmap."""

    if trades.empty:
        fig = go.Figure()
        fig.update_layout(title="Monthly PnL Heatmap")
        return fig
    df = trades.copy()
    df["exit_time"] = pd.to_datetime(df["full_exit_time"].fillna(df["exit_time"]), errors="coerce")
    df = df.dropna(subset=["exit_time"])
    pnl_col = "total_pnl" if "total_pnl" in df.columns else "pnl_amount"
    df["year"] = df["exit_time"].dt.year
    df["month"] = df["exit_time"].dt.month
    pivot = df.groupby(["year", "month"], as_index=False)[pnl_col].sum().pivot(index="year", columns="month", values=pnl_col).fillna(0.0)
    all_months = list(range(1, 13))
    pivot = pivot.reindex(columns=all_months, fill_value=0.0)
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot.values,
            x=month_labels,
            y=[str(year) for year in pivot.index.tolist()],
            colorscale="RdYlGn",
            reversescale=False,
            hovertemplate="Year %{y}<br>Month %{x}<br>PnL %{z:.2f}<extra></extra>",
        )
    )
    fig.update_layout(title="Monthly PnL Heatmap", height=420)
    return fig


def _equity_figure(daily_curve: pd.DataFrame) -> go.Figure:
    """Build a combined equity and drawdown figure."""

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
    if not daily_curve.empty:
        fig.add_trace(go.Scatter(x=daily_curve["date"], y=daily_curve["equity"], mode="lines", name="Equity"), row=1, col=1)
        fig.add_trace(
            go.Scatter(
                x=daily_curve["date"],
                y=daily_curve["drawdown"],
                mode="lines",
                name="Drawdown",
                fill="tozeroy",
                line=dict(color="#ff6b6b"),
                fillcolor="rgba(255,107,107,0.25)",
            ),
            row=2,
            col=1,
        )
    fig.update_layout(title="Equity Curve and Drawdown", height=600, showlegend=True)
    fig.update_yaxes(title_text="Equity", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown", row=2, col=1)
    return fig


def _trade_distribution_figure(trades: pd.DataFrame) -> go.Figure:
    """Build a histogram of individual trade PnL values."""

    fig = go.Figure()
    if trades.empty:
        fig.update_layout(title="Trade Distribution")
        return fig
    pnl_col = "total_pnl" if "total_pnl" in trades.columns else "pnl_amount"
    wins = trades[trades[pnl_col] > 0]
    losses = trades[trades[pnl_col] < 0]
    avg_win = float(wins[pnl_col].mean()) if not wins.empty else 0.0
    avg_loss = float(losses[pnl_col].mean()) if not losses.empty else 0.0
    fig.add_trace(go.Histogram(x=trades[pnl_col], nbinsx=30, name="Trade PnL", marker_color="#4ec9b0"))
    for value, label, color in [(avg_win, "Avg Win", "#22c55e"), (avg_loss, "Avg Loss", "#ef4444"), (0.0, "Breakeven", "#ffffff")]:
        fig.add_vline(x=value, line_dash="dash", line_color=color, annotation_text=label, annotation_position="top")
    fig.update_layout(title="Trade Distribution", xaxis_title="PnL", yaxis_title="Count", height=420, bargap=0.05)
    return fig


def _signal_quality_table(trades: pd.DataFrame) -> pd.DataFrame:
    """Compute signal quality metrics by strength."""

    if trades.empty or "strength" not in trades.columns:
        return pd.DataFrame(columns=["Strength", "Count", "Win Rate", "Avg PnL", "Avg Winner", "Avg Loser", "Profit Factor"])
    df = trades.copy()
    pnl_col = "total_pnl" if "total_pnl" in df.columns else "pnl_amount"
    rows = []
    for strength, group in df.groupby(df["strength"].fillna("NA")):
        winners = group[group[pnl_col] > 0]
        losers = group[group[pnl_col] < 0]
        gross_profit = float(winners[pnl_col].sum())
        gross_loss = float(abs(losers[pnl_col].sum()))
        profit_factor = float("inf") if gross_loss == 0 and gross_profit > 0 else (gross_profit / gross_loss if gross_loss else 0.0)
        rows.append(
            [
                strength,
                int(len(group)),
                round((len(winners) / len(group) * 100.0) if len(group) else 0.0, 1),
                _fmt_inr(float(group[pnl_col].mean())),
                _fmt_inr(float(winners[pnl_col].mean()) if not winners.empty else 0.0),
                _fmt_inr(float(losers[pnl_col].mean()) if not losers.empty else 0.0),
                f"{profit_factor:.2f}" if math.isfinite(profit_factor) else "inf",
            ]
        )
    return pd.DataFrame(rows, columns=["Strength", "Count", "Win Rate", "Avg PnL", "Avg Winner", "Avg Loser", "Profit Factor"])


def _exit_reason_pie(trades: pd.DataFrame) -> go.Figure:
    """Build a pie chart of exit reasons."""

    fig = go.Figure()
    if trades.empty or "exit_reason" not in trades.columns:
        fig.update_layout(title="Exit Reason Breakdown")
        return fig
    counts = trades["exit_reason"].fillna("NA").value_counts()
    fig.add_trace(go.Pie(labels=counts.index.tolist(), values=counts.values.tolist(), hole=0.35))
    fig.update_layout(title="Exit Reason Breakdown", height=420)
    return fig


def _time_bucket(value: pd.Timestamp) -> str:
    """Bucket an entry timestamp into the requested windows."""

    t = value.time()
    if dtime(9, 20) <= t < dtime(10, 0):
        return "09:20-10:00"
    if dtime(10, 0) <= t < dtime(11, 0):
        return "10:00-11:00"
    if dtime(11, 0) <= t < dtime(12, 0):
        return "11:00-12:00"
    if dtime(12, 0) <= t <= dtime(13, 0):
        return "12:00-13:00"
    return "Other"


def _time_of_day_chart(trades: pd.DataFrame) -> tuple[pd.DataFrame, go.Figure]:
    """Build a time-of-day analysis table and chart."""

    if trades.empty:
        empty = pd.DataFrame(columns=["Entry Window", "Trade Count", "Win Rate", "Avg PnL"])
        return empty, go.Figure()
    df = trades.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
    df = df.dropna(subset=["entry_time"])
    df["bucket"] = df["entry_time"].apply(_time_bucket)
    pnl_col = "total_pnl" if "total_pnl" in df.columns else "pnl_amount"
    buckets = ["09:20-10:00", "10:00-11:00", "11:00-12:00", "12:00-13:00"]
    rows = []
    win_rates = []
    for bucket in buckets:
        group = df[df["bucket"] == bucket]
        winners = group[group[pnl_col] > 0]
        win_rate = round((len(winners) / len(group) * 100.0) if len(group) else 0.0, 1)
        rows.append([bucket, int(len(group)), win_rate, _fmt_inr(float(group[pnl_col].mean()) if not group.empty else 0.0)])
        win_rates.append(win_rate)
    fig = go.Figure(go.Bar(x=buckets, y=win_rates, marker_color="#38bdf8"))
    fig.update_layout(title="Win Rate by Entry Window", yaxis_title="Win Rate %", height=420)
    return pd.DataFrame(rows, columns=["Entry Window", "Trade Count", "Win Rate", "Avg PnL"]), fig


def _render_summary_cards(metrics: dict[str, Any], daily_curve: pd.DataFrame, capital: float) -> str:
    """Render the KPI summary block."""

    net_pnl = metrics["net_pnl"]
    return f"""
    <div class="summary-grid">
      <div class="kpi"><div class="label">Net PnL</div><div class="value">{_fmt_inr(net_pnl)}</div></div>
      <div class="kpi"><div class="label">Win Rate</div><div class="value">{metrics['win_rate_pct']:.1f}%</div></div>
      <div class="kpi"><div class="label">Profit Factor</div><div class="value">{metrics['profit_factor']:.2f}</div></div>
      <div class="kpi"><div class="label">Max DD</div><div class="value">{_fmt_inr(float(daily_curve['drawdown'].max()) if not daily_curve.empty else 0.0)}</div></div>
      <div class="kpi"><div class="label">Sharpe</div><div class="value">{metrics['sharpe_ratio']:.2f}</div></div>
    </div>
    """


def _table_html(df: pd.DataFrame, class_name: str = "") -> str:
    """Render a dataframe as an HTML table."""

    if df.empty:
        return "<p class='empty'>No data available.</p>"
    return df.to_html(index=False, classes=f"report-table {class_name}".strip(), escape=False)


def _trade_table(trades: pd.DataFrame) -> str:
    """Render the full trade log as a sortable HTML table."""

    if trades.empty:
        return "<p class='empty'>No trades available.</p>"
    df = trades.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"], errors="coerce")
    df = df.sort_values("entry_time").reset_index(drop=True)
    display = df.copy()
    for column in ["entry_time", "partial_exit_time", "full_exit_time", "exit_time"]:
        if column in display.columns:
            display[column] = pd.to_datetime(display[column], errors="coerce").dt.strftime("%Y-%m-%d %H:%M:%S")
    numeric_cols = [column for column in ["total_pnl", "partial_pnl", "final_pnl", "entry_option_price", "partial_exit_price", "full_exit_price"] if column in display.columns]
    for column in numeric_cols:
        display[column] = display[column].apply(lambda value: f"{float(value):.2f}" if pd.notna(value) else "")
    rows_html = []
    for _, row in display.iterrows():
        pnl = float(row.get("total_pnl", 0.0) or 0.0)
        row_class = "profit" if pnl > 0 else "loss" if pnl < 0 else "flat"
        cells = "".join(f"<td>{html.escape(str(value))}</td>" for value in row.tolist())
        rows_html.append(f"<tr class='{row_class}'>{cells}</tr>")
    headers = "".join(f"<th onclick='sortTable({i})'>{html.escape(str(col))}</th>" for i, col in enumerate(display.columns.tolist()))
    return f"""
    <table id="tradeLog" class="trade-table">
      <thead><tr>{headers}</tr></thead>
      <tbody>
        {''.join(rows_html)}
      </tbody>
    </table>
    """


def _html_shell(title: str, body: str) -> str:
    """Wrap the report body in a full HTML document."""

    return f"""
    <!doctype html>
    <html lang="en">
    <head>
      <meta charset="utf-8" />
      <meta name="viewport" content="width=device-width, initial-scale=1" />
      <title>{html.escape(title)}</title>
      <style>
        :root {{
          --bg: #0b1220;
          --panel: #11192b;
          --panel-2: #162036;
          --text: #e5eefb;
          --muted: #9fb0cc;
          --green: #22c55e;
          --red: #ef4444;
          --blue: #38bdf8;
          --border: rgba(255,255,255,0.08);
        }}
        body {{
          margin: 0;
          background: radial-gradient(circle at top, #16213a 0%, var(--bg) 55%);
          color: var(--text);
          font-family: Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        }}
        .container {{
          max-width: 1400px;
          margin: 0 auto;
          padding: 32px 20px 60px;
        }}
        .header {{
          display: grid;
          gap: 8px;
          margin-bottom: 24px;
        }}
        .header h1 {{
          margin: 0;
          font-size: 2rem;
        }}
        .header .subtle {{
          color: var(--muted);
        }}
        .summary-grid {{
          display: grid;
          grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
          gap: 12px;
          margin: 16px 0 24px;
        }}
        .kpi {{
          background: rgba(17,25,43,0.92);
          border: 1px solid var(--border);
          border-radius: 16px;
          padding: 16px;
          box-shadow: 0 12px 30px rgba(0,0,0,0.18);
        }}
        .kpi .label {{
          color: var(--muted);
          font-size: 0.85rem;
          margin-bottom: 8px;
        }}
        .kpi .value {{
          font-size: 1.6rem;
          font-weight: 700;
        }}
        .panel {{
          background: rgba(17,25,43,0.92);
          border: 1px solid var(--border);
          border-radius: 18px;
          padding: 18px;
          margin-bottom: 20px;
        }}
        .panel h2 {{
          margin: 0 0 14px;
          font-size: 1.15rem;
        }}
        .report-table {{
          width: 100%;
          border-collapse: collapse;
          font-size: 0.92rem;
        }}
        .report-table th, .report-table td {{
          border-bottom: 1px solid var(--border);
          padding: 10px 8px;
          text-align: left;
        }}
        .report-table th {{
          cursor: pointer;
          color: #cfe3ff;
          position: sticky;
          top: 0;
          background: rgba(17,25,43,0.98);
        }}
        .report-table tr:nth-child(even) {{
          background: rgba(255,255,255,0.02);
        }}
        .trade-table tr.profit {{
          background: rgba(34,197,94,0.08);
        }}
        .trade-table tr.loss {{
          background: rgba(239,68,68,0.08);
        }}
        .trade-table tr.flat {{
          background: rgba(255,255,255,0.02);
        }}
        .empty {{
          color: var(--muted);
          font-style: italic;
        }}
        .chart {{
          margin-top: 8px;
        }}
        .two-col {{
          display: grid;
          grid-template-columns: 1fr 1fr;
          gap: 18px;
        }}
        @media (max-width: 1100px) {{
          .two-col {{
            grid-template-columns: 1fr;
          }}
        }}
      </style>
      <script>
        function sortTable(colIndex) {{
          const table = document.getElementById("tradeLog");
          const tbody = table.tBodies[0];
          const rows = Array.from(tbody.rows);
          const asc = table.getAttribute("data-sort-dir") !== "asc";
          rows.sort(function(a, b) {{
            const aText = a.cells[colIndex].innerText;
            const bText = b.cells[colIndex].innerText;
            const aNum = parseFloat(aText.replace(/[^0-9.-]/g, ""));
            const bNum = parseFloat(bText.replace(/[^0-9.-]/g, ""));
            const bothNumeric = !Number.isNaN(aNum) && !Number.isNaN(bNum);
            let cmp = bothNumeric ? aNum - bNum : aText.localeCompare(bText);
            return asc ? cmp : -cmp;
          }});
          rows.forEach(r => tbody.appendChild(r));
          table.setAttribute("data-sort-dir", asc ? "asc" : "desc");
        }}
      </script>
    </head>
    <body>
      <div class="container">
        {body}
      </div>
    </body>
    </html>
    """


def generate_report(result_dir: str | Path) -> Path:
    """Generate a single-file HTML report from a results directory."""

    result_dir = Path(result_dir).expanduser().resolve()
    _require_files(result_dir)

    trades = _trade_frame(_load_csv(result_dir / "trades.csv"))
    daily = _daily_curve(_load_csv(result_dir / "daily_pnl.csv"), capital=500000.0)
    capital = 500000.0
    run_config: dict[str, Any] = {}
    if (result_dir / "run_config.json").exists():
        try:
            run_config = json.loads((result_dir / "run_config.json").read_text(encoding="utf-8"))
            capital = float(run_config.get("capital", capital))
            daily = _daily_curve(_load_csv(result_dir / "daily_pnl.csv"), capital=capital)
        except Exception:
            pass

    metrics = _summary_metrics(trades, daily, capital)
    report_dir = result_dir
    report_path = report_dir / "report.html"

    if trades.empty or daily.empty:
        print(f"❌ Results directory is empty or incomplete: {result_dir}")
        sys.exit(1)

    monthly_fig = _monthly_heatmap(trades)
    equity_fig = _equity_figure(daily)
    dist_fig = _trade_distribution_figure(trades)
    exit_fig = _exit_reason_pie(trades)
    tod_table, tod_fig = _time_of_day_chart(trades)
    signal_table = _signal_quality_table(trades)

    header = f"""
    <div class="header">
      <h1>NIFTY ATM Backtest Report</h1>
      <div class="subtle">Strategy: {html.escape(str(run_config.get("strategy", "Unknown")))}</div>
      <div class="subtle">Period: {html.escape(str(run_config.get("start", "?")))} → {html.escape(str(run_config.get("end", "?")))}</div>
      <div class="subtle">Capital: {_fmt_inr(capital)} | Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</div>
    </div>
    """

    summary = _render_summary_cards(metrics, daily, capital)
    charts = f"""
    <div class="panel">
      <h2>Equity Curve</h2>
      <div class="chart">{equity_fig.to_html(full_html=False, include_plotlyjs=True)}</div>
    </div>
    <div class="panel">
      <h2>Monthly PnL Heatmap</h2>
      <div class="chart">{monthly_fig.to_html(full_html=False, include_plotlyjs=False)}</div>
    </div>
    <div class="panel">
      <h2>Trade Distribution</h2>
      <div class="chart">{dist_fig.to_html(full_html=False, include_plotlyjs=False)}</div>
    </div>
    <div class="two-col">
      <div class="panel">
        <h2>Signal Quality</h2>
        {_table_html(signal_table)}
      </div>
      <div class="panel">
        <h2>Exit Reason Breakdown</h2>
        <div class="chart">{exit_fig.to_html(full_html=False, include_plotlyjs=False)}</div>
      </div>
    </div>
    <div class="panel">
      <h2>Time-of-Day Performance</h2>
      <div class="chart">{tod_fig.to_html(full_html=False, include_plotlyjs=False)}</div>
      {_table_html(tod_table)}
    </div>
    <div class="panel">
      <h2>Full Trade Log</h2>
      {_trade_table(trades)}
    </div>
    """

    html_text = _html_shell("NIFTY Backtest Report", header + summary + charts)
    report_path.write_text(html_text, encoding="utf-8")
    print(f"📊 Report saved: {report_path}")
    print(f"   Open in browser: file://{report_path.resolve()}")
    return report_path


def main() -> None:
    """CLI entry point for generating a backtest report."""

    import argparse

    parser = argparse.ArgumentParser(description="Generate HTML report from a backtest results directory.")
    parser.add_argument("--dir", required=True, help="Path to the results directory")
    args = parser.parse_args()
    generate_report(args.dir)


if __name__ == "__main__":
    main()
