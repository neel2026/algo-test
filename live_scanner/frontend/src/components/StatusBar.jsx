export default function StatusBar({ connected, indicators, instrument, signalCount, state }) {
  const close = indicators.current_close ?? indicators.close
  const upper = indicators.bb_upper
  const lower = indicators.bb_lower
  let bbLabel = "N/A"
  if (close != null && upper != null && lower != null && upper !== lower) {
    if (close > upper) {
      bbLabel = "UPPER BAND"
    } else if (close < lower) {
      bbLabel = "LOWER BAND"
    } else {
      const percentB = Math.max(0, Math.min(100, ((close - lower) / (upper - lower)) * 100))
      bbLabel = `${percentB.toFixed(0)}%B`
    }
  }

  return (
    <div style={{ marginLeft: "auto", display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
      <span className="badge single" style={{ color: connected ? "#26a69a" : "#ef5350" }}>
        {connected ? "LIVE" : "OFFLINE"}
      </span>
      <span className="badge single">{instrument}</span>
      <span className="badge single">Signals: {signalCount}</span>
      <span className="badge single">RSI: {indicators.rsi != null ? indicators.rsi.toFixed(1) : "N/A"}</span>
      <span className="badge single">VIX: {indicators.vix != null ? indicators.vix.toFixed(1) : "N/A"}</span>
      <span className="badge single">BB: {bbLabel}</span>
      <span className="badge single">Interval: {state.interval ?? "N/A"}m</span>
    </div>
  )
}
