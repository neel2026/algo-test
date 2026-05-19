export default function StatusBar({ connected, indicators, instrument, signalCount, state }) {
  return (
    <div style={{ marginLeft: "auto", display: "flex", gap: 10, alignItems: "center", flexWrap: "wrap" }}>
      <span className="badge single" style={{ color: connected ? "#26a69a" : "#ef5350" }}>
        {connected ? "LIVE" : "OFFLINE"}
      </span>
      <span className="badge single">{instrument}</span>
      <span className="badge single">Signals: {signalCount}</span>
      <span className="badge single">RSI: {indicators.rsi != null ? indicators.rsi.toFixed(1) : "N/A"}</span>
      <span className="badge single">VIX: {indicators.vix != null ? indicators.vix.toFixed(1) : "N/A"}</span>
      <span className="badge single">Interval: {state.interval ?? "N/A"}m</span>
    </div>
  )
}
