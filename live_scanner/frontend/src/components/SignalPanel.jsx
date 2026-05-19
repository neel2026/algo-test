export default function SignalPanel({ signals, indicators }) {
  const latestSignal = signals.length ? signals[signals.length - 1] : null
  const close = indicators.current_close ?? indicators.close
  const upper = indicators.bb_upper
  const lower = indicators.bb_lower
  const bbPosition =
    close != null && upper != null && lower != null && upper !== lower
      ? Math.max(0, Math.min(100, ((close - lower) / (upper - lower)) * 100))
      : null
  let bbPositionLabel = "N/A"
  if (close != null && upper != null && lower != null) {
    if (close > upper) {
      bbPositionLabel = "UPPER BAND"
    } else if (close < lower) {
      bbPositionLabel = "LOWER BAND"
    } else if (bbPosition != null) {
      bbPositionLabel = `${bbPosition.toFixed(1)}%`
    }
  }
  const rsi = indicators.rsi
  const rsiClass = rsi == null ? "neutral" : rsi >= 50 ? "positive" : "negative"

  return (
    <div>
      <div className="panel-section">
        <div className="panel-title">Indicator Summary</div>
        <div className="indicator-grid">
          <div className="indicator-pill">
            <div className="label">RSI(9)</div>
            <div className={`value ${rsiClass}`}>{rsi != null ? rsi.toFixed(2) : "N/A"}</div>
          </div>
          <div className="indicator-pill">
            <div className="label">VIX</div>
            <div className="value">{indicators.vix != null ? indicators.vix.toFixed(2) : "N/A"}</div>
          </div>
          <div className="indicator-pill">
            <div className="label">BB Position</div>
            <div className="value">{bbPositionLabel}</div>
          </div>
        </div>
        <div className="bb-bar" title="Position between Bollinger Bands">
          {bbPosition != null ? <div className="bb-marker" style={{ left: `${bbPosition}%` }} /> : null}
        </div>
      </div>

      <div className="panel-section">
        <div className="panel-title">Live Signals</div>
        {!signals.length ? (
          <div style={{ color: "#8aa0b7", fontSize: "0.9rem" }}>Waiting for a setup...</div>
        ) : (
          signals
            .slice(-12)
            .reverse()
            .map((signal, index) => (
              <div
                key={signal.signal_id}
                className={`signal-card ${signal.action === "BUY_CE" ? "ce" : "pe"} ${index === 0 ? "latest" : ""}`}
              >
                <div className="headline">
                  <div className="action">
                    {signal.action} {signal.atm_strike}
                  </div>
                  <span className={`badge ${signal.strength}`}>{signal.strength}</span>
                </div>
                <div className="meta">
                  <div>Time: {new Date(signal.candle_time_ts * 1000).toLocaleTimeString("en-IN", { hour12: false })}</div>
                  <div>Target 1: {signal.target1_spot?.toFixed?.(2) ?? signal.target1_spot}</div>
                  <div>Target 2: {signal.target2_spot?.toFixed?.(2) ?? signal.target2_spot}</div>
                  <div>Stop Loss: {signal.stoploss_spot?.toFixed?.(2) ?? signal.stoploss_spot}</div>
                  <div>VIX: {signal.vix_at_entry != null ? signal.vix_at_entry.toFixed(2) : "N/A"} | Qty Multiplier: {signal.qty_multiplier}</div>
                  <div>Entry Window: {signal.entry_window_open ? "OPEN" : "CLOSED"}</div>
                </div>
              </div>
            ))
        )}
      </div>
    </div>
  )
}
