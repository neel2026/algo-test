import { useEffect, useMemo, useState } from "react"
import Chart from "./components/Chart"
import SignalPanel from "./components/SignalPanel"
import InstrumentSelector from "./components/InstrumentSelector"
import IntervalSelector from "./components/IntervalSelector"
import StatusBar from "./components/StatusBar"
import { useSSE } from "./hooks/useSSE"
import "./App.css"

const API_BASE = import.meta.env.VITE_API_BASE_URL || "/api"

export default function App() {
  const [authenticated, setAuthenticated] = useState(false)
  const [instrument, setInstrument] = useState("NSE_INDEX|Nifty 50")
  const [interval, setInterval] = useState(5)
  const [latestState, setLatestState] = useState({})
  const [historyCandles, setHistoryCandles] = useState([])
  const [historySignals, setHistorySignals] = useState([])
  const [historyIndicators, setHistoryIndicators] = useState({})
  const [historyMeta, setHistoryMeta] = useState({})
  const [historyLoading, setHistoryLoading] = useState(false)
  const { candles, ticks, signals, indicators, connected } = useSSE()
  const historyDays = 2

  useEffect(() => {
    fetch(`${API_BASE}/auth/status`)
      .then((response) => response.json())
      .then((data) => {
        if (data.authenticated) {
          setAuthenticated(true)
          return fetch(`${API_BASE}/state`)
            .then((response) => response.json())
            .then((state) => {
              setLatestState(state)
              if (state.current_instrument) setInstrument(state.current_instrument)
              if (state.interval) setInterval(state.interval)
            })
        }
        window.location.href = `${API_BASE}/auth/login`
        return null
      })
      .catch(() => {
        window.location.href = `${API_BASE}/auth/login`
      })
  }, [])

  useEffect(() => {
    if (!authenticated) return

    const syncState = () => {
      fetch(`${API_BASE}/state`)
        .then((response) => response.json())
        .then((scannerState) => setLatestState(scannerState))
        .catch(() => {})
    }

    syncState()
    const timer = window.setInterval(syncState, 5000)
    return () => window.clearInterval(timer)
  }, [authenticated])

  useEffect(() => {
    if (!authenticated) return

    const controller = new AbortController()
    setHistoryLoading(true)
    setHistoryMeta((previous) => ({
      ...previous,
      status: "loading",
      detail: `Loading ${historyDays} day(s) of ${interval}m candles for ${instrument}...`,
    }))
    fetch(
      `${API_BASE}/history/${encodeURIComponent(instrument)}?interval=${interval}&days=${historyDays}`,
      { signal: controller.signal },
    )
      .then((response) => response.json())
      .then((payload) => {
        setHistoryCandles(payload.candles ?? [])
        setHistorySignals(payload.signals ?? [])
        setHistoryIndicators(payload.indicators ?? {})
        setHistoryMeta(payload.meta ?? {})
        setHistoryLoading(false)
      })
      .catch((error) => {
        if (error?.name === "AbortError") return
        setHistoryCandles([])
        setHistorySignals([])
        setHistoryIndicators({})
        setHistoryMeta({
          status: "error",
          source: "frontend_fetch",
          reason: "request_failed",
          detail: "Frontend could not load the history endpoint.",
          candle_count: 0,
          signal_count: 0,
        })
        setHistoryLoading(false)
      })

    return () => {
      controller.abort()
    }
  }, [authenticated, instrument, interval, historyDays])

  const displayIndicators = useMemo(() => {
    const merged = {
      ...historyIndicators,
      ...latestState.latest_indicators,
      ...indicators,
    }
    if (merged.vix == null && latestState.current_vix != null) {
      merged.vix = latestState.current_vix
    }
    return merged
  }, [historyIndicators, latestState, indicators])

  const selectedSignals = useMemo(() => {
    const merged = [...historySignals, ...signals]
    const seen = new Set()
    return merged
      .filter((signal) => signal && (!signal.instrument || signal.instrument === instrument))
      .filter((signal) => {
        if (!signal.signal_id) return true
        if (seen.has(signal.signal_id)) return false
        seen.add(signal.signal_id)
        return true
      })
      .sort((a, b) => (a.candle_time_ts ?? 0) - (b.candle_time_ts ?? 0))
  }, [historySignals, signals, instrument])

  const handleInstrumentChange = (key) => {
    setInstrument(key)
    fetch(`${API_BASE}/settings/instrument?instrument_key=${encodeURIComponent(key)}`, { method: "POST" })
  }

  const handleIntervalChange = (minutes) => {
    setInterval(minutes)
    fetch(`${API_BASE}/settings/interval?minutes=${minutes}`, { method: "POST" })
  }

  if (!authenticated) {
    return <div className="loading">Connecting to Upstox...</div>
  }

  return (
    <div className="app">
      <header className="toolbar">
        <span className="logo">NIFTY Signal Scanner</span>
        <InstrumentSelector value={instrument} onChange={handleInstrumentChange} />
        <IntervalSelector value={interval} onChange={handleIntervalChange} />
        <StatusBar
          connected={connected}
          indicators={displayIndicators}
          instrument={instrument}
          signalCount={selectedSignals.length}
          state={latestState}
        />
      </header>

      <main className="main-layout">
        <div className="chart-container">
          <div className={`history-banner ${historyLoading ? "loading" : historyMeta.status || "empty"}`}>
            <span className="history-banner-title">
              History: {historyLoading ? "loading" : historyMeta.status || "unknown"}
            </span>
            <span>
              {historyMeta.candle_count ?? 0} candles, {historyMeta.signal_count ?? 0} signals
            </span>
            <span>{historyMeta.detail || "No history diagnostics yet."}</span>
          </div>
          <Chart
            key={`${instrument}-${interval}`}
            instrument={instrument}
            interval={interval}
            candles={historyCandles}
            newCandle={candles[candles.length - 1]}
            latestTick={ticks[ticks.length - 1]}
            signals={selectedSignals}
            historyMeta={historyMeta}
            historyLoading={historyLoading}
          />
        </div>
        <div className="signal-panel">
          <SignalPanel signals={selectedSignals} indicators={displayIndicators} />
        </div>
      </main>
    </div>
  )
}
