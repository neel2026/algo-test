import { useEffect, useRef } from "react"
import { CrosshairMode, createChart } from "lightweight-charts"

export default function Chart({ instrument, interval, candles, newCandle, latestTick, signals, historyMeta, historyLoading }) {
  const containerRef = useRef(null)
  const chartRef = useRef(null)
  const seriesRef = useRef(null)
  const seenSignalsRef = useRef(new Set())
  const markerListRef = useRef([])

  useEffect(() => {
    const chart = createChart(containerRef.current, {
      layout: {
        background: { color: "#0f0f0f" },
        textColor: "#d1d4dc",
      },
      grid: {
        vertLines: { color: "#1e1e2e" },
        horzLines: { color: "#1e1e2e" },
      },
      crosshair: { mode: CrosshairMode.Normal },
      rightPriceScale: { borderColor: "#2a2a3a" },
      timeScale: {
        borderColor: "#2a2a3a",
        timeVisible: true,
        secondsVisible: false,
      },
      width: containerRef.current.clientWidth,
      height: containerRef.current.clientHeight,
    })

    const candleSeries = chart.addCandlestickSeries({
      upColor: "#26a69a",
      downColor: "#ef5350",
      borderUpColor: "#26a69a",
      borderDownColor: "#ef5350",
      wickUpColor: "#26a69a",
      wickDownColor: "#ef5350",
    })

    chartRef.current = chart
    seriesRef.current = candleSeries
    seenSignalsRef.current = new Set()
    markerListRef.current = []

    const observer = new ResizeObserver(() => {
      chart.applyOptions({
        width: containerRef.current.clientWidth,
        height: containerRef.current.clientHeight,
      })
    })
    observer.observe(containerRef.current)

    return () => {
      observer.disconnect()
      chart.remove()
    }
  }, [instrument, interval])

  useEffect(() => {
    if (seriesRef.current && candles?.length) {
      seriesRef.current.setData(candles)
      chartRef.current?.timeScale().fitContent()
      seenSignalsRef.current = new Set()
      markerListRef.current = []
    }
  }, [candles])

  useEffect(() => {
    if (newCandle && seriesRef.current) {
      seriesRef.current.update({
        time: newCandle.time,
        open: newCandle.open,
        high: newCandle.high,
        low: newCandle.low,
        close: newCandle.close,
      })
    }
  }, [newCandle])

  useEffect(() => {
    if (latestTick && seriesRef.current && latestTick.candle_time_ts) {
      seriesRef.current.update({
        time: latestTick.candle_time_ts,
        close: latestTick.ltp,
      })
    }
  }, [latestTick])

  useEffect(() => {
    if (!signals.length || !seriesRef.current) return

    const newMarkers = []
    for (const signal of signals) {
      if (!signal?.signal_id || seenSignalsRef.current.has(signal.signal_id)) continue
      seenSignalsRef.current.add(signal.signal_id)
      newMarkers.push({
        time: signal.candle_time_ts,
        position: signal.action === "BUY_CE" ? "belowBar" : "aboveBar",
        color:
          signal.action === "BUY_CE"
            ? signal.tradeable
              ? "#26a69a"
              : "#748194"
            : signal.tradeable
              ? "#ef5350"
              : "#748194",
        shape: signal.action === "BUY_CE" ? "arrowUp" : "arrowDown",
        text: `${signal.action === "BUY_CE" ? "CE" : "PE"} ${signal.atm_strike}${signal.strength === "full" ? " FULL" : ""}`,
      })
    }

    if (!newMarkers.length) return
    markerListRef.current = [...markerListRef.current, ...newMarkers]
    seriesRef.current.setMarkers(markerListRef.current)
  }, [signals])

  return (
    <div className="chart-shell">
      {historyLoading ? (
        <div className="chart-loading-state">
          <div className="chart-spinner" />
          <div className="chart-loading-title">Loading candles</div>
          <div className="chart-loading-detail">
            Fetching historical data for {instrument} on the {interval}m interval.
          </div>
        </div>
      ) : null}
      {!candles?.length ? (
        <div className="chart-empty-state">
          <div className="chart-empty-title">No candles loaded</div>
          <div className="chart-empty-detail">
            {historyMeta?.detail || "The backend returned zero historical candles for this instrument."}
          </div>
        </div>
      ) : null}
      <div ref={containerRef} style={{ width: "100%", height: "100%", minHeight: "500px" }} />
    </div>
  )
}
