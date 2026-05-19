import { useEffect, useState } from "react"

const API_BASE = import.meta.env.VITE_API_BASE_URL || "/api"

function playBeep() {
  try {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext
    if (!AudioContextClass) return
    const audioContext = new AudioContextClass()
    const oscillator = audioContext.createOscillator()
    const gain = audioContext.createGain()
    oscillator.type = "sine"
    oscillator.frequency.value = 880
    gain.gain.value = 0.05
    oscillator.connect(gain)
    gain.connect(audioContext.destination)
    oscillator.start()
    oscillator.stop(audioContext.currentTime + 0.12)
  } catch {
    // Ignore audio errors in the browser.
  }
}

export function useSSE() {
  const [candles, setCandles] = useState([])
  const [ticks, setTicks] = useState([])
  const [signals, setSignals] = useState([])
  const [indicators, setIndicators] = useState({})
  const [connected, setConnected] = useState(false)

  useEffect(() => {
    const eventSource = new EventSource(`${API_BASE}/stream`)

    eventSource.onopen = () => setConnected(true)
    eventSource.onerror = () => setConnected(false)

    eventSource.onmessage = (event) => {
      const payload = JSON.parse(event.data)
      switch (payload.type) {
        case "status":
        case "heartbeat":
          setConnected(Boolean(payload.data?.connected ?? true))
          break
        case "candle":
          setCandles((previous) => [...previous.slice(-500), payload.data])
          break
        case "tick":
          setTicks((previous) => [...previous.slice(-20), payload.data])
          break
        case "signal":
          setSignals((previous) => [...previous.slice(-50), payload.data])
          playBeep()
          break
        case "indicators":
          setIndicators(payload.data)
          break
        default:
          break
      }
    }

    return () => eventSource.close()
  }, [])

  return { candles, ticks, signals, indicators, connected }
}
