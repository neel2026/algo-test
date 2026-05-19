import { useEffect, useMemo, useState } from "react"

const API_BASE = import.meta.env.VITE_API_BASE_URL || "/api"

export default function InstrumentSelector({ value, onChange }) {
  const [instruments, setInstruments] = useState({ indices: [], stocks: [] })
  const [query, setQuery] = useState("")

  useEffect(() => {
    fetch(`${API_BASE}/instruments`)
      .then((response) => response.json())
      .then((data) => setInstruments(data))
      .catch(() => setInstruments({ indices: [], stocks: [] }))
  }, [])

  const filtered = useMemo(() => {
    const q = query.trim().toLowerCase()
    if (!q) return instruments

    const match = (item) => item.label.toLowerCase().includes(q) || item.key.toLowerCase().includes(q)
    return {
      indices: instruments.indices.filter(match),
      stocks: instruments.stocks.filter(match),
    }
  }, [instruments, query])

  return (
    <div style={{ display: "grid", gap: 6, minWidth: 260 }}>
      <input
        type="text"
        placeholder="Search instruments..."
        value={query}
        onChange={(event) => setQuery(event.target.value)}
        style={{ padding: "8px 10px" }}
      />
      <select value={value} onChange={(event) => onChange(event.target.value)} style={{ padding: "8px 10px" }}>
        <optgroup label="Indices">
          {filtered.indices.map((item) => (
            <option key={item.key} value={item.key}>
              {item.label}
            </option>
          ))}
        </optgroup>
        <optgroup label="F&O Stocks">
          {filtered.stocks.map((item) => (
            <option key={item.key} value={item.key}>
              {item.label}
            </option>
          ))}
        </optgroup>
      </select>
    </div>
  )
}
