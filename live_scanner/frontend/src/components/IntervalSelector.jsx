const intervals = [1, 3, 5, 15, 30]

export default function IntervalSelector({ value, onChange }) {
  return (
    <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
      {intervals.map((minutes) => (
        <button
          key={minutes}
          onClick={() => onChange(minutes)}
          style={{
            padding: "8px 12px",
            borderRadius: 10,
            background: minutes === value ? "#26a69a" : "rgba(255,255,255,0.04)",
            color: minutes === value ? "#04110f" : "#d1d4dc",
            fontWeight: 700,
          }}
        >
          {minutes}m
        </button>
      ))}
    </div>
  )
}
