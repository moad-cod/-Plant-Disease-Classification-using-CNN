function ConfidenceBar({ value, compact = false }) {
  const pct = Math.max(0, Math.min(100, Number(value || 0) * 100))

  return (
    <div className={compact ? 'mini-shell' : 'bar-shell'}>
      <div className={compact ? 'mini-fill' : 'bar-fill'} style={{ width: `${pct}%` }} />
    </div>
  )
}

export default ConfidenceBar
