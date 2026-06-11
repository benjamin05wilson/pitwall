import type { OptimizeResult } from '../api'

// Risk/reward scatter: expected finish (x) vs tail risk CVaR10 (y). Lower-left wins.
export default function Frontier({ res }: { res: OptimizeResult }) {
  const W = 460, H = 260, pad = 38
  const pts = res.frontier
  const xs = pts.map((p) => p.mean_pos), ys = pts.map((p) => p.cvar10)
  const xmin = Math.min(...xs) - 0.4, xmax = Math.max(...xs) + 0.4
  const ymin = Math.min(...ys) - 0.6, ymax = Math.max(...ys) + 0.6
  const fx = (x: number) => pad + ((x - xmin) / (xmax - xmin)) * (W - pad - 12)
  const fy = (y: number) => H - pad - ((y - ymin) / (ymax - ymin)) * (H - pad - 12)
  const opt = new Set(res.frontier_optimal)
  const bestLabel = res.best.label

  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%" style={{ overflow: 'visible' }}>
      {/* axes */}
      <line x1={pad} y1={H - pad} x2={W - 8} y2={H - pad} stroke="var(--border-bright)" />
      <line x1={pad} y1={12} x2={pad} y2={H - pad} stroke="var(--border-bright)" />
      <text x={(W) / 2} y={H - 6} fill="var(--dimmer)" fontSize="10" textAnchor="middle">expected finish →</text>
      <text x={12} y={H / 2} fill="var(--dimmer)" fontSize="10" textAnchor="middle" transform={`rotate(-90 12 ${H / 2})`}>tail risk (CVaR10) →</text>
      {/* frontier connecting line */}
      <polyline
        points={pts.filter((p) => opt.has(p.label)).sort((a, b) => a.mean_pos - b.mean_pos)
          .map((p) => `${fx(p.mean_pos)},${fy(p.cvar10)}`).join(' ')}
        fill="none" stroke="var(--accent)" strokeWidth="1.5" strokeDasharray="4 3" opacity="0.7" />
      {pts.map((p) => {
        const isOpt = opt.has(p.label), isBest = p.label === bestLabel
        return (
          <g key={p.label}>
            <circle cx={fx(p.mean_pos)} cy={fy(p.cvar10)} r={isBest ? 7 : isOpt ? 5 : 4}
              fill={isBest ? 'var(--accent)' : isOpt ? 'rgba(55,190,255,0.5)' : 'var(--dimmer)'}
              stroke={isBest ? '#fff' : 'none'} strokeWidth="1.5">
              <title>{p.label}: E[P{p.mean_pos}], CVaR {p.cvar10}</title>
            </circle>
            {isBest && (
              <text x={fx(p.mean_pos) + 10} y={fy(p.cvar10) + 3} fill="var(--accent)" fontSize="11"
                fontFamily="var(--mono)">{p.label}</text>
            )}
          </g>
        )
      })}
    </svg>
  )
}
