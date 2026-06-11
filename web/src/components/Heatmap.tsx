import type { Heatmap as HM } from '../api'
import { posColor } from '../util'

export default function Heatmap({ hm }: { hm: HM }) {
  const cols = hm.laps.length
  const grid = `40px repeat(${cols}, 1fr)`
  const tick = Math.max(1, Math.round(cols / 8))
  return (
    <div>
      <div className="heatmap">
        {hm.rows.map((row, ri) => (
          <div className="hm-row" key={row} style={{ gridTemplateColumns: grid }}>
            <div className="hm-axis-y">{row}</div>
            {hm.z[ri].map((v, ci) => {
              const isBest = hm.best.row === row && hm.laps[ci] === hm.best.lap
              return (
                <div
                  key={ci}
                  className={`hm-cell${isBest ? ' best' : ''}`}
                  style={{ background: posColor(v) }}
                  title={`${row} · pit lap ${hm.laps[ci]} · E[finish] P${v?.toFixed(1)}`}
                />
              )
            })}
          </div>
        ))}
      </div>
      <div className="hm-axis-x" style={{ paddingLeft: 40 }}>
        {hm.laps.filter((_, i) => i % tick === 0).map((l) => <span key={l}>{l}</span>)}
      </div>
      <div className="legend">
        <span>better</span><div className="scale" /><span>worse</span>
        <span style={{ marginLeft: 'auto', color: 'var(--accent)' }}>
          ◎ optimum: {hm.best.row} @ lap {hm.best.lap} → P{hm.best.mean_pos.toFixed(1)}
        </span>
      </div>
    </div>
  )
}
