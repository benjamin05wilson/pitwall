import { useEffect, useMemo, useRef, useState } from 'react'
import { api, type RaceMap as RM, type RaceCar, type RaceOption, type Replay, type Ghost, type WetStrategy } from './api'

const KIND_COLOR: Record<string, string> = { sc: '#f59e0b', flag: '#ef4444', penalty: '#fb923c', incident: '#fbbf24', retire: '#f87171' }

export default function RaceMap({ onPickDriver }: { onPickDriver?: (year: number, gp: string, code: string) => void }) {
  const [races, setRaces] = useState<RaceOption[]>([])
  const [sel, setSel] = useState(0)
  const [rm, setRm] = useState<RM | null>(null)
  const [loading, setLoading] = useState(false)
  const [playing, setPlaying] = useState(false)
  const [speed, setSpeed] = useState(15)
  const [clockF, setClockF] = useState(0)        // float frame index
  const [pick, setPick] = useState<string | null>(null)
  const [strat, setStrat] = useState<Replay | null>(null)
  const [ghost, setGhost] = useState<Ghost | null>(null)
  const [ghostLoading, setGhostLoading] = useState(false)
  const [wet, setWet] = useState<WetStrategy | null>(null)
  const baseRef = useRef({ f: 0, t: 0 })

  useEffect(() => { api.replayOptions().then(setRaces) }, [])
  useEffect(() => {
    if (!races.length) return
    const r = races[sel]
    setLoading(true); setPlaying(false); setClockF(0); setPick(null); setStrat(null)
    api.raceMap(r.year, r.gp).then((d) => { setRm(d); setLoading(false) }).catch(() => { setRm(null); setLoading(false) })
  }, [races, sel])

  // Fetch the selected car's best-response strategy vs the predicted field.
  useEffect(() => {
    if (!rm || !pick) { setStrat(null); setGhost(null); setWet(null); return }
    setGhost(null); setWet(null)
    api.fieldStrategy(rm.year, rm.gp, pick).then(setStrat).catch(() => setStrat(null))
    api.wetStrategy(rm.year, rm.gp, pick).then((w) => setWet(w.error ? null : w)).catch(() => setWet(null))
  }, [rm, pick])

  const runGhost = () => {
    if (!rm || !pick) return
    setGhostLoading(true)
    api.ghost(rm.year, rm.gp, pick).then((g) => { setGhost(g); setGhostLoading(false) })
      .catch(() => setGhostLoading(false))
  }

  const N = rm?.n_frames ?? 1
  // Time-based via setInterval (advances from elapsed wall-clock so the speed
  // control is exact, smooth in the foreground, and still progresses when the
  // tab is backgrounded — rAF would pause there).
  useEffect(() => {
    if (!playing || !rm) return
    baseRef.current = { f: clockF, t: performance.now() }
    // speed=1 advances exactly 1 race-second per real-second (true real time,
    // the full ~90 min). step = race-seconds per frame.
    const FPS = 1 / rm.step
    const id = window.setInterval(() => {
      const f = Math.min(N - 1, baseRef.current.f + ((performance.now() - baseRef.current.t) / 1000) * FPS * speed)
      setClockF(f)
      if (f >= N - 1) setPlaying(false)
    }, 33)
    return () => window.clearInterval(id)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [playing, speed, rm, N])

  const tf = useMemo(() => {
    if (!rm) return null
    const [xmin, xmax, ymin, ymax] = rm.track.bounds
    const W = 1000, H = 560, pad = 30
    const s = Math.min((W - 2 * pad) / (xmax - xmin), (H - 2 * pad) / (ymax - ymin))
    const ox = (W - (xmax - xmin) * s) / 2, oy = (H - (ymax - ymin) * s) / 2
    return { W, H, px: (x: number) => ox + (x - xmin) * s, py: (y: number) => H - (oy + (y - ymin) * s) }
  }, [rm])

  const trackPath = useMemo(() => {
    if (!rm || !tf) return ''
    return rm.track.x.map((x, i) => `${i === 0 ? 'M' : 'L'}${tf.px(x).toFixed(1)},${tf.py(rm.track.y[i]).toFixed(1)}`).join(' ') + 'Z'
  }, [rm, tf])

  const frame = Math.floor(clockF)
  const tRace = rm ? clockF * rm.step : 0
  const f0 = frame, f1 = Math.min(N - 1, frame + 1), fr = clockF - frame

  const carXY = (c: RaceCar) => {
    const x0 = c.x[f0], y0 = c.y[f0]
    if (x0 == null || y0 == null) return null
    const x1 = c.x[f1], y1 = c.y[f1]
    if (x1 == null || y1 == null) return { x: x0, y: y0 }
    return { x: x0 + (x1 - x0) * fr, y: y0 + (y1 - y0) * fr }
  }
  const isOut = (c: RaceCar) => c.retired && c.retire_frame != null && frame >= c.retire_frame
  const curPos = (c: RaceCar) => { let p: number | null = null; for (const e of c.pos_timeline) { if (e.t <= tRace) p = e.pos; else break } return p }
  const curLap = (c: RaceCar) => { let l = 0; for (const e of c.pos_timeline) { if (e.t <= tRace) l = e.lap; else break } return l }

  const leaderboard = useMemo(() => {
    if (!rm) return []
    return rm.cars.filter((c) => !isOut(c) && curPos(c) != null)
      .sort((a, b) => (curPos(a)! - curPos(b)!))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rm, tRace])

  const recentEvents = useMemo(() =>
    rm ? rm.events.filter((e) => e.t != null && e.t <= tRace).reverse().slice(0, 8) : [],
    [rm, tRace])

  const lapNow = rm ? Math.min(rm.n_laps, Math.max(1, leaderboard[0] ? curLap(leaderboard[0]) : 1)) : 0
  const pickCar = rm?.cars.find((c) => c.code === pick)
  const stLap = pickCar ? curLap(pickCar) : 0
  const stState = strat && stLap > 0 ? strat.timeline[Math.min(strat.timeline.length - 1, stLap - 1)] : null

  // On-track GHOST: the AI-strategy car riding the real car's racing line,
  // time-shifted by the per-lap real−AI gap (common-random-numbers, so it is the
  // strategy effect alone). +gap ⇒ AI ahead ⇒ shown further along the line.
  const ghostGapPts = useMemo(() => {
    if (!ghost?.ghost.gap_s || !pickCar) return null
    const g = ghost.ghost.gap_s
    const pts = pickCar.pos_timeline.filter((e) => e.lap < g.length).map((e) => ({ t: e.t, gap: g[e.lap] }))
    return pts.length >= 2 ? pts : null
  }, [ghost, pickCar])

  const gapAtTime = (t: number): number => {
    const pts = ghostGapPts!
    if (t <= pts[0].t) return pts[0].gap
    for (let i = 1; i < pts.length; i++) {
      if (t <= pts[i].t) {
        const a = pts[i - 1], b = pts[i]
        const u = b.t === a.t ? 0 : (t - a.t) / (b.t - a.t)
        return a.gap + (b.gap - a.gap) * u
      }
    }
    return pts[pts.length - 1].gap
  }

  const ghostDot = useMemo(() => {
    if (!rm || !ghostGapPts || !pickCar || isOut(pickCar)) return null
    const gap = gapAtTime(tRace)
    const gf = clockF + gap / rm.step             // frames to shift along the line
    if (gf <= 0 || gf >= N - 1) return null        // off the back / already finished
    const a = Math.floor(gf), b = Math.min(N - 1, a + 1), u = gf - a
    const x0 = pickCar.x[a], y0 = pickCar.y[a], x1 = pickCar.x[b], y1 = pickCar.y[b]
    if (x0 == null || y0 == null || x1 == null || y1 == null) return null
    return { x: x0 + (x1 - x0) * u, y: y0 + (y1 - y0) * u, gap }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [rm, ghostGapPts, pickCar, clockF, tRace, N])

  return (
    <div style={{ padding: '14px 22px 30px' }}>
      <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 12, flexWrap: 'wrap' }}>
        <select value={sel} onChange={(e) => setSel(+e.target.value)} style={{ maxWidth: 340 }}>
          {races.map((r, i) => <option key={i} value={i}>{r.label}</option>)}
        </select>
        {loading && <span className="loading"><span className="spinner" /> loading telemetry…</span>}
        {rm && <span className="pill">{leaderboard.length} running · {(rm.cars.filter(isOut)).length} out</span>}
      </div>

      <div className="transport" style={{ marginBottom: 14 }}>
        <button className="play" onClick={() => setPlaying((p) => !p)} disabled={!rm}>{playing ? '⏸' : '▶'}</button>
        <button className="step" onClick={() => { setPlaying(false); setClockF(0) }} disabled={!rm}>⏮</button>
        <div className="lapcount">LAP <b>{lapNow}</b> / {rm?.n_laps ?? 0} · {fmtClock(tRace)}</div>
        <input className="scrub" type="range" min={0} max={N - 1} value={frame}
          onChange={(e) => { setPlaying(false); setClockF(+e.target.value) }} disabled={!rm} />
        <div className="speeds">{[1, 5, 15, 60].map((s) => <button key={s} className={speed === s ? 'active' : ''} onClick={() => setSpeed(s)} title={s === 1 ? 'real race time' : `${s}× real time`}>{s}×</button>)}</div>
      </div>

      {!rm || !tf ? <div className="loading" style={{ padding: 40 }}>Loading race…</div> : (
        <div className="racemap-grid">
          <div className="card" style={{ padding: 10 }}>
            <svg viewBox={`0 0 ${tf.W} ${tf.H}`} width="100%" style={{ display: 'block' }}>
              <path d={trackPath} fill="none" stroke="#243246" strokeWidth="15" strokeLinejoin="round" />
              <path d={trackPath} fill="none" stroke="#0c141d" strokeWidth="10" strokeLinejoin="round" />
              <circle cx={tf.px(rm.track.x[0])} cy={tf.py(rm.track.y[0])} r="4" fill="#fff" opacity="0.5" />
              {/* crash markers */}
              {rm.cars.filter(isOut).map((c) => {
                const rf = (c.retire_frame ?? 1) - 1
                const x = c.x[rf], y = c.y[rf]
                if (x == null || y == null) return null
                return <text key={'x' + c.num} x={tf.px(x)} y={tf.py(y) + 4} fontSize="14" fill="#f87171" textAnchor="middle" opacity="0.8">✕</text>
              })}
              {/* cars */}
              {rm.cars.map((c) => {
                const p = carXY(c)
                if (!p || isOut(c)) return null
                const on = pick === c.code
                return (
                  <g key={c.num} onClick={() => setPick(c.code)} style={{ cursor: 'pointer' }}>
                    {on && <circle cx={tf.px(p.x)} cy={tf.py(p.y)} r="12" fill="none" stroke="#fff" strokeWidth="1.5" />}
                    <circle cx={tf.px(p.x)} cy={tf.py(p.y)} r={on ? 8 : 6} fill={c.color} stroke="#0a0e14" strokeWidth="1.5" />
                    {(on || (curPos(c) ?? 99) <= 10) && (
                      <text x={tf.px(p.x) + 9} y={tf.py(p.y) + 4} fontSize="11" fontFamily="var(--mono)"
                        fill={on ? '#fff' : 'var(--dim)'} fontWeight={on ? 700 : 400}>{c.code}</text>
                    )}
                  </g>
                )
              })}
              {/* on-track AI ghost car (translucent, riding the real racing line) */}
              {ghostDot && (
                <g pointerEvents="none">
                  <circle cx={tf.px(ghostDot.x)} cy={tf.py(ghostDot.y)} r="11" fill="none"
                    stroke="var(--accent)" strokeWidth="1.3" strokeDasharray="3 2" opacity="0.85" />
                  <circle cx={tf.px(ghostDot.x)} cy={tf.py(ghostDot.y)} r="6.5" fill="var(--accent)"
                    stroke="#0a0e14" strokeWidth="1.2" opacity="0.5" />
                  <text x={tf.px(ghostDot.x) + 10} y={tf.py(ghostDot.y) - 7} fontSize="10.5"
                    fontFamily="var(--mono)" fill="var(--accent)" fontWeight={700}>
                    AI {ghostDot.gap >= 0 ? `+${ghostDot.gap.toFixed(0)}` : ghostDot.gap.toFixed(0)}s
                  </text>
                </g>
              )}
            </svg>
            {ghost && (
              <div className="muted" style={{ fontSize: 10.5, textAlign: 'center', marginTop: 4 }}>
                the translucent <b style={{ color: 'var(--accent)' }}>AI ghost</b> rides {pick}’s racing line, shifted by the live strategy gap
                {ghostDot && <> — currently <b style={{ color: 'var(--accent)' }}>{ghostDot.gap >= 0 ? `${ghostDot.gap.toFixed(0)}s ahead` : `${(-ghostDot.gap).toFixed(0)}s behind`}</b></>}
              </div>
            )}
          </div>

          <div className="rm-right">
            <div className="card">
              <h3>Running order</h3>
              <div className="lb">
                {leaderboard.map((c, i) => (
                  <button key={c.num} className={`lbrow${pick === c.code ? ' on' : ''}`} onClick={() => setPick(c.code)}>
                    <span className="lbp">{i + 1}</span>
                    <span className="lbdot" style={{ background: c.color }} />
                    <span className="lbc">{c.code}</span>
                    <span className="lbt">{c.team.split(' ')[0]}</span>
                    {i < 10 && <span className="lbpts">pts</span>}
                  </button>
                ))}
              </div>
            </div>

            {pick && (
              <div className="card">
                <h3>{pick} · strategy <span className="badge">live · lap {stLap}</span></h3>
                {stState ? (
                  <div className="stratbox">
                    <div className="strow">
                      <span className="tdot" style={{ background: compColor(stState.compound) }} />
                      <b>{stState.compound?.[0] ?? '—'}</b> <small>· {stState.tyre_age} laps</small>
                      <span className="muted" style={{ marginLeft: 'auto' }}>P{curPos(pickCar!) ?? '—'}</span>
                    </div>
                    <div className="strow muted">live deg <b style={{ color: 'var(--accent)' }}>{stState.deg?.toFixed(3) ?? '—'}</b> s/lap
                      {stState.deg != null && <> (90% {stState.deg_lo.toFixed(2)}–{stState.deg_hi.toFixed(2)})</>}</div>
                    {stState.plan ? (() => {
                      const p = stState.plan!
                      const dec = p.decision
                      const cls = dec === 'BOX NOW' ? 'box' : dec === 'RUN TO FLAG' ? 'flag' : 'stay'
                      const rerouted = stState.track !== 'green'
                      return (
                        <div className={`engine ${cls}`}>
                          <div className="engtop">
                            <span>STRATEGY ENGINE</span>
                            <span className="neval">⟳ sweeping {p.n_eval.toLocaleString()} outcomes</span>
                          </div>
                          <div className="decision">
                            {dec === 'BOX NOW' ? '🟢 BOX NOW' : dec === 'RUN TO FLAG' ? '🏁 RUN TO FLAG' : `STAY OUT — box lap ${p.first_pit}`}
                            {p.first_comp && dec !== 'RUN TO FLAG' && <span className="oncomp"> · {p.first_comp[0]}</span>}
                          </div>
                          <div className="plan-label">{p.label}{p.second_gap > 0 && <span className="muted"> · best by {p.second_gap}s</span>}</div>
                          {rerouted && <div className="reroute">⚡ {stState.track} — recalculated, cheap stop window open</div>}
                        </div>
                      )
                    })() : <div className="muted small">strategy engine warming up…</div>}
                    {stState.undercut?.viable && (
                      <div className="battle uc">⚡ UNDERCUT <b>{stState.undercut.code}</b> (P{stState.undercut.pos}) — box now · +{stState.undercut.gap}s ahead, his tyres box ~L{stState.undercut.pred_pit} · jump by ~{stState.undercut.margin}s</div>
                    )}
                    {stState.overcut?.viable && (
                      <div className="battle threat">⚠ THREAT <b>{stState.overcut.code}</b> (P{stState.overcut.pos}) behind · {Math.abs(stState.overcut.gap)}s back, could undercut you</div>
                    )}
                    <div className="muted" style={{ fontSize: 10.5 }}>predicting all {rm.cars.length} cars' tyres & pit windows · best response</div>
                    {!ghost && (
                      <button className="ghostbtn" onClick={runGhost} disabled={ghostLoading}>
                        {ghostLoading ? <><span className="spinner" /> simulating both races…</> : 'What if we’d followed the AI?'}
                      </button>
                    )}
                    {ghost && <GhostCard g={ghost} />}
                    {wet && wet.meta.peak_wetness > 0.1 && <WetCard w={wet} />}
                    <button className="openfull" onClick={() => onPickDriver?.(rm.year, rm.gp, pick)}>open full strategy view →</button>
                  </div>
                ) : <div className="loading">loading {pick}…</div>}
              </div>
            )}

            <div className="card">
              <h3>Race control</h3>
              <div className="feed">
                {recentEvents.length === 0 && <div className="loading">— green flag racing —</div>}
                {recentEvents.map((e, i) => (
                  <div key={i} className="fe" style={{ borderLeftColor: KIND_COLOR[e.kind] ?? 'var(--border-bright)' }}>
                    <span className="fl">L{e.lap ?? '–'}</span> {e.message}
                  </div>
                ))}
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

function fmtClock(s: number) { const m = Math.floor(s / 60); return `${m}:${Math.floor(s % 60).toString().padStart(2, '0')}` }
function compColor(c: string | null) { return ({ SOFT: '#ff4d4d', MEDIUM: '#ffd23f', HARD: '#e8eef5' } as Record<string, string>)[c ?? ''] ?? '#888' }
function inWin(s: { rec_pit: { lo: number; hi: number } | null }, lap: number) { return !!s.rec_pit && lap >= s.rec_pit.lo && lap <= s.rec_pit.hi }

function WetCard({ w }: { w: WetStrategy }) {
  const W = 320, H = 64, pad = 14
  const n = w.wetness.length - 1
  const fx = (l: number) => pad + ((l - 1) / Math.max(1, n - 1)) * (W - pad - 6)
  const fy = (val: number) => H - 8 - val * (H - 18)
  const area = w.wetness.slice(1).map((v, i) => `${i ? 'L' : 'M'}${fx(i + 1).toFixed(1)},${fy(v).toFixed(1)}`).join(' ')
  const tyreColor = (t: string) => t === 'INTER' || t === 'INTERMEDIATE' ? '#3fd16b' : t === 'WET' ? '#4da6ff' : '#ffd23f'
  return (
    <div className="ghostcard">
      <div className="ghead">
        <span>🌧 WET CROSSOVER — when to switch tyres</span>
        <span className="gd">peak {Math.round(w.meta.peak_wetness * 100)}% @L{w.meta.peak_lap}</span>
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} width="100%">
        <path d={`${area} L${fx(n).toFixed(1)},${H} L${fx(1).toFixed(1)},${H} Z`} fill="rgba(77,166,255,0.18)" stroke="none" />
        <path d={area} fill="none" stroke="#4da6ff" strokeWidth="1.6" />
        {w.ai.switches.map(([lp, c], i) => (
          <g key={i}>
            <line x1={fx(lp)} y1={4} x2={fx(lp)} y2={H} stroke={tyreColor(c)} strokeWidth="1.5" strokeDasharray="3 2" />
            <text x={fx(lp) + 3} y={13} fill={tyreColor(c)} fontSize="9">{c[0]}@{lp}</text>
          </g>
        ))}
      </svg>
      <div className="grow"><span className="gl ai" /> AI optimal <b>{w.ai.label}</b></div>
      {w.real && (
        <div className="grow"><span className="gl real" /> {w.real.driver} real <b>{w.real.stints.map(([lp, c]) => `${c[0]}@${lp}`).join(' → ')}</b></div>
      )}
      <div className="muted" style={{ fontSize: 10 }}>
        crossovers calibrated from real wet laps: slicks→inters at {Math.round(w.crossovers.slick_to_inter * 100)}% wet, inters→full-wets at {Math.round(w.crossovers.inter_to_wet * 100)}%
      </div>
    </div>
  )
}

function GhostCard({ g }: { g: Ghost }) {
  const W = 320, H = 130, pad = 18
  const N = Math.max(g.real.trace.length, g.ghost.trace.length, 2)
  const fy = (p: number) => pad + ((p - 1) / 19) * (H - 2 * pad)
  const fx = (l: number) => pad + ((l - 1) / (N - 1)) * (W - pad - 6)
  const path = (t: number[]) => t.map((p, i) => `${i ? 'L' : 'M'}${fx(i + 1).toFixed(1)},${fy(p).toFixed(1)}`).join(' ')
  const win = g.delta > 0.5, lose = g.delta < -0.5
  return (
    <div className="ghostcard">
      <div className="ghead">
        <span>GHOST — what if we’d followed the AI</span>
        <span className={`gd ${win ? 'good' : lose ? 'bad' : ''}`}>{win ? `AI +${g.delta}` : lose ? `AI ${g.delta}` : 'line-ball'}</span>
      </div>
      <div className="grow"><span className="gl real" /> real <b>{g.real.strategy}</b> → sim P{g.real.median_finish}</div>
      <div className="grow"><span className="gl ai" /> AI <b>{g.ghost.strategy}</b> → sim P{g.ghost.median_finish}</div>
      {g.ghost.reacted_sc && g.ghost.reacted_sc.length > 0
        ? <div className="battle uc" style={{ fontSize: 11, padding: '5px 8px', marginBottom: 4 }}>⚡ AI reacted live — took the cheap stop under the Safety Car (lap {g.ghost.reacted_sc.join(', ')})</div>
        : <div className="muted" style={{ fontSize: 10, marginBottom: 4 }}>AI re-decides every lap — same start tyre, it boxes under a Safety Car if its window aligns</div>}
      <svg viewBox={`0 0 ${W} ${H}`} width="100%">
        {[1, 5, 10, 15, 20].map((p) => <line key={p} x1={pad} y1={fy(p)} x2={W - 6} y2={fy(p)} stroke="var(--border)" />)}
        <text x={2} y={fy(1) + 3} fill="var(--dimmer)" fontSize="8">P1</text>
        <text x={2} y={fy(20) + 3} fill="var(--dimmer)" fontSize="8">P20</text>
        <path d={path(g.real.trace)} fill="none" stroke="var(--dim)" strokeWidth="1.6" />
        <path d={path(g.ghost.trace)} fill="none" stroke="var(--accent)" strokeWidth="2.2" />
      </svg>
      {g.ghost.sets_used && (
        <div className="muted" style={{ fontSize: 10 }}>
          ✓ legal on the weekend’s tyres — AI uses {Object.entries(g.ghost.sets_used).map(([c, n]) => `${n}×${c[0]}`).join(' ')}
          {g.ghost.feasible === false && <span style={{ color: 'var(--bad)' }}> · ⚠ exceeds allocation</span>}
        </div>
      )}
      <div className="muted" style={{ fontSize: 10.5 }}>actually finished P{g.meta.actual_finish ?? '—'} · both run vs the real field under the actual safety cars & retirements · pace, deg, pit-loss & SC-risk all calibrated on real data</div>
    </div>
  )
}
