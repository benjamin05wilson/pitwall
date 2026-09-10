import { useEffect, useMemo, useRef, useState } from 'react'
import { api, type Replay, type RaceOption, type DriverOption, type Scorecard } from './api'

const TEAM_COLOR: Record<string, string> = {
  Williams: '#37beff', 'Red Bull Racing': '#1b2a4a', Ferrari: '#e8002d', Mercedes: '#27f4d2',
  McLaren: '#ff8000', 'Aston Martin': '#229971', Alpine: '#0093cc', RB: '#6692ff',
  'Kick Sauber': '#52e252', Haas: '#b6babd', AlphaTauri: '#5e8faa', 'Alfa Romeo': '#c92d4b',
}
const COMP_COLOR: Record<string, string> = { SOFT: '#ff4d4d', MEDIUM: '#ffd23f', HARD: '#e8eef5', INTERMEDIATE: '#43b02a', WET: '#1e6cff' }
const TRACK_COLOR: Record<string, string> = { green: '#34d399', yellow: '#fbbf24', SC: '#f59e0b', VSC: '#fb923c', red: '#ef4444' }
const SPEEDS = [1, 2, 5]

export default function RaceReplay({ preset }: { preset?: { year: number; gp: string; driver: string } }) {
  const [races, setRaces] = useState<RaceOption[]>([])
  const [sel, setSel] = useState(0)
  const [drivers, setDrivers] = useState<DriverOption[]>([])
  const [driver, setDriver] = useState('ALB')
  const [replay, setReplay] = useState<Replay | null>(null)
  const [scorecard, setScorecard] = useState<Scorecard | null>(null)
  const [loading, setLoading] = useState(false)
  const [playing, setPlaying] = useState(false)
  const [live, setLive] = useState(false)
  const [source, setSource] = useState<string>('')
  const [speed, setSpeed] = useState(2)
  const [lap, setLap] = useState(1)
  const tick = useRef<number | undefined>(undefined)
  const evt = useRef<EventSource | null>(null)

  useEffect(() => { api.replayOptions().then(setRaces) }, [])

  // Apply a preset (e.g. a car clicked on the race map).
  useEffect(() => {
    if (!preset || !races.length) return
    const i = races.findIndex((r) => r.year === preset.year && r.gp === preset.gp)
    if (i >= 0) setSel(i)
    setDriver(preset.driver)
  }, [preset, races])

  useEffect(() => {
    if (!races.length) return
    const r = races[sel]
    api.replayDrivers(r.year, r.gp).then((d) => {
      setDrivers(d.drivers)
      const wil = d.drivers.find((x) => x.team.includes('Williams'))
      setDriver(wil ? wil.code : d.drivers[0]?.code)
    }).catch(() => setDrivers([]))
  }, [races, sel])

  // Stop live + reset on race/driver change.
  useEffect(() => { setLive(false) }, [races, sel, driver])

  // REPLAY mode: fetch the whole race.
  useEffect(() => {
    if (!races.length || !driver || live) return
    const r = races[sel]
    setLoading(true); setPlaying(false); setLap(1); setScorecard(null)
    api.replay(r.year, r.gp, driver).then((rp) => {
      setReplay(rp); setScorecard(rp.scorecard ?? null); setLoading(false)
    }).catch(() => { setReplay(null); setLoading(false) })
  }, [races, sel, driver, live])

  const nLaps = replay?.meta.n_laps ?? 1

  // REPLAY mode: play loop.
  useEffect(() => {
    window.clearInterval(tick.current)
    if (!playing || !replay || live) return
    tick.current = window.setInterval(() => {
      setLap((l) => { if (l >= nLaps) { setPlaying(false); return l } return l + 1 })
    }, 700 / speed)
    return () => window.clearInterval(tick.current)
  }, [playing, speed, replay, nLaps, live])

  // LIVE mode: stream lap-by-lap from the backend (SSE).
  useEffect(() => {
    evt.current?.close()
    if (!live || !races.length || !driver) return
    const r = races[sel]
    setLoading(true); setPlaying(false); setLap(0); setScorecard(null)
    setReplay((p) => (p ? { ...p, timeline: [] } : p))
    const es = new EventSource(`/api/live-stream?year=${r.year}&gp=${encodeURIComponent(r.gp)}&driver=${driver}&interval=${(1.1 / speed).toFixed(2)}`)
    evt.current = es
    es.onmessage = (m) => {
      const msg = JSON.parse(m.data)
      if (msg.type === 'meta') {
        setSource(msg.source)
        setReplay({ meta: msg.meta, actual_pit_laps: msg.actual_pit_laps, timeline: [] })
        setLoading(false)
      } else if (msg.type === 'lap') {
        setReplay((p) => (p ? { ...p, timeline: [...p.timeline, msg.state] } : p))
        setLap((l) => l + 1)
      } else if (msg.type === 'end') {
        setScorecard(msg.scorecard); es.close()
      }
    }
    es.onerror = () => { es.close(); setLoading(false) }
    return () => es.close()
  }, [live, races, sel, driver, speed])

  const cur = replay?.timeline[lap - 1]
  const team = replay?.meta.team ?? ''
  const teamCol = TEAM_COLOR[team] ?? '#37beff'

  const events = useMemo(() => {
    if (!replay) return []
    const ev: { lap: number; kind: string; text: string }[] = []
    let lastRec = -1, lastTrack = 'green'
    for (const t of replay.timeline) {
      if (t.lap > lap) break
      if (replay.actual_pit_laps.includes(t.lap)) ev.push({ lap: t.lap, kind: 'pit', text: `PIT — fresh ${t.compound ?? ''} tyres` })
      if (t.track !== 'green' && t.track !== lastTrack) ev.push({ lap: t.lap, kind: t.track, text: `${t.track.toUpperCase()} deployed` })
      lastTrack = t.track
      if (t.rec_pit && Math.abs(t.rec_pit.pit_lap - lastRec) > 1 && t.rec_pit.pit_lap < nLaps - 1) {
        ev.push({ lap: t.lap, kind: 'rec', text: `engine: box lap ${t.rec_pit.pit_lap} (window ${t.rec_pit.lo}–${t.rec_pit.hi})` })
        lastRec = t.rec_pit.pit_lap
      }
      if (t.anomaly) ev.push({ lap: t.lap, kind: 'warn', text: `anomalous lap — traffic / lock-up?` })
    }
    return ev.reverse().slice(0, 8)
  }, [replay, lap, nLaps])

  const shownStops = (scorecard?.stops ?? []).filter((s) => s.actual <= lap)

  return (
    <div style={{ padding: '14px 22px 30px' }}>
      <div style={{ display: 'flex', gap: 12, alignItems: 'center', marginBottom: 14, flexWrap: 'wrap' }}>
        <select value={sel} onChange={(e) => setSel(+e.target.value)} style={{ maxWidth: 320 }}>
          {races.map((r, i) => <option key={i} value={i}>{r.label}</option>)}
        </select>
        <select value={driver} onChange={(e) => setDriver(e.target.value)} style={{ maxWidth: 220 }}>
          {drivers.map((d) => <option key={d.code} value={d.code}>{d.code} — {d.team}{d.finish ? ` (P${d.finish})` : ''}</option>)}
        </select>
        <div className="livetoggle">
          <button className={!live ? 'active' : ''} onClick={() => setLive(false)}>▶ Replay</button>
          <button className={live ? 'active live' : ''} onClick={() => setLive(true)}>● Stream history</button>
        </div>
        {live && source && <span className="pill"><span className="dot">●</span> {'historical replay'}</span>}
        {loading && <span className="loading"><span className="spinner" /> {live ? 'connecting…' : 'loading race…'}</span>}
      </div>

      <div className="transport">
        <button className="play" onClick={() => setPlaying((p) => !p)} disabled={!replay || live}>{playing ? '⏸' : '▶'}</button>
        <button className="step" onClick={() => { setPlaying(false); setLap(1) }} disabled={!replay || live}>⏮</button>
        <div className="lapcount">LAP <b>{lap}</b> / {nLaps}</div>
        <input className="scrub" type="range" min={1} max={nLaps} value={Math.max(1, lap)}
          onChange={(e) => { setPlaying(false); setLap(+e.target.value) }} disabled={!replay || live} />
        <div className="speeds">
          {SPEEDS.map((s) => <button key={s} className={speed === s ? 'active' : ''} onClick={() => setSpeed(s)}>{s}×</button>)}
        </div>
      </div>

      {!replay ? <div className="loading" style={{ padding: 40 }}>Select a race to begin.</div> : (
        <>
          <div className="hero">
            <div className="pos-card" style={{ borderColor: teamCol }}>
              <div className="ttl">POSITION</div>
              <div className="bigpos" style={{ color: teamCol }}>{cur?.position ?? '—'}</div>
              <div className="drv">{replay.meta.driver} · {team}</div>
            </div>
            <div className="hero-cards">
              <Stat k="GAP AHEAD" v={cur?.gap_ahead != null ? `+${cur.gap_ahead}s` : '—'} />
              <Stat k="GAP BEHIND" v={cur?.gap_behind != null ? `+${cur.gap_behind}s` : '—'} />
              <div className="hstat">
                <div className="k">TYRE</div>
                <div className="tyre">
                  <span className="tdot" style={{ background: COMP_COLOR[cur?.compound ?? 'HARD'] }} />
                  <span className="v">{cur?.compound?.[0] ?? '—'} <small>· {cur?.tyre_age ?? 0} laps</small></span>
                </div>
              </div>
              <Stat k="LAP TIME" v={cur?.lap_time ? fmtLap(cur.lap_time) : '—'} />
              <div className="hstat" style={{ background: 'rgba(0,0,0,0.2)' }}>
                <div className="k">TRACK</div>
                <div className="v" style={{ color: TRACK_COLOR[cur?.track ?? 'green'] }}>
                  {(cur?.track ?? 'green') === 'green' ? '🟢 GREEN' : cur?.track === 'SC' ? '🟠 SAFETY CAR' : cur?.track === 'VSC' ? '🟠 VSC' : cur?.track === 'red' ? '🔴 RED' : '🟡 YELLOW'}
                </div>
              </div>
            </div>
          </div>

          <div className="reco-banner" style={{ borderColor: inWindow(cur) ? 'var(--good)' : 'var(--border-bright)' }}>
            {cur?.rec_pit && cur.rec_pit.pit_lap < nLaps - 1 ? (
              <>
                <span className="lbl">ENGINE CALL</span>
                <span className="big">BOX LAP <b>{cur.rec_pit.pit_lap}</b></span>
                <span className="win">window {cur.rec_pit.lo}–{cur.rec_pit.hi}</span>
                {inWindow(cur) && <span className="now">● IN WINDOW NOW</span>}
                <span className="sep" />
                <span className="muted">estimated deg <b>{cur.deg?.toFixed(3) ?? '—'}</b> s/lap (90% {cur.deg_lo.toFixed(3)}–{cur.deg_hi.toFixed(3)})</span>
              </>
            ) : <><span className="lbl">ENGINE CALL</span><span className="big">{(cur?.lap ?? 0) > 2 ? 'RUN TO THE FLAG' : 'GATHERING DATA…'}</span></>}
          </div>

          <div className="replay-grid">
            <div className="card">
              <h3>Historical tyre estimate <span className="badge">Kalman · updates each lap</span></h3>
              <DegChart replay={replay} lap={lap} />
            </div>
            <div className="card">
              <h3>Lap times</h3>
              <LapTrace replay={replay} lap={lap} />
            </div>
            <div className="card">
              <h3>Pit wall</h3>
              <div className="feed">
                {events.length === 0 && <div className="loading">— quiet so far —</div>}
                {events.map((e, i) => <div key={i} className={`fe fe-${e.kind}`}><span className="fl">L{e.lap}</span> {e.text}</div>)}
              </div>
            </div>
          </div>

          {shownStops.length > 0 && (
            <div className="card" style={{ marginTop: 14 }}>
              <h3>Engine vs the pit wall
                {scorecard && lap >= nLaps && <span className="badge">{scorecard.headline}</span>}</h3>
              <div className="scorecard">
                {shownStops.map((s, i) => (
                  <div key={i} className={`scrow ${s.in_window ? 'ok' : 'diff'}`}>
                    <span className="dot2">{s.in_window ? '✓' : '✕'}</span>
                    <span className="actual">Team boxed <b>lap {s.actual}</b></span>
                    <span className="arrow">→</span>
                    <span className="eng">engine: {s.engine != null ? `lap ${s.engine} (window ${s.lo}–${s.hi})` : '—'}</span>
                    <span className="vd">{s.verdict}</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          <div className="muted small" style={{ marginTop: 12 }}>
            {live ? 'Streaming selected historical observations lap by lap; no current telemetry.'
              : `Replaying real ${replay.meta.year} ${replay.meta.gp} data · retrospective analysis, not a forward prediction.`}
          </div>
        </>
      )}
    </div>
  )
}

function inWindow(c?: { lap: number; rec_pit: { lo: number; hi: number } | null }) {
  return !!c?.rec_pit && c.lap >= c.rec_pit.lo && c.lap <= c.rec_pit.hi
}
function fmtLap(s: number) { const m = Math.floor(s / 60); return `${m}:${(s - m * 60).toFixed(3).padStart(6, '0')}` }
function Stat({ k, v }: { k: string; v: string }) {
  return <div className="hstat"><div className="k">{k}</div><div className="v">{v}</div></div>
}

function DegChart({ replay, lap }: { replay: Replay; lap: number }) {
  const W = 520, H = 230, pad = 34
  const N = replay.meta.n_laps
  const pts = replay.timeline.filter((t) => t.lap <= lap && t.deg != null)
  const ymax = 0.3
  const fx = (l: number) => pad + ((l - 1) / (N - 1)) * (W - pad - 10)
  const fy = (d: number) => H - pad - (Math.max(0, Math.min(ymax, d)) / ymax) * (H - pad - 12)
  const band = pts.map((t) => `${fx(t.lap)},${fy(t.deg_hi)}`).concat(pts.slice().reverse().map((t) => `${fx(t.lap)},${fy(t.deg_lo)}`)).join(' ')
  const line = pts.map((t) => `${fx(t.lap)},${fy(t.deg!)}`).join(' ')
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%">
      {[0.1, 0.2, 0.3].map((g) => (
        <g key={g}><line x1={pad} y1={fy(g)} x2={W - 10} y2={fy(g)} stroke="var(--border)" />
          <text x={4} y={fy(g) + 3} fill="var(--dimmer)" fontSize="9">{g.toFixed(1)}</text></g>
      ))}
      {replay.actual_pit_laps.filter((p) => p <= lap).map((p) => (
        <line key={p} x1={fx(p)} y1={12} x2={fx(p)} y2={H - pad} stroke="var(--accent)" strokeDasharray="3 3" opacity="0.5" />
      ))}
      {pts.length > 1 && <polygon points={band} fill="rgba(55,190,255,0.16)" />}
      {pts.length > 1 && <polyline points={line} fill="none" stroke="var(--accent)" strokeWidth="2" />}
      {pts.length > 0 && <circle cx={fx(pts[pts.length - 1].lap)} cy={fy(pts[pts.length - 1].deg!)} r="3.5" fill="#fff" />}
      <text x={pad} y={H - 6} fill="var(--dimmer)" fontSize="9">deg rate (s/lap) · 90% credible band · vs lap</text>
    </svg>
  )
}

function LapTrace({ replay, lap }: { replay: Replay; lap: number }) {
  const W = 520, H = 230, pad = 40
  const N = replay.meta.n_laps
  const all = replay.timeline.filter((t) => t.lap_time != null).map((t) => t.lap_time!)
  const med = all.length ? all.slice().sort((a, b) => a - b)[Math.floor(all.length / 2)] : 95
  const lo = med - 4, hi = med + 8
  const pts = replay.timeline.filter((t) => t.lap <= lap && t.lap_time != null && t.lap_time! < hi + 6)
  const fx = (l: number) => pad + ((l - 1) / (N - 1)) * (W - pad - 10)
  const fy = (s: number) => H - pad - ((Math.max(lo, Math.min(hi, s)) - lo) / (hi - lo)) * (H - pad - 12)
  return (
    <svg viewBox={`0 0 ${W} ${H}`} width="100%">
      {replay.timeline.filter((t) => t.lap <= lap && t.track !== 'green').map((t) => (
        <rect key={t.lap} x={fx(t.lap) - 2} y={12} width={4} height={H - pad - 12} fill={TRACK_COLOR[t.track]} opacity="0.18" />
      ))}
      {replay.actual_pit_laps.filter((p) => p <= lap).map((p) => (
        <line key={p} x1={fx(p)} y1={12} x2={fx(p)} y2={H - pad} stroke="var(--accent)" strokeDasharray="3 3" opacity="0.6" />
      ))}
      <polyline points={pts.map((t) => `${fx(t.lap)},${fy(t.lap_time!)}`).join(' ')} fill="none" stroke="var(--teal)" strokeWidth="1.5" />
      {pts.length > 0 && <circle cx={fx(pts[pts.length - 1].lap)} cy={fy(pts[pts.length - 1].lap_time!)} r="3.5" fill="#fff" />}
      <text x={pad} y={H - 6} fill="var(--dimmer)" fontSize="9">lap time (s) · pits dashed · neutralised laps shaded</text>
    </svg>
  )
}
