import { useEffect, useRef, useState } from 'react'
import { api, type Circuit, type OptimizeResult, type Heatmap as HM, type SCResult, type Params } from './api'
import { pct } from './util'
import StintBar from './components/StintBar'
import Heatmap from './components/Heatmap'
import Frontier from './components/Frontier'
import SafetyCar from './components/SafetyCar'

const OBJECTIVES = ['podium', 'win', 'points', 'expected', 'robust']

export default function StrategyView() {
  const [circuits, setCircuits] = useState<Circuit[]>([])
  const [p, setP] = useState<Params>({ circuit: 'bahrain', grid: 3, delta: 0.3, objective: 'podium', scenarios: 400 })
  const [opt, setOpt] = useState<OptimizeResult | null>(null)
  const [hm, setHm] = useState<HM | null>(null)
  const [sc, setSc] = useState<SCResult | null>(null)
  const [busy, setBusy] = useState(false)
  const timer = useRef<number | undefined>(undefined)

  useEffect(() => { api.circuits().then(setCircuits) }, [])
  useEffect(() => {
    window.clearTimeout(timer.current)
    timer.current = window.setTimeout(async () => {
      setBusy(true)
      try {
        const [o, h, s] = await Promise.all([
          api.optimize(p),
          api.heatmap({ circuit: p.circuit, grid: p.grid, delta: p.delta }),
          api.sc({ circuit: p.circuit, grid: p.grid, delta: p.delta }),
        ])
        setOpt(o); setHm(h); setSc(s)
      } catch (e) { console.error(e) } finally { setBusy(false) }
    }, 220)
  }, [p])

  const circuit = circuits.find((c) => c.id === p.circuit)
  const nLaps = circuit?.n_laps ?? 57
  const set = (patch: Partial<Params>) => setP((prev) => ({ ...prev, ...patch }))

  return (
    <>
      <div style={{ display: 'flex', justifyContent: 'flex-end', gap: 8, padding: '0 22px' }}>
        {opt && <span className="pill">optimised in {opt.compute_ms} ms · {opt.scenarios} races · Rust</span>}
        <span className="pill"><span className="dot">●</span> {busy ? 'computing…' : 'live'}</span>
      </div>
      <div className="layout">
        <div className="card" style={{ height: 'fit-content' }}>
          <h3>Race setup</h3>
          <div className="field">
            <label>Circuit</label>
            <select value={p.circuit} onChange={(e) => set({ circuit: e.target.value })}>
              {circuits.map((c) => <option key={c.id} value={c.id}>{c.name}</option>)}
            </select>
          </div>
          <div className="field">
            <label>Starting grid <span className="val">P{p.grid}</span></label>
            <input type="range" min={1} max={20} value={p.grid} onChange={(e) => set({ grid: +e.target.value })} />
          </div>
          <div className="field">
            <label>Car pace vs field <span className="val">{p.delta > 0 ? '+' : ''}{p.delta.toFixed(2)} s/lap</span></label>
            <input type="range" min={-0.5} max={2.5} step={0.05} value={p.delta} onChange={(e) => set({ delta: +e.target.value })} />
          </div>
          <div className="field">
            <label>Objective</label>
            <div className="segmented">
              {OBJECTIVES.map((o) => (
                <button key={o} className={p.objective === o ? 'active' : ''} onClick={() => set({ objective: o })}>{o}</button>
              ))}
            </div>
          </div>
          {circuit && (
            <div style={{ marginTop: 20, borderTop: '1px solid var(--border)', paddingTop: 14 }}>
              <div className="metrics" style={{ gridTemplateColumns: '1fr 1fr' }}>
                <Mini k="laps" v={`${circuit.n_laps}`} />
                <Mini k="pit loss" v={`${circuit.pit_loss}s`} />
                <Mini k="overtaking" v={`${circuit.overtake.toFixed(2)}`} />
                <Mini k="P(safety car)" v={pct(circuit.p_sc)} />
              </div>
            </div>
          )}
        </div>

        <div className="main">
          <div className="card span2">
            <h3>Recommendation{busy && <span className="spinner" style={{ marginLeft: 8 }} />}</h3>
            {opt ? (
              <>
                <div className="reco">
                  <div className="headline">
                    <div className="strat-label">{opt.best.label}</div>
                    <div className="sub">{opt.best.n_stops}-stop · {opt.best.compounds.join('–')}
                      {opt.best.pit_laps.length > 0 && <> · box lap {opt.best.pit_laps.join(', ')}</>}</div>
                    <StintBar compounds={opt.best.compounds} pitLaps={opt.best.pit_laps} nLaps={nLaps} />
                  </div>
                  <div className="metrics">
                    <Metric k="exp. finish" v={`P${opt.best.mean_pos.toFixed(1)}`} cls="accent" />
                    <Metric k="win" v={pct(opt.best.p_win)} cls="good" />
                    <Metric k="podium" v={pct(opt.best.p_podium)} cls="good" />
                    <Metric k="points" v={pct(opt.best.p_points)} />
                    <Metric k="tail (CVaR)" v={`P${opt.best.cvar10.toFixed(1)}`} cls="warn" />
                    <Metric k="spread" v={`±${opt.best.std_pos.toFixed(1)}`} />
                    <Metric k="DNF" v={pct(opt.best.dnf)} />
                    <Metric k="stops" v={`${opt.best.n_stops}`} />
                  </div>
                </div>
                <div className="reason">
                  Optimised for <b>{opt.objective}</b> across <b>{opt.reason.n_scenarios}</b> stochastic races
                  (safety cars, VSCs, red flags, traffic & reliability) with common random numbers. Beats the next-best{' '}
                  <b>{opt.reason.runner_up}</b> by <b>{opt.reason.gap.toFixed(2)}</b> positions on expected finish.
                </div>
              </>
            ) : <div className="loading">computing…</div>}
          </div>

          <div className="card">
            <h3>Instant what-if — pit lap × compound
              {hm && <span className="badge">surrogate · {hm.n} strategies in {hm.compute_ms}ms</span>}</h3>
            {hm ? <Heatmap hm={hm} /> : <div className="loading">…</div>}
          </div>

          <div className="card">
            <h3>Risk / reward frontier</h3>
            {opt ? <Frontier res={opt} /> : <div className="loading">…</div>}
          </div>

          <div className="card">
            <h3>Safety-car counterfactual</h3>
            {sc ? <SafetyCar sc={sc} /> : <div className="loading">…</div>}
          </div>

          <div className="card">
            <h3>Candidate strategies</h3>
            {opt ? (
              <table>
                <thead><tr>
                  <th className="lbl">strategy</th><th>E[fin]</th><th>win</th><th>podium</th><th>CVaR</th><th>DNF</th>
                </tr></thead>
                <tbody>
                  {opt.ranked.slice(0, 9).map((r) => (
                    <tr key={r.label} className={r.label === opt.best.label ? 'best-row' : ''}>
                      <td className="lbl">{r.label}</td>
                      <td>{r.mean_pos.toFixed(2)}</td>
                      <td>{pct(r.p_win)}</td>
                      <td>{pct(r.p_podium)}</td>
                      <td>{r.cvar10.toFixed(1)}</td>
                      <td>{pct(r.dnf)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            ) : <div className="loading">…</div>}
          </div>
        </div>
      </div>
    </>
  )
}

function Metric({ k, v, cls = '' }: { k: string; v: string; cls?: string }) {
  return <div className="metric"><div className="k">{k}</div><div className={`v ${cls}`}>{v}</div></div>
}
function Mini({ k, v }: { k: string; v: string }) {
  return <div className="metric"><div className="k">{k}</div><div className="v" style={{ fontSize: 18 }}>{v}</div></div>
}
