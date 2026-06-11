import type { SCResult } from '../api'

function Panel({ title, p, base, flag }: { title: string; p: { pit_lap: number; lo: number; hi: number; pit_loss: number }; base: number; flag?: boolean }) {
  const shift = base - p.pit_lap
  return (
    <div className={`panel-sc${flag ? ' flag' : ''}`}>
      <div className="ttl">{title}</div>
      <div className="lap">
        {p.pit_lap}
        {shift > 0 && <span className="delta-down" style={{ fontSize: 14 }}> −{shift}</span>}
      </div>
      <div className="win">box window {p.lo}–{p.hi}</div>
      <div className="win">pit loss {p.pit_loss}s</div>
    </div>
  )
}

export default function SafetyCar({ sc }: { sc: SCResult }) {
  const base = sc.green.pit_lap
  return (
    <div>
      <div className="sc">
        <Panel title="Green flag" p={sc.green} base={base} />
        <Panel title="🟡 VSC now" p={sc.vsc} base={base} flag />
        <Panel title="🟠 Safety car now" p={sc.sc} base={base} flag />
      </div>
      <div className="sc-note">
        At lap {sc.current_lap}, a neutralisation makes the stop cheap <i>only inside its window</i>, so the
        engine pulls the box call forward — more so under a full <b>safety car</b> (loss {sc.sc.pit_loss}s)
        than a <b>VSC</b> (loss {sc.vsc.pit_loss}s) vs green ({sc.green.pit_loss}s). A <b>red flag</b>{' '}
        (≈{Math.round(sc.p_red * 100)}% chance) is a <b>free</b> tyre change — the biggest swing of all, and
        the simulator prices it into every recommendation.
      </div>
    </div>
  )
}
