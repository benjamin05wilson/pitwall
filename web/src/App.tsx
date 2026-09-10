import { useState } from 'react'
import RaceMap from './RaceMap'
import RaceReplay from './RaceReplay'
import StrategyView from './StrategyView'

export default function App() {
  const [mode, setMode] = useState<'race' | 'replay' | 'strategy'>('strategy')
  const [preset, setPreset] = useState<{ year: number; gp: string; driver: string } | undefined>()

  return (
    <div className="app">
      <div className="topbar">
        <div className="brand">
          <div className="logo">pit<span className="accent">wall</span></div>
          <div className="tag">race-strategy engine</div>
        </div>
        <div className="modes">
          <button className={mode === 'race' ? 'active' : ''} onClick={() => setMode('race')}>🏁 Race</button>
          <button className={mode === 'replay' ? 'active' : ''} onClick={() => setMode('replay')}>▶ Historical Replay</button>
          <button className={mode === 'strategy' ? 'active' : ''} onClick={() => setMode('strategy')}>⚙ Strategy Lab</button>
        </div>
        <div className="spacer" />
        <span className="pill"><span className="dot">●</span> offline modelling demonstrator</span>
      </div>

      {mode === 'race' && (
        <RaceMap onPickDriver={(year, gp, driver) => { setPreset({ year, gp, driver }); setMode('replay') }} />
      )}
      {mode === 'replay' && <RaceReplay preset={preset} />}
      {mode === 'strategy' && <StrategyView />}

      <div className="foot">
        pitwall · Monte-Carlo shortlist optimisation · optional surrogate · historical Bayesian replay
      </div>
    </div>
  )
}
