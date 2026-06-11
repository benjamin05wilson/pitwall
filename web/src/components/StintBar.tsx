// Visualise a strategy as compound-coloured stint segments across the race.
interface Props { compounds: string[]; pitLaps: number[]; nLaps: number }

export default function StintBar({ compounds, pitLaps, nLaps }: Props) {
  const bounds = [0, ...pitLaps, nLaps]
  return (
    <div className="stintbar">
      {compounds.map((c, i) => {
        const len = bounds[i + 1] - bounds[i]
        return (
          <div key={i} className={`seg ${c}`} style={{ flex: len }} title={`${c} ${len} laps`}>
            {len > 4 ? c[0] : ''}
          </div>
        )
      })}
    </div>
  )
}
