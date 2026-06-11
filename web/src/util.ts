// Map a finishing position to a green->yellow->red colour (1 = best = green).
export function posColor(pos: number | null, lo = 1, hi = 16): string {
  if (pos == null) return '#0b1119'
  const t = Math.max(0, Math.min(1, (pos - lo) / (hi - lo)))
  // green (good) -> amber -> red (bad)
  const stops = [
    [26, 156, 91],   // green
    [214, 198, 74],  // amber
    [192, 57, 43],   // red
  ]
  const seg = t < 0.5 ? 0 : 1
  const f = t < 0.5 ? t / 0.5 : (t - 0.5) / 0.5
  const a = stops[seg], b = stops[seg + 1]
  const c = a.map((x, i) => Math.round(x + (b[i] - x) * f))
  return `rgb(${c[0]},${c[1]},${c[2]})`
}

export const fmt = (x: number, d = 2) => x.toFixed(d)
export const pct = (x: number) => `${Math.round(x * 100)}%`
