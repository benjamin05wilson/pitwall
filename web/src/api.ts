// Typed client for the pitwall FastAPI backend.

export interface Circuit {
  id: string; name: string; n_laps: number;
  pit_loss: number; overtake: number; p_sc: number;
}

export interface Ensemble {
  mean_pos: number; median_pos: number; p_win: number; p_podium: number;
  p_points: number; cvar10: number; std_pos: number; dnf: number;
}

export interface Best extends Ensemble {
  label: string; n_stops: number; pit_laps: number[]; compounds: string[];
}

export interface Ranked extends Ensemble {
  label: string; stops: number; det_time: number;
}

export interface OptimizeResult {
  objective: string; scenarios: number; compute_ms: number;
  best: Best;
  reason: { runner_up: string; gap: number; pit_loss: number; n_scenarios: number };
  ranked: Ranked[];
  frontier: { label: string; mean_pos: number; cvar10: number }[];
  frontier_optimal: string[];
}

export interface Heatmap {
  laps: number[]; rows: string[]; z: (number | null)[][];
  n: number; compute_ms: number;
  best: { row: string; lap: number; mean_pos: number };
}

interface PitPanel { pit_lap: number; lo: number; hi: number; pit_loss: number }
export interface SCResult {
  current_lap: number;
  green: PitPanel;
  vsc: PitPanel;
  sc: PitPanel;
  red_flag: { pit_loss: number; note: string };
  p_sc: number;
  p_red: number;
}

export interface Params {
  circuit: string; grid: number; delta: number; objective: string; scenarios: number;
}

export interface LivePlan {
  decision: string; first_pit: number | null; first_comp: string | null;
  label: string; rem_time: number; n_eval: number; second_gap: number;
  remaining_stops: number; alternatives: { label: string; delta: number }[];
}
export interface Battle {
  code: string; pos: number; gap: number; pred_pit: number | null;
  per_lap: number; margin: number; viable: boolean; tyre: string | null; age: number;
}
export interface ReplayLap {
  lap: number; position: number | null; lap_time: number | null;
  compound: string | null; tyre_age: number;
  gap_ahead: number | null; gap_behind: number | null; track: string;
  is_pit: boolean; deg: number | null; deg_lo: number; deg_hi: number;
  fresh_pace: number; rec_pit: { pit_lap: number; lo: number; hi: number } | null; anomaly: boolean;
  plan: LivePlan | null;
  undercut?: Battle | null; overcut?: Battle | null;
}
export interface ScoreStop { actual: number; engine: number | null; lo?: number; hi?: number; verdict: string; in_window: boolean }
export interface Scorecard { stops: ScoreStop[]; agreed: number; total: number; headline: string }
export interface Replay {
  meta: { year: number; gp: string; circuit_id: string; driver: string; team: string; n_laps: number; pit_loss: number };
  actual_pit_laps: number[];
  timeline: ReplayLap[];
  scorecard?: Scorecard;
}
export interface PosPoint { lap: number; t: number; pos: number | null }
export interface RaceCar {
  num: string; code: string; team: string; color: string;
  x: (number | null)[]; y: (number | null)[]; finish: number | null; status: string;
  retired: boolean; retire_frame: number | null; pos_timeline: PosPoint[];
}
export interface RaceEvent { t: number | null; lap: number | null; kind: string; flag: string | null; message: string }
export interface RaceMap {
  track: { x: number[]; y: number[]; bounds: number[] };
  t0: number; step: number; n_frames: number; n_laps: number;
  gp: string; year: number; cars: RaceCar[]; events: RaceEvent[];
}
export interface GhostSide {
  strategy: string; pit_laps: number[]; compounds: string[]; adaptive?: boolean; reacted_sc?: number[];
  median_finish: number; mean_finish: number; p_podium: number; p_points: number; trace: number[];
  // Tyre-set allocation feasibility (the AI plan only uses sets the team had).
  sets_used?: Record<string, number>; allocation?: Record<string, number>; feasible?: boolean;
  // Per-lap cumulative time gap (real − AI; +ve ⇒ AI ahead) — drives the on-track ghost car.
  gap_s?: number[];
}
export interface Ghost {
  meta: { year: number; gp: string; driver: string; team: string; n_laps: number; actual_finish: number | null };
  real: GhostSide; ghost: GhostSide; delta: number; verdict: string;
}
export interface WetStrategy {
  meta: { year: number; gp: string; circuit_id: string; n_laps: number; peak_wetness: number; peak_lap: number };
  wetness: number[];
  ai: { label: string; switches: [number, string][]; tyres: string[] };
  crossovers: { slick_to_inter: number; inter_to_wet: number };
  real?: { driver: string; stints: [number, string][] };
  error?: string;
}
export interface RaceOption { year: number; gp: string; label: string }
export interface DriverOption { code: string; team: string; grid: number; finish: number | null }

async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(path, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${path} ${r.status}`);
  return r.json();
}

export const api = {
  circuits: (): Promise<Circuit[]> => fetch('/api/circuits').then((r) => r.json()),
  optimize: (p: Params) => post<OptimizeResult>('/api/optimize', p),
  heatmap: (p: Pick<Params, 'circuit' | 'grid' | 'delta'>) =>
    post<Heatmap>('/api/heatmap', p),
  sc: (p: Pick<Params, 'circuit' | 'grid' | 'delta'>) =>
    post<SCResult>('/api/sc-counterfactual', p),
  calibrate: (country: string, circuit: string, years: number[]) =>
    post<any>('/api/calibrate', { country, circuit, years }),
  backtest: (year: number, country: string, circuit: string, pool_years: number[]) =>
    post<any>('/api/backtest', { year, country, circuit, pool_years }),
  replayOptions: (): Promise<RaceOption[]> => fetch('/api/replay-options').then((r) => r.json()),
  replayDrivers: (year: number, gp: string) =>
    post<{ drivers: DriverOption[] }>('/api/replay-drivers', { year, gp }),
  replay: (year: number, gp: string, driver: string) =>
    post<Replay>('/api/replay', { year, gp, driver }),
  fieldStrategy: (year: number, gp: string, driver: string) =>
    post<Replay>('/api/field-strategy', { year, gp, driver }),
  raceMap: (year: number, gp: string) => post<RaceMap>('/api/race-map', { year, gp }),
  ghost: (year: number, gp: string, driver: string) => post<Ghost>('/api/ghost', { year, gp, driver }),
  wetStrategy: (year: number, gp: string, driver: string) =>
    post<WetStrategy>('/api/wet-strategy', { year, gp, driver }),
};
