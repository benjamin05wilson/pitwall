//! Rust accelerator for the pitwall F1 Monte-Carlo race simulator.
//!
//! Ports `pitwall.sim.race.RaceSimulator.run_once` (the pure-Python source of
//! truth) into a tight inner loop. The Python layer flattens a fixed field of
//! cars + a bank of scenarios into flat arrays and calls `simulate_batch`, which
//! returns the focal car's finishing position (and total time) per scenario.
//!
//! Each scenario owns a fresh `Pcg64` seeded from the scenario seed. Bit-parity
//! with NumPy's RNG streams is NOT expected (different generators / draw order);
//! the goal is statistical agreement within Monte-Carlo error.

use pyo3::prelude::*;
use rand::Rng;
use rand_distr::{Distribution, Normal};
use rand_pcg::Pcg64;
use rand::SeedableRng;

#[pyfunction]
fn ping() -> &'static str {
    "pitwall native extension online"
}

/// Per-car static model parameters, laid out as flat arrays of length `n`.
struct Field {
    n: usize,
    base_pace: Vec<f64>,   // t_quali_ref + race_pace_gap + driver_delta
    k_fuel: Vec<f64>,
    m_start: Vec<f64>,
    burn: Vec<f64>,        // already-resolved per-lap fuel burn
    sigma: Vec<f64>,       // lap_noise_sigma
    top_speed_delta: Vec<f64>,
    cold_penalty: Vec<f64>,
    deg_mult_sc: Vec<f64>,
    deg_mult_vsc: Vec<f64>,
    lap_mult_sc: Vec<f64>,
    lap_mult_vsc: Vec<f64>,
    pit_loss: Vec<f64>,
    sc_saving_frac: Vec<f64>,
    vsc_saving_frac: Vec<f64>,
    team_failure_prob: Vec<f64>,
    // Per-car per-compound tyre coefficients, compound index in [SOFT,MEDIUM,HARD].
    // Flattened row-major: k0[i*3 + c], etc.
    k0: Vec<f64>,
    k1: Vec<f64>,
    k2: Vec<f64>,
    start_compound: Vec<i64>,   // index 0/1/2
    grid: Vec<i64>,             // starting grid slot per car (1-indexed)
    grid_gap: f64,              // seconds of initial cum-time offset per grid slot
    // Pit plan per car: map lap -> new compound index. Stored as a parallel
    // ragged structure via offsets into pit_laps / pit_comps.
    pit_offsets: Vec<usize>,    // length n+1
    pit_laps: Vec<i64>,
    pit_comps: Vec<i64>,
}

impl Field {
    #[inline]
    fn tyre_penalty(&self, car: usize, comp: i64, age_int: i64) -> f64 {
        let idx = car * 3 + comp as usize;
        let a = age_int as f64;
        let mut pen = self.k0[idx] + self.k1[idx] * a + self.k2[idx] * a * a;
        if age_int == 0 {
            pen += self.cold_penalty[car];
        }
        pen
    }

    /// Deterministic green lap time, mirroring LapTimeModel.green_lap.
    /// `age` is truncated to int exactly as race.py does via `int(age[i])`.
    #[inline]
    fn green_lap(&self, car: usize, comp: i64, age: f64, lap: i64, n_laps: i64) -> f64 {
        let base = self.base_pace[car];
        // fuel: remaining = max(0, m_start - burn*(lap-1)); penalty = k_fuel*remaining
        let remaining = (self.m_start[car] - self.burn[car] * (lap - 1) as f64).max(0.0);
        let fuel = self.k_fuel[car] * remaining;
        let age_int = age as i64; // truncation toward zero (age >= 0)
        let tyre = self.tyre_penalty(car, comp, age_int);
        let _ = n_laps;
        base + fuel + tyre
    }

    /// Pit lap lookup: returns Some(new_compound_idx) if `car` pits at `lap`.
    #[inline]
    fn pit_at(&self, car: usize, lap: i64) -> Option<i64> {
        let lo = self.pit_offsets[car];
        let hi = self.pit_offsets[car + 1];
        for k in lo..hi {
            if self.pit_laps[k] == lap {
                return Some(self.pit_comps[k]);
            }
        }
        None
    }

    #[inline]
    fn effective_pit_loss(&self, car: usize, regime: u8) -> f64 {
        match regime {
            1 => self.pit_loss[car] * (1.0 - self.sc_saving_frac[car]), // SC
            2 => self.pit_loss[car] * (1.0 - self.vsc_saving_frac[car]), // VSC
            _ => self.pit_loss[car],
        }
    }
}

/// Circuit-level overtaking parameters (shared across the field).
struct Overtake {
    threshold_s: f64,
    velocity_mod: f64,
    p_overtake: f64,
    drs_bonus: f64,
    drs_window: f64,
    drs_enable_lap: i64,
    drs_sc_delay: i64,
    min_follow_gap: f64,
    duel_penalty: f64,
    loser_penalty: f64,
}

/// Sort indices of `running` cars by ascending cum time. Returns the order as a
/// Vec of global car indices. Uses a stable sort so ties keep input order, which
/// matches numpy.argsort's default (stable for the contiguous slice we feed it).
#[inline]
fn order_by_cum(running: &[usize], cum: &[f64]) -> Vec<usize> {
    let mut order: Vec<usize> = running.to_vec();
    order.sort_by(|&a, &b| cum[a].partial_cmp(&cum[b]).unwrap_or(std::cmp::Ordering::Equal));
    order
}

/// Bunch running cars into a 1.0s-gap train behind the leader (SC only).
#[inline]
fn bunch_field(cum: &mut [f64], retired: &[bool]) {
    let running: Vec<usize> = (0..cum.len()).filter(|&i| !retired[i]).collect();
    if running.len() <= 1 {
        return;
    }
    let order = order_by_cum(&running, cum);
    let sc_gap = 1.0;
    let lead_time = cum[order[0]];
    for (k, &c) in order.iter().enumerate() {
        cum[c] = lead_time + (k as f64) * sc_gap;
    }
}

/// Front-to-back overtaking / track-position resolution (green racing only).
#[inline]
fn resolve_positions(
    field: &Field,
    cum: &mut [f64],
    green_pace: &[f64],
    retired: &[bool],
    ov: &Overtake,
    drs_live: bool,
    rng: &mut Pcg64,
) {
    let running: Vec<usize> = (0..cum.len()).filter(|&i| !retired[i]).collect();
    if running.len() <= 1 {
        return;
    }
    let order = order_by_cum(&running, cum);
    for k in 1..order.len() {
        let ahead = order[k - 1];
        let cur = order[k];
        let gap = cum[cur] - cum[ahead];
        if gap >= ov.min_follow_gap {
            continue;
        }
        let pace_adv = green_pace[ahead] - green_pace[cur]; // >0 means cur faster
        let mut threshold = ov.threshold_s;
        if drs_live && gap <= ov.drs_window {
            threshold += ov.drs_bonus; // negative -> easier
        }
        threshold += ov.velocity_mod * field.top_speed_delta[cur];
        // Evaluate the coin flip in the same short-circuit order as Python:
        // `pace_adv >= threshold and rng.random() < p_overtake`. Python only
        // draws the uniform when the threshold is cleared, so we mirror that to
        // keep RNG consumption proportional.
        let passed = if pace_adv >= threshold {
            rng.gen::<f64>() < ov.p_overtake
        } else {
            false
        };
        if passed {
            cum[ahead] += ov.loser_penalty;
        } else {
            cum[cur] = cum[ahead] + ov.min_follow_gap + ov.duel_penalty;
            cum[ahead] += ov.duel_penalty * 0.5;
        }
    }
}

/// Finishing position (1-indexed) of focal car `fi`, by (laps desc, cum asc).
#[inline]
fn position_of(cum: &[f64], laps_done: &[i64], fi: usize) -> i64 {
    let li = laps_done[fi];
    let ci = cum[fi];
    let mut pos = 1i64;
    for j in 0..cum.len() {
        if j == fi {
            continue;
        }
        // key_j > key_i where key = (laps_done, -cum)
        let lj = laps_done[j];
        let cj = cum[j];
        let greater = lj > li || (lj == li && (-cj) > (-ci));
        if greater {
            pos += 1;
        }
    }
    pos
}

/// Run one scenario; return (focal_position, focal_total_time).
fn run_once(
    field: &Field,
    ov: &Overtake,
    regime: &[u8], // length n_laps+1, index 0 unused (regime[lap])
    n_laps: i64,
    focal_idx: usize,
    seed: u64,
) -> (i64, f64) {
    let n = field.n;
    let l = n_laps;

    // Stagger the start by grid slot: each car begins (grid-1)*grid_gap seconds
    // back, so track position is real (mirrors RaceSimulator.run_once).
    let mut cum: Vec<f64> = (0..n)
        .map(|i| (field.grid[i] - 1) as f64 * field.grid_gap)
        .collect();
    let mut age = vec![0.0f64; n];
    let mut compound: Vec<i64> = field.start_compound.clone();
    let mut retired = vec![false; n];
    let mut laps_done = vec![0i64; n];
    let mut green_pace = vec![0.0f64; n];

    let mut rng = Pcg64::seed_from_u64(seed);

    // Sample retirements up front, in car order, mirroring race.py exactly so
    // the RNG-consumption pattern (one uniform per car; an extra integer draw on
    // failure) matches the Python loop structure.
    let mut retire_lap = vec![l + 1; n];
    for i in 0..n {
        if rng.gen::<f64>() < field.team_failure_prob[i] {
            // rng.integers(1, L+1) -> uniform int in [1, L]
            let r: i64 = rng.gen_range(1..=l);
            retire_lap[i] = r;
        }
    }

    // Per-car normal samplers for lap noise.
    // (Normal::new only fails for sigma < 0; sigmas here are positive.)
    let normals: Vec<Normal<f64>> = (0..n)
        .map(|i| Normal::new(0.0, field.sigma[i].max(1e-12)).unwrap())
        .collect();

    let mut last_restart: i64 = -10;

    for lap in 1..=l {
        let reg = if (lap as usize) < regime.len() {
            regime[lap as usize]
        } else {
            0
        };
        // restart lap = first green lap immediately after a neutralisation
        let prev = if lap - 1 >= 1 && ((lap - 1) as usize) < regime.len() {
            regime[(lap - 1) as usize]
        } else {
            0
        };
        if reg == 0 && (prev == 1 || prev == 2) {
            last_restart = lap;
        }

        for i in 0..n {
            if retired[i] {
                continue;
            }
            if lap >= retire_lap[i] {
                retired[i] = true;
                cum[i] = f64::INFINITY;
                continue;
            }

            let gp = field.green_lap(i, compound[i], age[i], lap, l);
            green_pace[i] = gp;

            let mut lap_t;
            match reg {
                0 => {
                    // green: + Gaussian noise, age += 1
                    let noise = normals[i].sample(&mut rng);
                    lap_t = gp + noise;
                    age[i] += 1.0;
                }
                1 => {
                    // SC
                    lap_t = gp * field.lap_mult_sc[i];
                    age[i] += field.deg_mult_sc[i];
                }
                2 => {
                    // VSC
                    lap_t = gp * field.lap_mult_vsc[i];
                    age[i] += field.deg_mult_vsc[i];
                }
                _ => {
                    // RED flag: slow neutralised lap (same multiplier as SC);
                    // tyres are reset FOR FREE (handled below). No age increment.
                    lap_t = gp * field.lap_mult_sc[i];
                }
            }

            if let Some(new_comp) = field.pit_at(i, lap) {
                // A scheduled stop under a red flag is free (no pit loss).
                lap_t += if reg == 3 {
                    0.0
                } else {
                    field.effective_pit_loss(i, reg)
                };
                compound[i] = new_comp;
                age[i] = 0.0;
            } else if reg == 3 {
                // Free fresh tyres for everyone under a red flag.
                age[i] = 0.0;
            }

            cum[i] += lap_t;
            laps_done[i] = lap;
        }

        if reg == 1 || reg == 3 {
            // Field bunching under SC / red flag: compress to a train.
            bunch_field(&mut cum, &retired);
        }

        if reg == 0 {
            let drs_live = lap >= ov.drs_enable_lap && (lap - last_restart) > ov.drs_sc_delay;
            resolve_positions(field, &mut cum, &green_pace, &retired, ov, drs_live, &mut rng);
        }
    }

    let pos = position_of(&cum, &laps_done, focal_idx);
    (pos, cum[focal_idx])
}

/// Run K scenarios for a fixed field; return (positions, total_times).
///
/// Per-car arrays have length n. Per-compound arrays have length n*3 (row-major,
/// car-major; compound order [SOFT, MEDIUM, HARD]). The pit plan is a ragged set
/// of (lap, new_compound_idx) given by `pit_offsets` (len n+1) into the flat
/// `pit_laps` / `pit_comps`. Per-scenario inputs: `regimes` is a flat array of
/// length K*(n_laps+1) (0=green,1=SC,2=VSC,3=RED), `seeds` has length K.
#[pyfunction]
#[allow(clippy::too_many_arguments)]
fn simulate_batch(
    n_cars: usize,
    n_laps: i64,
    focal_idx: usize,
    // per-car
    base_pace: Vec<f64>,
    k_fuel: Vec<f64>,
    m_start: Vec<f64>,
    burn: Vec<f64>,
    sigma: Vec<f64>,
    top_speed_delta: Vec<f64>,
    cold_penalty: Vec<f64>,
    deg_mult_sc: Vec<f64>,
    deg_mult_vsc: Vec<f64>,
    lap_mult_sc: Vec<f64>,
    lap_mult_vsc: Vec<f64>,
    pit_loss: Vec<f64>,
    sc_saving_frac: Vec<f64>,
    vsc_saving_frac: Vec<f64>,
    team_failure_prob: Vec<f64>,
    // per-car per-compound (len n*3)
    k0: Vec<f64>,
    k1: Vec<f64>,
    k2: Vec<f64>,
    start_compound: Vec<i64>,
    grid: Vec<i64>,
    grid_gap: f64,
    // ragged pit plan
    pit_offsets: Vec<usize>,
    pit_laps: Vec<i64>,
    pit_comps: Vec<i64>,
    // circuit overtaking
    ov_threshold_s: f64,
    ov_velocity_mod: f64,
    ov_p_overtake: f64,
    ov_drs_bonus: f64,
    ov_drs_window: f64,
    ov_drs_enable_lap: i64,
    ov_drs_sc_delay: i64,
    ov_min_follow_gap: f64,
    ov_duel_penalty: f64,
    ov_loser_penalty: f64,
    // per-scenario
    regimes: Vec<u8>, // len K*(n_laps+1)
    seeds: Vec<u64>,  // len K
) -> PyResult<(Vec<i64>, Vec<f64>)> {
    let field = Field {
        n: n_cars,
        base_pace,
        k_fuel,
        m_start,
        burn,
        sigma,
        top_speed_delta,
        cold_penalty,
        deg_mult_sc,
        deg_mult_vsc,
        lap_mult_sc,
        lap_mult_vsc,
        pit_loss,
        sc_saving_frac,
        vsc_saving_frac,
        team_failure_prob,
        k0,
        k1,
        k2,
        start_compound,
        grid,
        grid_gap,
        pit_offsets,
        pit_laps,
        pit_comps,
    };
    let ov = Overtake {
        threshold_s: ov_threshold_s,
        velocity_mod: ov_velocity_mod,
        p_overtake: ov_p_overtake,
        drs_bonus: ov_drs_bonus,
        drs_window: ov_drs_window,
        drs_enable_lap: ov_drs_enable_lap,
        drs_sc_delay: ov_drs_sc_delay,
        min_follow_gap: ov_min_follow_gap,
        duel_penalty: ov_duel_penalty,
        loser_penalty: ov_loser_penalty,
    };

    let stride = (n_laps + 1) as usize;
    let k = seeds.len();
    let mut positions = vec![0i64; k];
    let mut times = vec![0.0f64; k];

    for s in 0..k {
        let regime = &regimes[s * stride..(s + 1) * stride];
        let (pos, t) = run_once(&field, &ov, regime, n_laps, focal_idx, seeds[s]);
        positions[s] = pos;
        times[s] = t;
    }

    Ok((positions, times))
}

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(ping, m)?)?;
    m.add_function(wrap_pyfunction!(simulate_batch, m)?)?;
    Ok(())
}
