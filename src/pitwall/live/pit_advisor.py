"""Turn a live tyre belief into a pit-stop recommendation with uncertainty.

Given the Kalman filter's current posterior over [fresh pace, degradation rate],
this projects the remaining race and finds the pit lap that minimises remaining
time. Crucially it propagates the *degradation uncertainty* into a **credible
pit window** (by sampling the posterior), so the strategist sees not just "box
lap 34" but "box lap 32-36, tightening as the tyre talks to us".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .updater import TyreBelief


@dataclass
class PitAdvice:
    optimal_pit_lap: int
    window_lo: int            # 5th percentile pit lap across the belief
    window_hi: int            # 95th percentile
    stay_out_cost_curve: np.ndarray   # remaining time vs candidate pit lap
    candidate_laps: np.ndarray
    expected_gain_vs_now: float       # s saved by pitting at optimum vs pitting now


def _pit_loss_at(pit_lap: int, pit_loss: float, cheap_until: int | None,
                 cheap_pit_loss: float | None) -> float:
    """Pit loss for stopping on ``pit_lap``. A cheap stop (Safety Car / VSC) is
    only available up to ``cheap_until`` — the heart of the "pit now or lose it"
    decision; after the window it reverts to the full green-flag loss."""
    if cheap_until is not None and cheap_pit_loss is not None and pit_lap <= cheap_until:
        return cheap_pit_loss
    return pit_loss


def _remaining_time(
    pit_lap: int, current_lap: int, current_age: int, n_laps: int,
    a: float, b: float, fresh_pace: float, fresh_deg: float, pit_loss: float,
    cheap_until: int | None = None, cheap_pit_loss: float | None = None,
) -> float:
    """Deterministic remaining-stint time if pitting at end of ``pit_lap``:
    run the worn current tyre to ``pit_lap`` (extrapolating a + b*age), take the
    (possibly cheap) pit loss, then run a fresh set to the flag."""
    t = 0.0
    age = current_age
    for _ in range(current_lap + 1, pit_lap + 1):
        age += 1
        t += a + b * age
    t += _pit_loss_at(pit_lap, pit_loss, cheap_until, cheap_pit_loss)
    fresh_age = 0
    for _ in range(pit_lap + 1, n_laps + 1):
        fresh_age += 1
        t += fresh_pace + fresh_deg * fresh_age
    return t


def advise(
    belief: TyreBelief,
    *,
    current_lap: int,
    current_age: int,
    n_laps: int,
    pit_loss: float,
    fresh_pace: float | None = None,
    fresh_deg: float | None = None,
    cheap_until: int | None = None,
    cheap_pit_loss: float | None = None,
    n_samples: int = 400,
    seed: int = 0,
) -> PitAdvice:
    """Recommend a pit lap from the live belief, with a Bayesian window.

    ``fresh_pace``/``fresh_deg`` default to the believed fresh pace and a slightly
    gentler degradation. Pass ``cheap_until`` + ``cheap_pit_loss`` to model a
    Safety-Car window where the stop is cheap only for the next few laps — this is
    what flips the recommendation to "pit now"."""
    a, b = belief.fresh_pace, belief.deg_rate
    fp = fresh_pace if fresh_pace is not None else a
    fd = fresh_deg if fresh_deg is not None else max(0.0, b * 0.85)

    cand = np.arange(current_lap + 1, n_laps)  # must leave >=1 lap on fresh set
    if cand.size == 0:
        return PitAdvice(current_lap, current_lap, current_lap,
                         np.array([0.0]), np.array([current_lap]), 0.0)

    curve = np.array([
        _remaining_time(int(p), current_lap, current_age, n_laps, a, b, fp, fd,
                        pit_loss, cheap_until, cheap_pit_loss)
        for p in cand
    ])
    opt_idx = int(np.argmin(curve))
    opt_lap = int(cand[opt_idx])
    pit_now_cost = curve[0]
    gain = float(pit_now_cost - curve[opt_idx])

    # Bayesian window: sample (a, b) from the posterior, re-solve each draw.
    rng = np.random.default_rng(seed)
    try:
        draws = rng.multivariate_normal([a, b], belief.cov, size=n_samples)
    except np.linalg.LinAlgError:
        draws = np.column_stack([np.full(n_samples, a), np.full(n_samples, b)])
    opt_laps = []
    for ad, bd in draws:
        bd = max(0.0, bd)
        fdd = max(0.0, bd * 0.85)
        c = np.array([
            _remaining_time(int(p), current_lap, current_age, n_laps, ad, bd, fp, fdd,
                            pit_loss, cheap_until, cheap_pit_loss)
            for p in cand
        ])
        opt_laps.append(int(cand[int(np.argmin(c))]))
    lo, hi = int(np.percentile(opt_laps, 5)), int(np.percentile(opt_laps, 95))
    return PitAdvice(opt_lap, lo, hi, curve, cand, gain)
