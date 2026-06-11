"""Per-circuit constants.

Values are seeded from the research build-bible (pit-loss table; TUM 2019
``t_gap_overtake`` thresholds; per-circuit P(>=1 SC)) and approximate recent
pole/lap-count references. They are deliberate *priors*: the calibration harness
overrides pole pace and tyre coefficients from real FastF1 data per event.

``circuit_id`` matches the Ergast/Jolpica ``circuitId`` so a profile can be keyed
straight from a results pull.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CircuitProfile:
    circuit_id: str
    name: str
    n_laps: int
    pole_ref_s: float        # approx recent pole lap (s) - base-pace prior
    pit_loss_s: float        # green pit time loss vs staying out
    overtake_threshold_s: float  # t_gap_overtake (s/lap advantage to pass)
    p_safety_car: float      # P(>=1 SC) prior for this circuit


# pit_loss / threshold / p_sc grounded in research; pole_ref ~2023-24 approx.
CIRCUITS: dict[str, CircuitProfile] = {
    "bahrain":       CircuitProfile("bahrain", "Bahrain", 57, 89.7, 23.0, 1.38, 0.45),
    "jeddah":        CircuitProfile("jeddah", "Saudi Arabia", 50, 88.3, 19.0, 1.50, 0.70),
    "albert_park":   CircuitProfile("albert_park", "Australia", 58, 76.9, 20.0, 2.70, 0.55),
    "suzuka":        CircuitProfile("suzuka", "Japan", 53, 88.7, 22.0, 1.26, 0.40),
    "shanghai":      CircuitProfile("shanghai", "China", 56, 93.0, 22.0, 1.50, 0.40),
    "miami":         CircuitProfile("miami", "Miami", 57, 87.0, 19.0, 1.80, 0.55),
    "imola":         CircuitProfile("imola", "Emilia-Romagna", 63, 74.5, 28.0, 2.50, 0.30),
    "monaco":        CircuitProfile("monaco", "Monaco", 78, 71.0, 20.0, 3.75, 0.28),
    "villeneuve":    CircuitProfile("villeneuve", "Canada", 70, 72.0, 19.0, 3.75, 0.60),
    "catalunya":     CircuitProfile("catalunya", "Spain", 66, 72.2, 22.0, 2.31, 0.30),
    "red_bull_ring": CircuitProfile("red_bull_ring", "Austria", 71, 64.3, 20.0, 2.01, 0.35),
    "silverstone":   CircuitProfile("silverstone", "Britain", 52, 86.7, 23.0, 1.35, 0.40),
    "hungaroring":   CircuitProfile("hungaroring", "Hungary", 70, 76.0, 21.0, 2.42, 0.30),
    "spa":           CircuitProfile("spa", "Belgium", 44, 103.7, 19.0, 1.83, 0.45),
    "zandvoort":     CircuitProfile("zandvoort", "Netherlands", 72, 70.0, 22.0, 3.00, 0.50),
    "monza":         CircuitProfile("monza", "Italy", 53, 80.0, 24.0, 1.76, 0.27),
    "baku":          CircuitProfile("baku", "Azerbaijan", 51, 100.5, 20.0, 1.60, 0.86),
    "marina_bay":    CircuitProfile("marina_bay", "Singapore", 62, 90.0, 28.0, 3.75, 1.00),
    "americas":      CircuitProfile("americas", "USA (COTA)", 56, 94.5, 21.0, 1.83, 0.40),
    "rodriguez":     CircuitProfile("rodriguez", "Mexico", 71, 77.0, 21.0, 2.00, 0.40),
    "interlagos":    CircuitProfile("interlagos", "Brazil", 71, 70.0, 20.0, 1.60, 0.50),
    "vegas":         CircuitProfile("vegas", "Las Vegas", 50, 94.0, 19.0, 1.50, 0.70),
    "losail":        CircuitProfile("losail", "Qatar", 57, 80.0, 24.0, 1.80, 0.30),
    "yas_marina":    CircuitProfile("yas_marina", "Abu Dhabi", 58, 83.5, 20.0, 2.07, 0.30),
}

# Sensible fallback when a circuit is unknown.
DEFAULT_PROFILE = CircuitProfile(
    "unknown", "Unknown", 57, 85.0, 22.0, 1.80, 0.545
)


def get_profile(circuit_id: str) -> CircuitProfile:
    return CIRCUITS.get(circuit_id, DEFAULT_PROFILE)
