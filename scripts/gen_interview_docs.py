"""Assemble INTERVIEW_QA.md and MODELLING_DECISIONS.md from the adversarially
interview-prep workflow output. Requires external workflow JSON; not an experiment runner."""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

SRC = sys.argv[1] if len(sys.argv) > 1 else None
ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
DOCS.mkdir(exist_ok=True)

if SRC is None:
    raise SystemExit("Usage: python scripts/gen_interview_docs.py WORKFLOW.json (external result.dimensions input; overwrites docs)")
data = json.load(open(SRC))
dims = data.get("result", data)["dimensions"]

BADGE = {"warm-up": "warm-up", "core": "core", "killer": "KILLER"}


def slug(s: str) -> str:
    return s.split("—")[0].split(":")[0].strip()


def gh_anchor(s: str) -> str:
    """GitHub heading-anchor slug: lowercase, drop punctuation, spaces -> hyphens."""
    return re.sub(r"[^a-z0-9 \-]", "", s.lower()).replace(" ", "-")


# --------------------------------------------------------------------------- #
# 1) INTERVIEW Q&A                                                             #
# --------------------------------------------------------------------------- #
qa_lines: list[str] = []
qa_lines.append("# Pitwall — interview Q&A\n")
qa_lines.append(
    "Anticipated questions from a Williams strategy/modelling panel, each with a "
    "tight, defensible answer grounded in the actual implementation. Generated per "
    "area by agents that read the code, then hardened by an adversarial panel that "
    "rewrote any hand-wavy answer and added the follow-up the first pass missed.\n")
qa_lines.append("## Before you walk in — three things\n")
qa_lines.append(
    "1. **Lead with the safety-car reactions, not Norris.** The Norris/2023-Bahrain "
    "`+7` is partly a broken-real-car artefact (his real race was a mechanical 6-stop), "
    "so it compares a healthy AI car to a broken real one. Headline the **Canada / "
    "Monaco safety-car cases** — clean cars, pure strategy, the AI taking the cheap "
    "stop on the right lap. Keep Norris as an *extreme-case* aside.\n")
qa_lines.append(
    "2. **Volunteer the limitations before they're found.** Raising the inert "
    "temperature coefficient, the SC-saving prior, and the thin wet sample *first* "
    "reads as senior; getting caught reads as junior. See `MODELLING_DECISIONS.md`.\n")
qa_lines.append(
    "3. **Demo from recordings, not a live click-through.** A server hiccup mid-demo "
    "is worse than no demo. Pre-record the on-track ghost and the Monaco wet crossover.\n")
qa_lines.append("\n---\n")

toc = []
for d in dims:
    name = slug(d["dimension"])
    toc.append(f"- [{name}](#{gh_anchor(name)})")
qa_lines.append("## Contents\n\n" + "\n".join(toc) + "\n\n---\n")

for d in dims:
    name = slug(d["dimension"])
    qa_lines.append(f"\n## {name}\n")
    # warm-up, core, killer order
    order = {"warm-up": 0, "core": 1, "killer": 2}
    for qa in sorted(d["final_qa"], key=lambda q: order.get(q["difficulty"], 1)):
        qa_lines.append(f"\n**Q ({BADGE.get(qa['difficulty'], qa['difficulty'])}). "
                        f"{qa['question'].strip()}**\n")
        qa_lines.append(f"{qa['answer'].strip()}\n")
        fu = (qa.get("follow_up") or "").strip()
        if fu and fu.lower() not in ("", "none", "n/a"):
            qa_lines.append(f"\n> *Likely follow-up:* {fu}\n")
    qa_lines.append("\n---\n")

(DOCS / "INTERVIEW_QA.md").write_text("\n".join(qa_lines))


# --------------------------------------------------------------------------- #
# 2) MODELLING DECISIONS & LIMITATIONS                                         #
# --------------------------------------------------------------------------- #
md: list[str] = []
md.append("# Pitwall — modelling decisions & limitations\n")
md.append(
    "The reasoning behind the engine's modelling and engineering choices, and an "
    "honest register of what it does *not* yet do. The limitations section is "
    "deliberately the longest: knowing exactly where the model is weak — and saying "
    "so before being asked — is the point.\n")

md.append("\n## Limitations to volunteer first\n")
md.append(
    "The handful a sharp reviewer *will* find. Raise them yourself, framed as "
    "\"here's the boundary and here's what would move it\":\n")
VOLUNTEER = [
    ("Temperature → degradation coupling is built but shipped **inert** (`temp_coeff = 0.0`).",
     "The two-stage S/M/H fit returned a wrong-signed slope (r = -0.55 over 45 points). Shipping "
     "a wrong-signed coefficient is strictly worse than zero — it would invert reality (steepen "
     "wear on cool tracks). The plumbing is live and native-consistent, so reactivation is a "
     "one-line swap, not a rebuild.",
     "Pirelli per-compound construction data + tyre-surface-temperature telemetry; fit within a "
     "single construction year rather than pooling across Pirelli re-specs."),
    ("The SC/VSC cheap-stop discount is the **published prior** (0.50 / 0.35), not a data fit.",
     "`PitOut - PitIn` measures pit-lane transit time, which barely changes under a safety car; the "
     "time *saved* is time the field loses crawling, which that telemetry can't see. So the quantity "
     "is computed but honestly not shipped — it isn't the estimand. This is the single biggest "
     "unverified lever on undercut/overcut value, and I flag it as such.",
     "Field-synchronous gap reconstruction (all 20 cars on a common clock), or a ghost-style sim "
     "estimate of time-lost-to-field under SC vs green, validated on real neutralised stops."),
    ("The Norris/Bahrain `+7` headline is partly a **broken-real-car artefact**.",
     "His real race was a mechanical 6-stop, so the gain mixes 'remove the failure' with 'better "
     "strategy'. The clean pure-strategy gains (the safety-car cases) are smaller — 1 to 3 places — "
     "and those are the honest demo. The ghost reports the sim finish *and* the actual finish so the "
     "comparison is transparent.",
     "Headline the clean cases; keep Norris as an extreme illustration only."),
    ("The overtaking / track-position model still uses **uncalibrated 2019 TUM constants** "
     "(`p_overtake = 0.25`, thresholds).",
     "Everything *strategy-facing* is calibrated (deg, pace, pit loss, SC probability, dirty air, wet "
     "pace); the overtaking layer is the part still on literature priors. It affects how cleanly "
     "track-position trades resolve, not the core stint maths.",
     "A logistic P(pass | pace_delta, gap, circuit) fit from position+gap+DRS telemetry — a clean "
     "sklearn/PyTorch surrogate."),
    ("Wet calibration rests on **2 races / 31 full-wet laps** (the intermediate floor is firmer at "
     "471 laps).",
     "Each race's own dry laps cancel circuit/setup, and the *ordering* validates externally — the "
     "wet DP calls Verstappen's real inter switch to the lap at Zandvoort (L62) and Monaco (L56). "
     "Sample sizes are surfaced (`n_wet = 31`) so the magnitude is never over-trusted.",
     "Accumulate future full-wet races; weight the full-wet floor toward the better-sampled "
     "intermediate floor via the physical bowl until more data exists."),
    ("Dirty-air partial r-values are **modest (~0.13–0.29)**.",
     "Wake depends on unobserved aero setup and driver inputs (steering, lift-and-coast, DRS), so low "
     "r is expected. What matters for strategy is the sign and the cross-circuit ordering, and both "
     "are physically correct (Monaco 0.83 ≫ Monza 0.13) on large held-lap counts.",
     "Team CFD wake maps, or telemetry features (throttle/brake/DRS/lateral position) to lift the "
     "explained variance."),
    ("No formal **confidence intervals** on the calibrated priors; the sim carries a few-positions "
     "**absolute** error.",
     "That's why the ghost reports the real-vs-AI *delta* under common random numbers as the "
     "apples-to-apples quantity, not the absolute sim finish. Backtest Spearman is ~0.72 — good for "
     "ordering, not a claim of point precision.",
     "Bootstrap/Bayesian CIs on each calibrated parameter; report them in `priors.json` alongside n."),
]
for lim, why, fix in VOLUNTEER:
    md.append(f"\n### {lim}\n")
    md.append(f"- **Why it's acceptable:** {why}")
    md.append(f"- **What would fix it:** {fix}\n")

md.append("\n---\n")
md.append("\n## Key decisions, by area\n")
for d in dims:
    name = slug(d["dimension"])
    md.append(f"\n### {name}\n")
    for dec in d["decisions"]:
        md.append(f"- {dec.strip()}")
    md.append("")

md.append("\n---\n")
md.append("\n## Full limitations register\n")
md.append("Every limitation surfaced during the review, by area, for completeness.\n")
for d in dims:
    name = slug(d["dimension"])
    md.append(f"\n### {name}\n")
    for lm in d["limitations"]:
        md.append(f"- **{lm['limitation'].strip()}**")
        md.append(f"  - *Acceptable because:* {lm['why_acceptable'].strip()}")
        md.append(f"  - *Fixed by:* {lm['what_fixes_it'].strip()}")
    md.append("")

(DOCS / "MODELLING_DECISIONS.md").write_text("\n".join(md))

qa_n = sum(len(d["final_qa"]) for d in dims)
lim_n = sum(len(d["limitations"]) for d in dims)
dec_n = sum(len(d["decisions"]) for d in dims)
print(f"Wrote docs/INTERVIEW_QA.md ({qa_n} Q&A) and docs/MODELLING_DECISIONS.md "
      f"({dec_n} decisions, {lim_n} limitations)")
