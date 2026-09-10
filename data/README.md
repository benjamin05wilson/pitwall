# Data provenance and redistribution notes

The offline demonstration uses only authored synthetic fixtures under `fixtures/` and circuit/default parameters in source. It does not read the public-data caches below.

| Tracked files | Source indicated by loader/artifact | Provenance limits |
|---|---|---|
| `cache/openf1/*.json` | OpenF1 API; request parameters encoded in filenames; Bahrain session/lap/stint/driver responses | Fetch timestamps, response licence notices and content hashes were not recorded by the original cache writer. |
| `cache/jolpica/*.json` | Jolpica Ergast-compatible API; year/round/endpoint/pagination encoded in filenames | Historical calendars/results and some lap/pit data; original fetch dates and redistribution grants are not retained. |
| `calibrated/bahrain.json` | Derived calibration summary written by Pitwall calibration code | In-sample summary, incomplete generating environment/command provenance. |
| `../src/pitwall/data/calibrated/priors.json` | Derived public-data priors from the build-priors workflow | Contains fit/source notes but not a complete immutable raw-input manifest. |

The code's MIT licence is preserved. These third-party factual/data caches are separately sourced; this repository does not assert that its code licence grants rights over upstream data. Original upstream attribution is retained here and in [OpenF1 loader](../src/pitwall/data/openf1.py) and [Jolpica loader](../src/pitwall/data/jolpica.py). Redistribution permission/status was not independently established during this pass; do not present bundled caches as an independently licensed dataset. New redistributions should establish applicable upstream terms and provenance. No third-party cache was added or modified for the demonstration.

The small synthetic fixture explicitly records its generating formula and planted anomaly. Its generated outputs are illustrations of this code, not substituted evidence for prior real-data claims.
