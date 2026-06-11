"""Client for the Jolpica-Ergast F1 API (the community successor to the
shut-down ergast.com API).

We use Jolpica for *structured historical results*: classified finishing order,
per-lap times, and pit-stop durations. These are exactly the fields needed to
calibrate the lap-time / tyre-degradation model on real races. FastF1 (see
``fastf1_loader.py``) is used for the richer telemetry + stint/compound data.

Design notes
------------
* All responses are cached to disk (``data/cache/jolpica``) so repeated runs and
  calibration sweeps never re-hit the API. The cache key is the full request URL.
* Jolpica paginates with ``limit``/``offset`` (max ``limit`` is 100). Lap-time
  endpoints return ~1000+ rows per race, so we transparently page through them.
* Jolpica enforces rate limits (burst ~4 req/s, ~500 req/hr). We back off on HTTP
  429 with exponential delay and honour ``Retry-After`` when present.

The client is intentionally dependency-light (``requests`` + ``pandas``) so the
data layer has no heavy optional dependencies.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import pandas as pd
import requests

BASE_URL = "https://api.jolpi.ca/ergast/f1"
DEFAULT_CACHE = Path(__file__).resolve().parents[3] / "data" / "cache" / "jolpica"
MAX_LIMIT = 100  # Jolpica hard cap on page size


@dataclass(frozen=True)
class JolpicaConfig:
    base_url: str = BASE_URL
    cache_dir: Path = DEFAULT_CACHE
    timeout: float = 20.0
    max_retries: int = 5
    backoff_base: float = 1.5
    min_interval: float = 0.30  # seconds between requests (polite throttle)


class JolpicaClient:
    """Thin, cached, paginating client over the Jolpica-Ergast REST API."""

    def __init__(self, config: JolpicaConfig | None = None) -> None:
        self.cfg = config or JolpicaConfig()
        self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "pitwall/0.1 (strategy-engine)"})
        self._last_request = 0.0

    # -- low level ---------------------------------------------------------
    def _cache_path(self, url: str) -> Path:
        safe = (
            url.replace(self.cfg.base_url, "")
            .strip("/")
            .replace("/", "__")
            .replace("?", "__")
            .replace("&", "_")
            .replace("=", "-")
        )
        return self.cfg.cache_dir / f"{safe or 'root'}.json"

    def _throttle(self) -> None:
        dt = time.monotonic() - self._last_request
        if dt < self.cfg.min_interval:
            time.sleep(self.cfg.min_interval - dt)

    def _get(self, url: str, *, use_cache: bool = True) -> dict[str, Any]:
        cache_path = self._cache_path(url)
        if use_cache and cache_path.exists():
            return json.loads(cache_path.read_text())

        last_err: Exception | None = None
        for attempt in range(self.cfg.max_retries):
            self._throttle()
            try:
                resp = self._session.get(url, timeout=self.cfg.timeout)
                self._last_request = time.monotonic()
            except requests.RequestException as exc:  # network blip
                last_err = exc
                time.sleep(self.cfg.backoff_base ** attempt)
                continue

            if resp.status_code == 429:
                wait = float(resp.headers.get("Retry-After", self.cfg.backoff_base ** attempt))
                time.sleep(wait)
                continue
            if resp.status_code >= 500:
                time.sleep(self.cfg.backoff_base ** attempt)
                continue
            resp.raise_for_status()
            payload = resp.json()
            cache_path.write_text(json.dumps(payload))
            return payload

        raise RuntimeError(f"Jolpica request failed after retries: {url}") from last_err

    def _paged(self, path: str) -> Iterator[dict[str, Any]]:
        """Yield each MRData payload for a path, paging until exhausted."""
        offset = 0
        while True:
            sep = "&" if "?" in path else "?"
            url = f"{self.cfg.base_url}/{path}{sep}limit={MAX_LIMIT}&offset={offset}"
            payload = self._get(url)
            mr = payload["MRData"]
            yield mr
            total = int(mr.get("total", 0))
            offset += MAX_LIMIT
            if offset >= total:
                break

    # -- high level: results ----------------------------------------------
    def race_results(self, season: int, rnd: int) -> pd.DataFrame:
        """Classified finishing order for one race."""
        payload = self._get(f"{self.cfg.base_url}/{season}/{rnd}/results.json")
        races = payload["MRData"]["RaceTable"]["Races"]
        if not races:
            return pd.DataFrame()
        race = races[0]
        rows = []
        for r in race["Results"]:
            rows.append(
                {
                    "season": season,
                    "round": rnd,
                    "race_name": race["raceName"],
                    "circuit_id": race["Circuit"]["circuitId"],
                    "position": int(r["position"]),
                    "grid": int(r["grid"]),
                    "driver_id": r["Driver"]["driverId"],
                    "code": r["Driver"].get("code"),
                    "constructor": r["Constructor"]["constructorId"],
                    "status": r["status"],
                    "laps": int(r["laps"]),
                    "points": float(r["points"]),
                }
            )
        return pd.DataFrame(rows)

    # -- high level: lap times --------------------------------------------
    def lap_times(self, season: int, rnd: int) -> pd.DataFrame:
        """Per-driver, per-lap times for one race (transparently paginated).

        Returns columns: lap, driver_id, position, time_str, lap_time_s.
        """
        rows: list[dict[str, Any]] = []
        for mr in self._paged(f"{season}/{rnd}/laps.json"):
            races = mr["RaceTable"]["Races"]
            if not races:
                continue
            for lap in races[0].get("Laps", []):
                lap_no = int(lap["number"])
                for timing in lap["Timings"]:
                    rows.append(
                        {
                            "lap": lap_no,
                            "driver_id": timing["driverId"],
                            "position": int(timing["position"]),
                            "time_str": timing["time"],
                            "lap_time_s": _parse_lap_time(timing["time"]),
                        }
                    )
        df = pd.DataFrame(rows)
        return df.sort_values(["driver_id", "lap"]).reset_index(drop=True) if not df.empty else df

    # -- high level: pit stops --------------------------------------------
    def pit_stops(self, season: int, rnd: int) -> pd.DataFrame:
        """Pit-stop events: driver, lap, stop number, stationary duration."""
        rows: list[dict[str, Any]] = []
        for mr in self._paged(f"{season}/{rnd}/pitstops.json"):
            races = mr["RaceTable"]["Races"]
            if not races:
                continue
            for stop in races[0].get("PitStops", []):
                rows.append(
                    {
                        "driver_id": stop["driverId"],
                        "stop": int(stop["stop"]),
                        "lap": int(stop["lap"]),
                        "duration_s": _parse_seconds(stop.get("duration")),
                    }
                )
        return pd.DataFrame(rows)

    def season_schedule(self, season: int) -> pd.DataFrame:
        payload = self._get(f"{self.cfg.base_url}/{season}.json")
        races = payload["MRData"]["RaceTable"]["Races"]
        return pd.DataFrame(
            [
                {
                    "season": season,
                    "round": int(r["round"]),
                    "race_name": r["raceName"],
                    "circuit_id": r["Circuit"]["circuitId"],
                    "date": r.get("date"),
                }
                for r in races
            ]
        )


def _parse_lap_time(text: str) -> float:
    """'1:32.418' -> 92.418 seconds; '92.418' -> 92.418."""
    text = text.strip()
    if ":" in text:
        mins, secs = text.split(":")
        return int(mins) * 60 + float(secs)
    return float(text)


def _parse_seconds(text: str | None) -> float | None:
    if text is None:
        return None
    try:
        return _parse_lap_time(text)
    except ValueError:
        return None
