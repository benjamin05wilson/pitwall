"""Client for the OpenF1 API (https://openf1.org).

OpenF1 is the calibration backbone: unlike Ergast/Jolpica it exposes
**compound-labelled stints** (compound, lap range, true tyre age at stint start)
and per-lap durations + sector speeds. This is what makes genuine per-compound
tyre-degradation calibration possible. We pair it with Jolpica for pit-stop
durations and classified results.

(The richer FastF1 / official F1 live-timing feed is IP-blocked in some
environments; OpenF1 fills that gap with the fields the model actually needs.)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import requests

BASE_URL = "https://api.openf1.org/v1"
DEFAULT_CACHE = Path(__file__).resolve().parents[3] / "data" / "cache" / "openf1"


@dataclass(frozen=True)
class OpenF1Config:
    base_url: str = BASE_URL
    cache_dir: Path = DEFAULT_CACHE
    timeout: float = 30.0
    max_retries: int = 5
    min_interval: float = 0.25


class OpenF1Client:
    def __init__(self, config: OpenF1Config | None = None) -> None:
        self.cfg = config or OpenF1Config()
        self.cfg.cache_dir.mkdir(parents=True, exist_ok=True)
        self._session = requests.Session()
        self._session.headers.update({"User-Agent": "pitwall/0.1"})
        self._last = 0.0

    def _get(self, endpoint: str, params: dict[str, Any]) -> list[dict]:
        key = endpoint + "__" + "_".join(f"{k}-{v}" for k, v in sorted(params.items()))
        cache = self.cfg.cache_dir / f"{key}.json"
        if cache.exists():
            return json.loads(cache.read_text())
        url = f"{self.cfg.base_url}/{endpoint}"
        for attempt in range(self.cfg.max_retries):
            dt = time.monotonic() - self._last
            if dt < self.cfg.min_interval:
                time.sleep(self.cfg.min_interval - dt)
            try:
                resp = self._session.get(url, params=params, timeout=self.cfg.timeout)
                self._last = time.monotonic()
            except requests.RequestException:
                time.sleep(1.5 ** attempt)
                continue
            if resp.status_code == 429 or resp.status_code >= 500:
                time.sleep(float(resp.headers.get("Retry-After", 1.5 ** attempt)))
                continue
            if resp.status_code == 404:
                return []  # missing session/year — treat as no data, not an error
            resp.raise_for_status()
            data = resp.json()
            cache.write_text(json.dumps(data))
            return data
        raise RuntimeError(f"OpenF1 request failed: {url} {params}")

    # -- session lookup ----------------------------------------------------
    def session_key(self, year: int, country: str, session_name: str = "Race") -> int:
        rows = self._get("sessions", {"year": year, "country_name": country,
                                      "session_name": session_name})
        if not rows:
            raise ValueError(f"No OpenF1 session for {year} {country} {session_name}")
        return int(rows[0]["session_key"])

    def session_info(self, year: int, country: str, session_name: str = "Race") -> dict:
        rows = self._get("sessions", {"year": year, "country_name": country,
                                      "session_name": session_name})
        return rows[0] if rows else {}

    # -- core tables -------------------------------------------------------
    def stints(self, session_key: int) -> pd.DataFrame:
        rows = self._get("stints", {"session_key": session_key})
        df = pd.DataFrame(rows)
        return df.sort_values(["driver_number", "stint_number"]).reset_index(drop=True) if len(df) else df

    def laps(self, session_key: int) -> pd.DataFrame:
        rows = self._get("laps", {"session_key": session_key})
        df = pd.DataFrame(rows)
        return df.sort_values(["driver_number", "lap_number"]).reset_index(drop=True) if len(df) else df

    def drivers(self, session_key: int) -> pd.DataFrame:
        return pd.DataFrame(self._get("drivers", {"session_key": session_key}))
