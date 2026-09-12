"""Vectorized 36-band RF ground-truth source with optional CSV replay."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import config

REQUIRED_COLUMNS = {
    "timestep", "band_id", "freq_ghz", "is_transmitting", "power_dbm", "aoa_deg",
    "pulse_width_us", "emitter_id", "emitter_class", "threat_level",
}
CLASSES = np.array(["Radar", "Comm", "DataLink", "Navigation", "ElectronicSupport", "Unknown"])
THREAT_LEVELS = np.array(["Low", "Medium", "High"])


@dataclass
class ReplayRow:
    timestep: int
    band_id: int
    freq_ghz: float
    is_transmitting: bool
    power_dbm: float
    aoa_deg: float
    pulse_width_us: float
    emitter_id: str
    emitter_class: str
    threat_level: str


class RFEnvironment:
    """Returns only ground truth; it has no scheduler or learning knowledge."""

    def __init__(self, csv_path: Path | None = None, band_count: int = config.BAND_COUNT) -> None:
        self.band_count = band_count
        self.step = 0
        self.rng = np.random.default_rng(26055)
        self.csv_path = csv_path or config.DATA_SOURCE
        self.replay: dict[int, list[ReplayRow]] = self._load_csv(self.csv_path)
        self.source_name = self.csv_path.name if self.replay else "synthetic_rf_environment"
        self.source_mode = "LIVE REPLAY" if self.replay else "SYNTHETIC"
        self._active = np.zeros(band_count, dtype=bool)
        self._power = np.full(band_count, -104.0)
        self._aoa = self.rng.uniform(0, 360, band_count)
        self._class_idx = self.rng.integers(0, len(CLASSES), band_count)
        self._threat_idx = self.rng.integers(0, 3, band_count)
        self._emitter_num = np.arange(100, 100 + band_count)

    def _load_csv(self, path: Path) -> dict[int, list[ReplayRow]]:
        if not path.is_file():
            return {}
        try:
            raw = np.genfromtxt(path, delimiter=",", names=True, dtype=None, encoding="utf-8")
            if raw.dtype.names is None or not REQUIRED_COLUMNS.issubset(raw.dtype.names):
                return {}
            rows: dict[int, list[ReplayRow]] = {}
            for value in np.atleast_1d(raw):
                row = ReplayRow(
                    timestep=int(value["timestep"]), band_id=int(value["band_id"]),
                    freq_ghz=float(value["freq_ghz"]), is_transmitting=str(value["is_transmitting"]).lower() in {"1", "true", "yes"},
                    power_dbm=float(value["power_dbm"]), aoa_deg=float(value["aoa_deg"]),
                    pulse_width_us=float(value["pulse_width_us"]), emitter_id=str(value["emitter_id"]),
                    emitter_class=str(value["emitter_class"]), threat_level=str(value["threat_level"]),
                )
                rows.setdefault(row.timestep, []).append(row)
            return rows
        except (OSError, ValueError, TypeError):
            return {}

    def _frequency(self, band_id: int) -> float:
        return config.BAND_BASE_GHZ + band_id * config.BAND_STEP_GHZ

    def _synthetic_tick(self) -> list[dict[str, Any]]:
        # Markov-like persistence gives emitters realistic dwell time, while NumPy generates all bands at once.
        starts = self.rng.random(self.band_count) < 0.07
        stops = self.rng.random(self.band_count) < 0.15
        self._active = np.where(self._active, ~stops, starts)
        if not self._active.any():
            self._active[self.rng.integers(0, self.band_count)] = True
        retunes = self.rng.random(self.band_count) < 0.08
        self._aoa = (self._aoa + self.rng.normal(0, 1.8, self.band_count)) % 360
        self._class_idx = np.where(retunes, self.rng.integers(0, len(CLASSES), self.band_count), self._class_idx)
        self._threat_idx = np.where(retunes, self.rng.choice([0, 1, 2], self.band_count, p=[0.58, 0.28, 0.14]), self._threat_idx)
        self._power = np.where(self._active, self.rng.normal(-42, 12, self.band_count), self.rng.normal(-104, 3.5, self.band_count))
        active_ids = np.flatnonzero(self._active)
        return [
            {
                "id": f"E-{self._emitter_num[i]:03d}", "band_id": int(i), "freq_ghz": round(self._frequency(int(i)), 2),
                "power_dbm": round(float(self._power[i]), 1), "aoa_deg": round(float(self._aoa[i]), 1),
                "pulse_width_us": round(float(self.rng.uniform(0.4, 18)), 2), "class": str(CLASSES[self._class_idx[i]]),
                "threat_level": str(THREAT_LEVELS[self._threat_idx[i]]),
            }
            for i in active_ids
        ]

    def advance(self) -> dict[str, Any]:
        self.step += 1
        if self.replay:
            # Replay dataset accurately, returning empty list if no emitters transmit at this step.
            # Wrap around when step exceeds max recorded step.
            max_step = max(self.replay.keys())
            eff_step = ((self.step - 1) % max_step) + 1
            rows = self.replay.get(eff_step, [])
            emitters = [
                {"id": row.emitter_id, "band_id": row.band_id, "freq_ghz": row.freq_ghz, "power_dbm": row.power_dbm,
                 "aoa_deg": row.aoa_deg, "pulse_width_us": row.pulse_width_us, "class": row.emitter_class,
                 "threat_level": row.threat_level}
                for row in rows if row.is_transmitting
            ]
        else:
            emitters = self._synthetic_tick()
        transmitting = np.zeros(self.band_count, dtype=bool)
        powers = self.rng.normal(-104, 3.5, self.band_count)
        for emitter in emitters:
            if 0 <= emitter["band_id"] < self.band_count:
                transmitting[emitter["band_id"]] = True
                powers[emitter["band_id"]] = emitter["power_dbm"]
        return {
            "step": self.step,
            "bands": {"is_transmitting": transmitting.tolist(), "power_dbm": powers.round(1).tolist()},
            "active_emitters": emitters,
            "source_name": self.source_name,
            "source_mode": self.source_mode,
        }
