"""Central runtime configuration.

Replace scheduler code only in schedulers/open_loop.py and schedulers/smart_scan.py.
Set these values (or matching environment variables) when integrating real policies.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

OPEN_LOOP_IMPL = os.getenv("OPEN_LOOP_IMPL", "round_robin_placeholder")
SMART_SCAN_MODEL_PATH = os.getenv("SMART_SCAN_MODEL_PATH", "")
DATA_SOURCE = Path(os.getenv("DATA_SOURCE", str(ROOT / "rf_dataset.csv")))
COMPARE_EVERY_N_STEPS = max(10, int(os.getenv("COMPARE_EVERY_N_STEPS", "50")))
TICK_HZ = 10
TOTAL_STEPS = int(os.getenv("TOTAL_STEPS", "2000"))
BAND_COUNT = 36
BAND_BASE_GHZ = 0.5
BAND_STEP_GHZ = 0.5
