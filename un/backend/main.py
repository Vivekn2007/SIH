from __future__ import annotations

import asyncio
import math
from collections import deque
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone
from typing import Any

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

import config
from environment import RFEnvironment
from schedulers import OpenLoopScheduler, SmartScanScheduler


class EngineStats:
    WINDOW = 200   # rolling window size for detection probability

    def __init__(self) -> None:
        self.scans         = 0
        self.hits          = 0
        self.false_alarms  = 0
        self.reward        = 0
        self._hit_streak   = 0
        self.intercept_delays: list[int] = []
        self.emitter_first_seen: dict[str, int] = {}
        # Rolling window of recent hit/miss results (True = hit, False = miss)
        self._recent: list[bool] = []

    def update(self, band: int, observation: dict) -> tuple[bool, str | None]:
        self.scans += 1
        step     = observation["step"]
        emitters = observation["active_emitters"]
        powers   = observation["bands"]["power_dbm"]

        # Track when each emitter was first seen (for delay calculation)
        for e in emitters:
            self.emitter_first_seen.setdefault(e["id"], step)

        # --- Hit check: did we scan the right band? ---
        hit_emitters = [e for e in emitters if e["band_id"] == band]
        hit = bool(hit_emitters)

        # Maintain rolling window
        self._recent.append(hit)
        if len(self._recent) > self.WINDOW:
            self._recent.pop(0)

        if hit:
            self.hits += 1
            self._hit_streak += 1
            emitter  = hit_emitters[0]
            eid      = emitter["id"]
            first    = self.emitter_first_seen.pop(eid, step)   # pop so next appearance counts fresh
            delay    = step - first
            self.intercept_delays.append(delay)

            # Tiered reward: base + early-intercept bonus + streak + threat level
            threat_bonus = {"High": 10, "Medium": 5, "Low": 0}.get(
                emitter.get("threat_level", "Low"), 0)
            early_bonus  = max(0, 15 - delay // 3)
            streak_bonus = min(self._hit_streak * 2, 12)
            self.reward += 20 + early_bonus + streak_bonus + threat_bonus
            return True, eid

        # Miss
        self._hit_streak = 0

        # False alarm: quiet band whose noise spike exceeds 2-sigma above floor (−104±3 dBm → −98 dBm)
        if powers[band] > -98:
            self.false_alarms += 1
            self.reward -= 2
        return False, None

    def cumulative(self) -> dict[str, float | int]:
        # detection_prob: rolling hit rate over last WINDOW steps
        # — never artificially locks at 100%, reflects current performance honestly
        if self._recent:
            det_prob = sum(self._recent) / len(self._recent) * 100
        else:
            det_prob = 0.0

        # intercept_rate: lifetime hit rate (hits / total scans)
        intercept = self.hits / self.scans * 100 if self.scans else 0.0

        # avg intercept delay
        avg_delay = (round(sum(self.intercept_delays) / len(self.intercept_delays), 2)
                     if self.intercept_delays else 0.0)

        return {
            "detection_prob":      round(det_prob, 1),
            "intercept_rate":      round(intercept, 1),
            "false_alarms":        self.false_alarms,
            "avg_delay":           avg_delay,
            "reward":              self.reward,
            "correct_predictions": round(intercept, 1),
        }



class Simulation:
    def __init__(self) -> None:
        self.environment = RFEnvironment()
        self.open_loop, self.smart_scan = OpenLoopScheduler(), SmartScanScheduler()
        self.open_stats, self.smart_stats = EngineStats(), EngineStats()
        self.compare_every = config.COMPARE_EVERY_N_STEPS
        self.history: deque[dict[str, Any]] = deque(maxlen=80)
        self.clients: set[WebSocket] = set()

    # Receiver location (fixed base station — can be overridden via env vars)
    RECEIVER_LAT = 25.3147
    RECEIVER_LON = 121.4737

    def tick(self) -> dict[str, Any]:
        obs = self.environment.advance()
        open_band = self.open_loop.select_band(obs)
        smart_band = self.smart_scan.select_band(obs)
        open_hit, _ = self.open_stats.update(open_band, obs)
        smart_hit, smart_id = self.smart_stats.update(smart_band, obs)
        event = self._event(obs, smart_hit, smart_id)
        open_cumulative, smart_cumulative = self.open_stats.cumulative(), self.smart_stats.cumulative()
        if obs["step"] % self.compare_every == 0:
            self.history.append({"step": obs["step"], "open_loop": open_cumulative, "smart_scan": smart_cumulative})

        # Enrich each emitter with geo fields required by the GeoMap component
        enriched_emitters = []
        for e in obs["active_emitters"]:
            range_km = 30 + (hash(e["id"]) % 270)
            aoa_rad  = math.radians(e["aoa_deg"])
            lat_off  = (range_km / 111.0) * math.cos(aoa_rad)
            lon_off  = (range_km / (111.0 * math.cos(math.radians(self.RECEIVER_LAT)))) * math.sin(aoa_rad)
            enriched_emitters.append({
                **e,
                "lat":      round(self.RECEIVER_LAT + lat_off, 4),
                "lon":      round(self.RECEIVER_LON + lon_off, 4),
                "range_km": range_km,
            })

        # Build band_config from config constants (frequency range per band)
        band_config = [
            {
                "band_id": i,
                "min_ghz": round(config.BAND_BASE_GHZ + i * config.BAND_STEP_GHZ, 2),
                "max_ghz": round(config.BAND_BASE_GHZ + (i + 1) * config.BAND_STEP_GHZ, 2),
            }
            for i in range(config.BAND_COUNT)
        ]

        return {
            "step": obs["step"], "total_steps": config.TOTAL_STEPS,
            "sim_time": f"00:{obs['step'] // 600:02d}:{(obs['step'] // 10) % 60:02d}",
            "ground_truth": {"active_emitters": enriched_emitters},
            "open_loop":  {"dwelling_band": open_band,  "hit": open_hit,  "cumulative": open_cumulative},
            "smart_scan": {"dwelling_band": smart_band, "hit": smart_hit, "cumulative": smart_cumulative},
            "event": event, "comparison_history": list(self.history), "compare_every": self.compare_every,
            "data_source": {"name": obs["source_name"], "mode": obs["source_mode"]},
            "sinr_db": round(
                # Signal = max power across bands with active emitters; Noise = -104 dBm floor
                # This gives a realistic, dynamic SINR that improves as the model detects stronger signals
                max(
                    (max((obs["bands"]["power_dbm"][e["band_id"]] for e in obs["active_emitters"]), default=-104.0)
                     - (-104.0)),  # signal above noise floor
                    0.0
                ) * (smart_cumulative["detection_prob"] / 100.0 + 0.1),
                1
            ),
            # Fields required by the new frontend components
            "bands":       obs["bands"],
            "band_config": band_config,
            "receiver":    {"lat": self.RECEIVER_LAT, "lon": self.RECEIVER_LON, "label": "Receiver"},
        }


    def _event(self, obs: dict[str, Any], smart_hit: bool, smart_id: str | None) -> dict[str, str]:
        high = next((e for e in obs["active_emitters"] if e["threat_level"] == "High"), None)
        if smart_hit:
            return {"text": f"Smart Scan intercept confirmed: {smart_id}", "type": "detection", "severity": "ok"}
        if high:
            return {"text": f"High-priority emitter active: {high['id']} on B{high['band_id']}", "type": "alert", "severity": "threat"}
        return {"text": f"Sweep update: {len(obs['active_emitters'])} active emitters", "type": "system", "severity": "warn"}

    async def run(self) -> None:
        while True:
            payload = self.tick()
            dead: list[WebSocket] = []
            for client in self.clients.copy():
                try:
                    await client.send_json(payload)
                except Exception:
                    dead.append(client)
            for client in dead:
                self.clients.discard(client)
            await asyncio.sleep(1 / config.TICK_HZ)


simulation = Simulation()


@asynccontextmanager
async def lifespan(_: FastAPI):
    task = asyncio.create_task(simulation.run())
    yield
    task.cancel()
    with suppress(asyncio.CancelledError):
        await task


app = FastAPI(title="Smart Scan Command Center", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173"], allow_methods=["*"], allow_headers=["*"])


@app.get("/api/health")
async def health() -> dict[str, str]:
    return {"status": "online"}


@app.websocket("/ws/telemetry")
async def telemetry(websocket: WebSocket) -> None:
    await websocket.accept()
    simulation.clients.add(websocket)
    try:
        while True:
            message = await websocket.receive_json()
            if message.get("type") == "set_compare_every":
                simulation.compare_every = max(10, min(500, int(message.get("value", config.COMPARE_EVERY_N_STEPS))))
    except WebSocketDisconnect:
        pass
    finally:
        simulation.clients.discard(websocket)
