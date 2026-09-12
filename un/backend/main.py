from __future__ import annotations

import asyncio
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
    def __init__(self) -> None:
        self.scans = self.hits = self.false_alarms = self.reward = 0
        self.emitter_first_seen: dict[str, int] = {}
        self.intercept_delays: list[int] = []

    def update(self, band: int, observation: dict) -> tuple[bool, str | None]:
        self.scans += 1
        emitters = observation["active_emitters"]
        for e in emitters:
            self.emitter_first_seen.setdefault(e["id"], observation["step"])
        hit_emitters = [e for e in emitters if e["band_id"] == band]
        hit = bool(hit_emitters)
        if hit:
            self.hits += 1
            self.reward += 10
            emitter = hit_emitters[0]
            first = self.emitter_first_seen.pop(emitter["id"], observation["step"])
            self.intercept_delays.append(observation["step"] - first)
            return True, emitter["id"]
        # A false alarm is only counted when a quiet ground-truth band has anomalously high noise.
        if observation["bands"]["power_dbm"][band] > -96:
            self.false_alarms += 1
            self.reward -= 2
        else:
            self.reward -= 1
        return False, None

    def cumulative(self) -> dict[str, float | int]:
        rate = self.hits / self.scans if self.scans else 0.0
        return {
            "detection_prob": round(rate * 100, 1),
            "intercept_rate": round(rate * 100, 1),
            "false_alarms": self.false_alarms,
            "avg_delay": round(sum(self.intercept_delays) / len(self.intercept_delays), 2) if self.intercept_delays else 0.0,
            "reward": self.reward,
            "correct_predictions": round(rate * 100, 1),
        }


class Simulation:
    def __init__(self) -> None:
        self.environment = RFEnvironment()
        self.open_loop, self.smart_scan = OpenLoopScheduler(), SmartScanScheduler()
        self.open_stats, self.smart_stats = EngineStats(), EngineStats()
        self.compare_every = config.COMPARE_EVERY_N_STEPS
        self.history: deque[dict[str, Any]] = deque(maxlen=80)
        self.clients: set[WebSocket] = set()

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
        return {
            "step": obs["step"], "total_steps": config.TOTAL_STEPS,
            "sim_time": f"00:{obs['step'] // 600:02d}:{(obs['step'] // 10) % 60:02d}",
            "ground_truth": {"active_emitters": obs["active_emitters"]},
            "open_loop": {"dwelling_band": open_band, "hit": open_hit, "cumulative": open_cumulative},
            "smart_scan": {"dwelling_band": smart_band, "hit": smart_hit, "cumulative": smart_cumulative},
            "event": event, "comparison_history": list(self.history), "compare_every": self.compare_every,
            "data_source": {"name": obs["source_name"], "mode": obs["source_mode"]},
            "sinr_db": round(8 + (smart_cumulative["intercept_rate"] / 8), 1),
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
