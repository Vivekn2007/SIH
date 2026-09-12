from .base import SchedulerPolicy

import os
import torch
import numpy as np
from .base import SchedulerPolicy
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from ew_smart_scan_rl import DQNScheduler, EWState, PeriodicEmitterTracker
except ImportError:
    DQNScheduler, EWState, PeriodicEmitterTracker = None, None, None


class SmartScanScheduler(SchedulerPolicy):
    """
    Hybrid Smart Scan Scheduler — optimised for high detection probability.

    Strategy:
      Phase 1 (0-35):  Full deterministic sweep — every band visited once per
                       macrocycle so no emitter is permanently missed.
      Phase 2 (36-149): Intelligent tracking using a three-layer score:
        Layer A — Observed power      (instant reaction to strong signals)
        Layer B — Periodic urgency    (predict when each emitter will pulse next)
        Layer C — DQN Q-values        (learned long-term policy)

    Adaptive Dwell:
      • If current band is ACTIVE (emitter detected) → stay up to MAX_ACTIVE_DWELL steps
        (exploit the confirmed hit — keep tracking the emitter)
      • If current band is QUIET → move after just MIN_QUIET_DWELL steps
        (don't waste time on empty bands, go find the next emitter fast)

    High-Threat Priority:
      Bands with a High-threat emitter get a 2× score multiplier so the model
      always prioritises them over Medium/Low threats.
    """

    # --- Tunable constants ------------------------------------------------
    MAX_ACTIVE_DWELL  = 8    # stay up to 8 steps on a confirmed-active band
    MIN_QUIET_DWELL   = 2    # leave a quiet band after just 2 steps
    POWER_WEIGHT      = 0.45 # strongest weight — instant power evidence
    URGENCY_WEIGHT    = 0.35 # second — periodic prediction
    MODEL_WEIGHT      = 0.20 # third — DQN (grows more useful as model trains)
    HIGH_THREAT_BOOST = 2.0  # score multiplier for High-threat emitter bands
    MED_THREAT_BOOST  = 1.4  # score multiplier for Medium-threat emitter bands
    # -----------------------------------------------------------------------

    def __init__(self, band_count: int = 36) -> None:
        self.band_count = band_count
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if DQNScheduler is not None:
            self.scheduler = DQNScheduler(device=self.device)
            model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'ew_smart_scan_dqn.pth')
            if os.path.exists(model_path):
                checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
                self.scheduler.policy_net.load_state_dict(checkpoint['policy_net'])
                print(f"[SmartScan] Loaded model from {model_path}")
            else:
                print(f"[SmartScan] Warning: Model not found — using rule-based fallback.")
            self.state   = EWState()
            self.tracker = PeriodicEmitterTracker()
        else:
            self.scheduler = None

        self._reset_state()

    def _reset_state(self) -> None:
        self.last_action   = 0
        self.last_obs      = None
        self.dwell_count   = 0
        self.last_detected = False       # was previous step a hit?
        # Memory: bands known to have active emitters (decays over time)
        self._active_memory = np.zeros(self.band_count, dtype=np.float32)

    def reset(self) -> None:
        if self.scheduler is not None:
            self.state.reset()
            self.tracker.reset()
        self._reset_state()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _update_state(self, time_step: int) -> bool:
        """Process outcome of last action. Returns True if last band had a hit."""
        if self.last_obs is None:
            return False

        detected    = self.last_obs["bands"]["is_transmitting"][self.last_action]
        threat_tier = 0
        for e in self.last_obs["active_emitters"]:
            if e["band_id"] == self.last_action:
                t = e.get("threat_level", "Low")
                if   t == "High":   threat_tier = 2
                elif t == "Medium": threat_tier = 1
                break

        # Update active memory: boost on hit, decay everywhere
        self._active_memory *= 0.92          # gentle exponential decay
        if detected:
            self._active_memory[self.last_action] = min(1.0,
                self._active_memory[self.last_action] + 0.5)

        if self.scheduler is not None:
            if detected:
                self.tracker.record_detection(self.last_action, max(0, time_step - 1))
            est_period = self.tracker.estimated_periods[self.last_action]
            self.state.update(self.last_action, detected, threat_tier, est_period)

        return detected

    def _dqn_scores(self, state_vec: np.ndarray) -> np.ndarray:
        """Softmax-normalised Q-values → [0,1] per band."""
        if self.scheduler is None:
            return np.ones(self.band_count, dtype=np.float32) / self.band_count
        with torch.no_grad():
            st  = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
            qv  = self.scheduler.policy_net(st).squeeze(0).cpu().numpy()
        qv -= qv.max()
        exp = np.exp(qv)
        return exp / exp.sum()

    def _power_scores(self, observation: dict) -> np.ndarray:
        """Normalise observed power [−110, −20 dBm] → [0, 1]."""
        powers = np.array(observation["bands"]["power_dbm"], dtype=np.float32)
        scores = (powers + 110.0) / 90.0
        return np.clip(scores, 0.0, 1.0)

    def _threat_multipliers(self, observation: dict) -> np.ndarray:
        """Return per-band threat boost multipliers."""
        mults = np.ones(self.band_count, dtype=np.float32)
        for e in observation["active_emitters"]:
            bid = e.get("band_id", -1)
            if 0 <= bid < self.band_count:
                t = e.get("threat_level", "Low")
                if   t == "High":   mults[bid] = max(mults[bid], self.HIGH_THREAT_BOOST)
                elif t == "Medium": mults[bid] = max(mults[bid], self.MED_THREAT_BOOST)
        return mults

    def _composite_scores(self, observation: dict, time_step: int) -> np.ndarray:
        """Compute final composite score per band."""
        urgency = (self.tracker.get_urgency_scores(time_step)
                   if self.scheduler is not None
                   else np.zeros(self.band_count, dtype=np.float32))
        state_vec = (self.state.encode(urgency)
                     if self.scheduler is not None
                     else np.zeros(218, dtype=np.float32))

        pow_s = self._power_scores(observation)
        urg_s = urgency / (urgency.max() + 1e-8)
        dqn_s = self._dqn_scores(state_vec)

        # Memory layer: give extra weight to bands recently seen active
        mem_s = self._active_memory / (self._active_memory.max() + 1e-8)

        composite = (self.POWER_WEIGHT   * pow_s +
                     self.URGENCY_WEIGHT * urg_s +
                     self.MODEL_WEIGHT   * dqn_s +
                     0.10                * mem_s)  # 10% memory bonus

        # Apply threat multiplier on top
        composite *= self._threat_multipliers(observation)
        return composite

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def select_band(self, observation: dict) -> int:
        time_step = observation["step"]

        # ── Phase 1: deterministic sweep (bands 0 → 35) ────────────────
        cycle_phase = time_step % 150
        if cycle_phase < self.band_count:
            action = cycle_phase
            self._update_state(time_step)
            self.last_obs      = observation
            self.last_action   = action
            self.last_detected = False
            self.dwell_count   = 0
            return action

        # ── Phase 2: intelligent track phase ───────────────────────────
        detected = self._update_state(time_step)
        self.last_obs      = observation
        self.last_detected = detected

        scores = self._composite_scores(observation, time_step)
        best   = int(np.argmax(scores))

        if best == self.last_action:
            self.dwell_count += 1
            # Adaptive dwell limit:
            #   • if we're getting hits → stay longer (exploit the active emitter)
            #   • if quiet → bail out quickly (search for active bands)
            limit = self.MAX_ACTIVE_DWELL if detected else self.MIN_QUIET_DWELL

            if self.dwell_count >= limit:
                # Force move: pick highest-scoring band that isn't current
                scores[self.last_action] = -1.0
                best = int(np.argmax(scores))
                self.dwell_count = 0
        else:
            self.dwell_count = 0

        self.last_action = best
        return best
