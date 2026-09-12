from .base import SchedulerPolicy

import os
import torch
import numpy as np
from .base import SchedulerPolicy
import sys

# Add parent directory to path to import ew_smart_scan_rl
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
try:
    from ew_smart_scan_rl import DQNScheduler, EWState, PeriodicEmitterTracker
except ImportError:
    DQNScheduler, EWState, PeriodicEmitterTracker = None, None, None


class SmartScanScheduler(SchedulerPolicy):
    """
    Hybrid Smart Scan Scheduler.

    Combines DQN model hints with a rule-based fallback to guarantee
    dynamic band-hopping even when the model is undertrained.

    Phases (per 150-slot macrocycle):
      0-35  : Deterministic sweep (band = cycle_phase), covers all 36 bands
      36-149: Smart Track Phase - picks band by weighted score combining:
                (a) DQN Q-value confidence
                (b) Periodic urgency from PeriodicEmitterTracker
                (c) Raw detection power from observation
              Dwell limiter: forces move after MAX_DWELL_SLOTS on the same band.
    """

    MAX_DWELL_SLOTS   = 6   # max consecutive steps on the same band before forced move
    MODEL_WEIGHT      = 0.35  # weight for DQN Q-values in scoring
    URGENCY_WEIGHT    = 0.40  # weight for periodic tracker urgency
    POWER_WEIGHT      = 0.25  # weight for raw observed power

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
                print(f"[SmartScan] Warning: Model not found at {model_path}. Using rule-based fallback.")
            self.state   = EWState()
            self.tracker = PeriodicEmitterTracker()
        else:
            self.scheduler = None

        self._reset_state()

    def _reset_state(self) -> None:
        self.last_action    = 0
        self.last_obs       = None
        self.dwell_count    = 0   # consecutive steps on same band
        self.sweep_ptr      = 0   # pointer for fallback round-robin

    def reset(self) -> None:
        if self.scheduler is not None:
            self.state.reset()
            self.tracker.reset()
        self._reset_state()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_state(self, time_step: int) -> bool:
        """Update EWState and tracker from last observation. Returns detected flag."""
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

        if self.scheduler is not None:
            if detected:
                self.tracker.record_detection(self.last_action, max(0, time_step - 1))
            est_period = self.tracker.estimated_periods[self.last_action]
            self.state.update(self.last_action, detected, threat_tier, est_period)

        return detected

    def _dqn_scores(self, state_vec: np.ndarray) -> np.ndarray:
        """Return softmax-normalised Q-values as a score vector [0-1] per band."""
        if self.scheduler is None:
            return np.ones(self.band_count, dtype=np.float32) / self.band_count
        with torch.no_grad():
            st  = torch.tensor(state_vec, dtype=torch.float32).unsqueeze(0).to(self.device)
            qv  = self.scheduler.policy_net(st).squeeze(0).cpu().numpy()
        # Softmax to convert raw Q-values → probabilities (avoids argmax collapse)
        qv  = qv - qv.max()
        exp = np.exp(qv)
        return exp / exp.sum()

    def _power_scores(self, observation: dict) -> np.ndarray:
        """Normalise observed power across bands to [0-1]."""
        powers = np.array(observation["bands"]["power_dbm"], dtype=np.float32)
        # Map from typical range [-110, -20] dBm to [0, 1]
        scores = (powers - (-110.0)) / (90.0)
        return np.clip(scores, 0.0, 1.0)

    def _pick_best_band(self, observation: dict, time_step: int) -> int:
        """Compute weighted composite score and return best band index."""
        urgency = (self.tracker.get_urgency_scores(time_step)
                   if self.scheduler is not None
                   else np.zeros(self.band_count, dtype=np.float32))

        state_vec = (self.state.encode(urgency)
                     if self.scheduler is not None
                     else np.zeros(218, dtype=np.float32))

        dqn_s   = self._dqn_scores(state_vec)
        urg_s   = urgency / (urgency.max() + 1e-8)
        pow_s   = self._power_scores(observation)

        composite = (self.MODEL_WEIGHT   * dqn_s +
                     self.URGENCY_WEIGHT * urg_s +
                     self.POWER_WEIGHT   * pow_s)

        return int(np.argmax(composite))

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def select_band(self, observation: dict) -> int:
        time_step = observation["step"]

        # Phase 1: initial deterministic sweep (bands 0..35)
        cycle_phase = time_step % 150
        if cycle_phase < self.band_count:
            action = cycle_phase       # guaranteed to visit every band once
            self._update_state(time_step)
            self.last_obs    = observation
            self.last_action = action
            self.dwell_count = 0
            return action

        # Phase 2: Smart track phase
        self._update_state(time_step)
        self.last_obs = observation

        best = self._pick_best_band(observation, time_step)

        # Dwell limiter: if we've been on the same band too long, force a move
        if best == self.last_action:
            self.dwell_count += 1
            if self.dwell_count >= self.MAX_DWELL_SLOTS:
                # Pick second-best from composite scores
                urgency = (self.tracker.get_urgency_scores(time_step)
                           if self.scheduler is not None
                           else np.zeros(self.band_count, dtype=np.float32))
                state_vec = (self.state.encode(urgency)
                             if self.scheduler is not None
                             else np.zeros(218, dtype=np.float32))
                dqn_s    = self._dqn_scores(state_vec)
                urg_s    = urgency / (urgency.max() + 1e-8)
                pow_s    = self._power_scores(observation)
                comp     = (self.MODEL_WEIGHT   * dqn_s +
                            self.URGENCY_WEIGHT * urg_s +
                            self.POWER_WEIGHT   * pow_s)
                comp[self.last_action] = -1.0   # exclude current band
                best = int(np.argmax(comp))
                self.dwell_count = 0
        else:
            self.dwell_count = 0

        self.last_action = best
        return best
