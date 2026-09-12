r"""
===========================================================================
Electronic Warfare Smart Scan Strategy — Complete RL-Based Solution
===========================================================================
Supports TWO modes:
  A) Synthetic simulation (default, no dataset required)
  B) Real-data mode (loads from F:\SIH\newProcessed)

Usage:
  # Synthetic (self-contained, no dataset):
    python ew_smart_scan_rl.py

  # Real data:
    python ew_smart_scan_rl.py --data-root "F:\SIH\newProcessed" --data-mode scan

  # Real data with 300 training episodes:
    python ew_smart_scan_rl.py --data-root "F:\SIH\newProcessed" --train-episodes 300 --eval-episodes 30 --steps 500
"""

import os
import math
import random
import argparse
import numpy as np
from collections import deque, defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ===========================================================================
# 1. RECEIVER HARDWARE SPECIFICATIONS (500 MHz — 18 000 MHz)
# ===========================================================================
TOTAL_BANDS = 36
FREQ_MIN_MHZ = 500.0
FREQ_MAX_MHZ = 18000.0
IBW_MHZ = 500.0


def freq_to_band(freq_mhz):
    band = int((freq_mhz - FREQ_MIN_MHZ) // IBW_MHZ)
    return int(np.clip(band, 0, TOTAL_BANDS - 1))


def band_to_center_freq(band_idx):
    return FREQ_MIN_MHZ + band_idx * IBW_MHZ + IBW_MHZ / 2.0


def compute_pulse_threat_score(freq_mhz, pw_us=5.0, pri_us=200.0, pri_jitter_ratio=0.0):
    score = 0.0
    if freq_mhz >= 8000.0:
        score += 3.0
    elif freq_mhz >= 2000.0:
        score += 1.5
    if pw_us < 1.0:
        score += 2.5
    elif pw_us < 10.0:
        score += 1.0
    if pri_us < 50.0 or pri_jitter_ratio > 0.15:
        score += 2.5
    elif pri_us < 200.0:
        score += 1.0
    if score >= 5.0:
        return 2
    elif score >= 2.5:
        return 1
    return 0


def compute_reward(is_hit, is_false_alarm, threat_tier, is_first_intercept, band_distance):
    reward = 0.0
    if is_hit:
        reward += 1.0
        if threat_tier == 1:
            reward += 2.0
        elif threat_tier == 2:
            reward += 6.0
        if is_first_intercept:
            reward += 20.0
    else:
        reward -= 0.05
    if is_false_alarm:
        reward -= 0.5
    reward -= 0.03 * band_distance
    return reward


# ===========================================================================
# 2. RF ENVIRONMENT SIMULATOR (synthetic + real-data modes)
# ===========================================================================
class Emitter:
    TYPES = ["fixed", "periodic_scan", "frequency_agile", "burst"]

    def __init__(self, emitter_id, emitter_type, bands, power_dbm, threat_tier,
                 scan_period=None, dwell_duration=None, hop_sequence=None,
                 burst_on=None, burst_off=None, pw_us=5.0, pri_us=200.0,
                 pri_jitter_ratio=0.0):
        self.emitter_id = emitter_id
        self.emitter_type = emitter_type
        self.bands = bands
        self.power_dbm = power_dbm
        self.threat_tier = threat_tier
        self.scan_period = scan_period
        self.dwell_duration = dwell_duration
        self.hop_sequence = hop_sequence
        self.burst_on = burst_on
        self.burst_off = burst_off
        self.pw_us = pw_us
        self.pri_us = pri_us
        self.pri_jitter_ratio = pri_jitter_ratio

    def is_transmitting(self, time_step):
        if self.emitter_type == "fixed":
            return True, self.bands[0]
        elif self.emitter_type == "periodic_scan":
            phase = time_step % self.scan_period
            return (phase < self.dwell_duration, self.bands[0] if phase < self.dwell_duration else None)
        elif self.emitter_type == "frequency_agile":
            hop_idx = time_step % len(self.hop_sequence)
            return True, self.hop_sequence[hop_idx]
        elif self.emitter_type == "burst":
            cycle = self.burst_on + self.burst_off
            phase = time_step % cycle
            return (phase < self.burst_on, self.bands[0] if phase < self.burst_on else None)
        return False, None


class RFEnvironment:
    def __init__(self, num_emitters=8, num_bands=TOTAL_BANDS, seed=None,
                 dataset_loader=None):
        self.num_bands = num_bands
        self.num_emitters = num_emitters
        self.time_step = 0
        self.rng = np.random.RandomState(seed)
        self.emitters = []

        self.dataset_loader = dataset_loader
        self.use_real_data = dataset_loader is not None
        self._scenario = None
        self._config_idx = 0

        if not self.use_real_data:
            self._generate_emitters()

    def _generate_emitters(self):
        self.emitters = []
        type_dist = (
            ["fixed"] * max(1, self.num_emitters // 4) +
            ["periodic_scan"] * max(1, int(self.num_emitters * 0.3)) +
            ["frequency_agile"] * max(1, self.num_emitters // 4) +
            ["burst"] * max(1, self.num_emitters // 5)
        )
        while len(type_dist) < self.num_emitters:
            type_dist.append(self.rng.choice(Emitter.TYPES))
        type_dist = type_dist[:self.num_emitters]
        self.rng.shuffle(type_dist)

        for i, etype in enumerate(type_dist):
            band = self.rng.randint(0, self.num_bands)
            freq_mhz = band_to_center_freq(band)
            if freq_mhz >= 8000:
                pw_us = self.rng.uniform(0.1, 2.0)
                pri_us = self.rng.uniform(10, 100)
            elif freq_mhz >= 2000:
                pw_us = self.rng.uniform(1.0, 10.0)
                pri_us = self.rng.uniform(50, 500)
            else:
                pw_us = self.rng.uniform(5.0, 50.0)
                pri_us = self.rng.uniform(200, 2000)
            pri_jitter = (self.rng.uniform(0.05, 0.30)
                          if etype in ("frequency_agile", "burst")
                          else self.rng.uniform(0.0, 0.05))
            threat = compute_pulse_threat_score(freq_mhz, pw_us, pri_us, pri_jitter)
            power_dbm = self.rng.uniform(-90, -20)
            kw = dict(emitter_id=i, emitter_type=etype, bands=[band],
                      power_dbm=power_dbm, threat_tier=threat,
                      pw_us=pw_us, pri_us=pri_us, pri_jitter_ratio=pri_jitter)
            if etype == "periodic_scan":
                kw["scan_period"] = self.rng.randint(10, 80)
                kw["dwell_duration"] = self.rng.randint(2, max(3, kw["scan_period"] // 4))
            elif etype == "frequency_agile":
                n = self.rng.randint(3, 8)
                hops = sorted(self.rng.choice(self.num_bands, size=n, replace=False).tolist())
                kw["hop_sequence"] = hops
                kw["bands"] = hops
            elif etype == "burst":
                kw["burst_on"] = self.rng.randint(3, 20)
                kw["burst_off"] = self.rng.randint(10, 60)
            self.emitters.append(Emitter(**kw))

    def reset(self, randomize=True, episode_seed=None, config_idx=None):
        self.time_step = 0
        self._config_idx = config_idx if config_idx is not None else 0

        if self.use_real_data:
            if config_idx is not None:
                self._scenario = self.dataset_loader.load_scenario(config_idx)
            else:
                idx = self.rng.randint(0, self.dataset_loader.num_configs)
                self._scenario = self.dataset_loader.load_scenario(idx)
                self._config_idx = idx

            self.emitters = []
            for m in self._scenario.emitter_metadata:
                self.emitters.append(Emitter(
                    emitter_id=m['emitter_id'],
                    emitter_type='data_driven',
                    bands=[0],
                    power_dbm=-60.0,
                    threat_tier=m['threat_tier'],
                    pw_us=m.get('pw_us', 5.0),
                    pri_us=m.get('pri_us', 200.0),
                    pri_jitter_ratio=m.get('pri_jitter_ratio', 0.0),
                ))
            self.num_emitters = len(self.emitters)
        else:
            if episode_seed is not None:
                self.rng = np.random.RandomState(episode_seed)
                self._generate_emitters()
            elif randomize:
                self._generate_emitters()

    def step(self):
        if self.use_real_data:
            return self._step_real()
        return self._step_synthetic()

    def _step_real(self):
        s = self._scenario
        t = self.time_step % s.num_steps
        ground_truth = s.band_occupancy[:, t].copy()
        band_powers = s.band_powers[:, t].copy()
        offset = self._config_idx * 1000
        emitter_info = [(eid + offset, band, pa, threat)
                        for (eid, band, pa, threat) in s.emitter_info_per_step[t]]
        self.time_step += 1
        return ground_truth, band_powers, emitter_info

    def _step_synthetic(self):
        ground_truth = np.zeros(self.num_bands, dtype=np.float32)
        band_powers = np.full(self.num_bands, -100.0, dtype=np.float32)
        emitter_info = []
        for em in self.emitters:
            active, band = em.is_transmitting(self.time_step)
            if active and band is not None:
                ground_truth[band] = 1.0
                band_powers[band] = max(band_powers[band], em.power_dbm)
                emitter_info.append((em.emitter_id, band, em.power_dbm, em.threat_tier))
        self.time_step += 1
        return ground_truth, band_powers, emitter_info


# ===========================================================================
# 3. RECEIVER MODEL
# ===========================================================================
class ReceiverModel:
    def __init__(self, sensitivity_dbm=-80.0, pfa_rate=0.02,
                 tuning_cost_per_band=0.08, seed=None):
        self.sensitivity_dbm = sensitivity_dbm
        self.pfa_rate = pfa_rate
        self.tuning_cost_per_band = tuning_cost_per_band
        self.current_band = 0
        self.rng = np.random.RandomState(seed)

    def observe(self, band_idx, ground_truth, band_powers):
        self.current_band = band_idx
        real_signal = ground_truth[band_idx] > 0.5
        signal_power = band_powers[band_idx]
        if real_signal and signal_power >= self.sensitivity_dbm:
            return True, True, False, signal_power
        elif not real_signal and self.rng.random() < self.pfa_rate:
            return True, False, True, self.sensitivity_dbm + self.rng.uniform(0, 5)
        else:
            return False, False, False, -200.0

    def tuning_latency(self, from_band, to_band):
        return self.tuning_cost_per_band * abs(to_band - from_band)


# ===========================================================================
# 4. STATE REPRESENTATION
# ===========================================================================
class EWState:
    STATE_DIM = TOTAL_BANDS * 6 + 2

    def __init__(self, num_bands=TOTAL_BANDS):
        self.num_bands = num_bands
        self.reset()

    def reset(self):
        self.time_since_visit = np.zeros(self.num_bands, dtype=np.float32)
        self.last_observation = np.zeros(self.num_bands, dtype=np.float32)
        self.hit_counts = np.zeros(self.num_bands, dtype=np.float32)
        self.visit_counts = np.ones(self.num_bands, dtype=np.float32)
        self.estimated_periods = np.zeros(self.num_bands, dtype=np.float32)
        self.threat_levels = np.zeros(self.num_bands, dtype=np.float32)
        self.current_band = 0
        self.time_step = 0

    def tick(self):
        self.time_since_visit += 1.0
        self.time_step += 1

    def update(self, band_idx, detected, threat_tier, estimated_period=0.0):
        self.time_since_visit += 1.0
        self.time_since_visit[band_idx] = 0.0
        self.last_observation[band_idx] = 1.0 if detected else 0.0
        self.visit_counts[band_idx] += 1.0
        if detected:
            self.hit_counts[band_idx] += 1.0
        if threat_tier > self.threat_levels[band_idx]:
            self.threat_levels[band_idx] = float(threat_tier)
        if estimated_period > 0:
            self.estimated_periods[band_idx] = estimated_period
        self.current_band = band_idx
        self.time_step += 1

    def encode(self, urgency_scores=None):
        hit_rates = self.hit_counts / self.visit_counts
        tsv_norm = np.clip(np.log1p(self.time_since_visit) / np.log1p(500.0), 0, 1)
        period_norm = np.clip(self.estimated_periods / 100.0, 0, 1)
        threat_norm = self.threat_levels / 2.0
        urgency_norm = (np.asarray(urgency_scores, dtype=np.float32)
                        if urgency_scores is not None
                        else np.zeros(self.num_bands, dtype=np.float32))
        per_band = np.stack([tsv_norm, self.last_observation, hit_rates,
                             period_norm, threat_norm, urgency_norm],
                            axis=1).flatten()
        global_feats = np.array([
            self.current_band / float(self.num_bands),
            math.sin(2 * math.pi * self.time_step / 100.0),
        ], dtype=np.float32)
        return np.concatenate([per_band, global_feats]).astype(np.float32)


# ===========================================================================
# 5. PERIODIC EMITTER TRACKER
# ===========================================================================
class PeriodicEmitterTracker:
    def __init__(self, num_bands=TOTAL_BANDS, history_len=200, min_detections=4):
        self.num_bands = num_bands
        self.history_len = history_len
        self.min_detections = min_detections
        self.reset()

    def reset(self):
        self.detection_times = [deque(maxlen=self.history_len) for _ in range(self.num_bands)]
        self.estimated_periods = np.zeros(self.num_bands, dtype=np.float32)
        self.last_detection_time = np.full(self.num_bands, -1, dtype=np.int64)

    def record_detection(self, band_idx, time_step):
        self.detection_times[band_idx].append(time_step)
        self.last_detection_time[band_idx] = time_step
        self._update_period_estimate(band_idx)

    def _update_period_estimate(self, band_idx):
        times = list(self.detection_times[band_idx])
        if len(times) < self.min_detections:
            return
        intervals = np.diff(times).astype(np.float64)
        if len(intervals) < 3:
            return
        median_iv = np.median(intervals)
        if median_iv <= 0:
            return
        valid = sum(1 for iv in intervals
                    if abs(iv / median_iv - round(iv / median_iv)) < 0.25)
        if valid >= len(intervals) * 0.5:
            self.estimated_periods[band_idx] = float(median_iv)

    def predict_next_illumination(self, band_idx, current_time):
        period = self.estimated_periods[band_idx]
        last = self.last_detection_time[band_idx]
        if period <= 0 or last < 0:
            return -1
        dt = current_time - last
        if dt <= 0:
            return current_time
        return int(last + math.ceil(dt / period) * period)

    def get_urgency_scores(self, current_time):
        scores = np.zeros(self.num_bands, dtype=np.float32)
        for b in range(self.num_bands):
            nxt = self.predict_next_illumination(b, current_time)
            if nxt < 0:
                continue
            dt = nxt - current_time
            if dt <= 0:
                scores[b] = 1.0
            elif dt <= 5:
                scores[b] = 1.0 - dt / 5.0
        return scores


# ===========================================================================
# 6. DQN AGENT WITH INTERLEAVED SEARCH / TRACK SCHEDULING
# ===========================================================================
class ReplayBuffer:
    def __init__(self, capacity=50000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states, dtype=np.float32),
                np.array(actions, dtype=np.int64),
                np.array(rewards, dtype=np.float32),
                np.array(next_states, dtype=np.float32),
                np.array(dones, dtype=np.float32))

    def __len__(self):
        return len(self.buffer)


class DQNetwork(nn.Module):
    def __init__(self, state_dim, num_actions, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_actions),
        )

    def forward(self, x):
        return self.net(x)


class DQNScheduler:
    def __init__(self, state_dim=EWState.STATE_DIM, num_bands=TOTAL_BANDS,
                 lr=1e-3, gamma=0.95, epsilon_start=1.0, epsilon_end=0.05,
                 epsilon_decay=10000, target_update_freq=200, batch_size=64,
                 device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_bands = num_bands
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.steps_done = 0

        self.policy_net = DQNetwork(state_dim, num_bands).to(self.device)
        self.target_net = DQNetwork(state_dim, num_bands).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.replay_buffer = ReplayBuffer(capacity=50000)

    def get_epsilon(self):
        return self.epsilon_end + (self.epsilon_start - self.epsilon_end) * \
               math.exp(-self.steps_done / self.epsilon_decay)

    def select_action(self, state, time_step=0, evaluate=False):
        # 1. Operational ESM Cycle: 36 slots Search -> 114 slots Track (150-slot macrocycle)
        cycle_phase = time_step % 150
        if cycle_phase < self.num_bands:
            # Deterministic Search: Sweep all bands in sequence to guarantee discovery
            return cycle_phase

        # 2. Track Phase: Epsilon-Greedy Exploration during Training
        if not evaluate and random.random() < self.get_epsilon():
            return random.randint(0, self.num_bands - 1)

        # 3. Track Phase: Policy Exploitation targeting High-Threat Emitters
        with torch.no_grad():
            st = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            return self.policy_net(st).argmax(dim=1).item()

    def train_step(self):
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        states, actions, rewards, nstates, dones = self.replay_buffer.sample(self.batch_size)
        s = torch.tensor(states, dtype=torch.float32).to(self.device)
        a = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)
        r = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)
        ns = torch.tensor(nstates, dtype=torch.float32).to(self.device)
        d = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)

        q = self.policy_net(s).gather(1, a)
        with torch.no_grad():
            nq = self.target_net(ns).max(dim=1, keepdim=True)[0]
            tgt = r + self.gamma * nq * (1 - d)
        loss = F.smooth_l1_loss(q, tgt)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        return loss.item()


# ===========================================================================
# 7. HUBER-STABILIZED HYBRID ARRIVAL PREDICTOR
# ===========================================================================
class InterceptTimePredictor(nn.Module):
    def __init__(self, state_dim=EWState.STATE_DIM, num_bands=TOTAL_BANDS,
                 hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2), nn.ReLU(),
            nn.Linear(hidden_dim // 2, num_bands),
        )

    def forward(self, state):
        return self.net(state)


class HybridInterceptPredictor:
    def __init__(self, state_dim=EWState.STATE_DIM, num_bands=TOTAL_BANDS, lr=1e-3, device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.predictor = InterceptTimePredictor(state_dim, num_bands).to(self.device)
        self.optimizer = optim.Adam(self.predictor.parameters(), lr=lr)
        self.num_bands = num_bands
        self.buffer = deque(maxlen=10000)
        self.max_horizon = 50.0

    def record(self, state, actual_intercept_times):
        self.buffer.append((state.copy(), actual_intercept_times.copy()))

    def train_step(self, batch_size=32):
        if len(self.buffer) < batch_size:
            return 0.0
        batch = random.sample(list(self.buffer), batch_size)
        states, targets = zip(*batch)

        s = torch.tensor(np.array(states), dtype=torch.float32).to(self.device)
        # Normalize targets to [0, 1] range to eliminate loss explosions
        norm_targets = torch.tensor(np.array(targets) / self.max_horizon, dtype=torch.float32).to(self.device)

        # Predict normalized values using Sigmoid projection
        preds = torch.sigmoid(self.predictor(s))

        # Smooth L1 (Huber) loss guarantees linear penalty on large deviations
        loss = F.smooth_l1_loss(preds, norm_targets)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.predictor.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def predict(self, state, tracker=None, current_time=0):
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
            # Rescale normalized output back to [0, 50.0] slots
            neural_preds = (torch.sigmoid(self.predictor(s)).squeeze(0).cpu().numpy()) * self.max_horizon

        # Physics override: exact periodic tracking takes precedence
        if tracker is not None:
            for b in range(self.num_bands):
                nxt = tracker.predict_next_illumination(b, current_time)
                if nxt >= current_time:
                    neural_preds[b] = float(min(nxt - current_time, self.max_horizon))
        return neural_preds


class UniformSweepTimePredictor:
    """Deterministic circular sweep predictor for fair baseline comparison."""
    def __init__(self, num_bands=TOTAL_BANDS):
        self.num_bands = num_bands

    def predict(self, state, tracker=None, current_time=0):
        curr_band = int(round(state[-2] * self.num_bands)) % self.num_bands
        preds = np.zeros(self.num_bands, dtype=np.float32)
        for b in range(self.num_bands):
            steps_away = (b - curr_band) % self.num_bands
            preds[b] = float(steps_away if steps_away > 0 else self.num_bands)
        return preds

    def record(self, state, actual):
        pass

    def train_step(self, batch_size=32):
        return 0.0


# ===========================================================================
# 8. FIGURES OF MERIT
# ===========================================================================
class FiguresOfMerit:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_hits = 0
        self.total_misses = 0
        self.total_false_alarms = 0
        self.total_correct_empty = 0
        self.total_transmissions_visible = 0
        self.min_detected_power = float('inf')
        self.total_time_slots = 0
        self.total_wall_clock_slots = 0
        self.cumulative_reward = 0.0
        self.correct_predictions = 0
        self.total_predictions = 0
        self.intercept_time_errors = []
        self.emitter_first_intercept = {}
        self.emitter_appearance = {}

    def record_observation(self, is_hit, is_false_alarm, signal_power,
                           reward, band_had_signal):
        self.total_time_slots += 1
        self.total_wall_clock_slots += 1
        self.cumulative_reward += reward
        if band_had_signal:
            self.total_transmissions_visible += 1
            if is_hit:
                self.total_hits += 1
                if signal_power > -200:
                    self.min_detected_power = min(self.min_detected_power, signal_power)
            else:
                self.total_misses += 1
        else:
            if is_false_alarm:
                self.total_false_alarms += 1
            else:
                self.total_correct_empty += 1

    def record_blind_slot(self):
        self.total_wall_clock_slots += 1

    def record_prediction(self, predicted_band, actual_active_bands):
        self.total_predictions += 1
        if actual_active_bands[predicted_band] > 0.5:
            self.correct_predictions += 1

    def record_intercept_time(self, predicted_time, actual_time):
        self.intercept_time_errors.append(abs(predicted_time - actual_time))

    def record_emitter_event(self, emitter_id, time_step, intercepted):
        if emitter_id not in self.emitter_appearance:
            self.emitter_appearance[emitter_id] = time_step
        if intercepted and emitter_id not in self.emitter_first_intercept:
            self.emitter_first_intercept[emitter_id] = time_step

    def compute(self):
        total_empty = self.total_false_alarms + self.total_correct_empty
        first_delays = [self.emitter_first_intercept[e] - self.emitter_appearance[e]
                        for e in self.emitter_appearance
                        if e in self.emitter_first_intercept]
        return {
            "P_d": self.total_hits / max(self.total_transmissions_visible, 1),
            "P_fa": self.total_false_alarms / max(total_empty, 1),
            "Sensitivity_dBm": (self.min_detected_power
                                if self.min_detected_power < float('inf') else -200.0),
            "Avg_Intercept_Rate": self.total_hits / max(self.total_time_slots, 1),
            "Avg_Intercept_Rate_WallClock": self.total_hits / max(self.total_wall_clock_slots, 1),
            "Avg_Reward": self.cumulative_reward / max(self.total_time_slots, 1),
            "Pct_Correct_Predictions": (self.correct_predictions
                                        / max(self.total_predictions, 1) * 100.0),
            "Avg_Intercept_Time_Error": (np.mean(self.intercept_time_errors)
                                         if self.intercept_time_errors else 0.0),
            "Avg_First_Intercept_Delay": (np.mean(first_delays)
                                          if first_delays else float('inf')),
        }

    def print_report(self, title="Figures of Merit"):
        m = self.compute()
        print(f"\n{'='*60}")
        print(f"  {title}")
        print(f"{'='*60}")
        print(f"  1. P(Detection)             : {m['P_d']:.4f}")
        print(f"  2. P(False Alarm)            : {m['P_fa']:.4f}")
        print(f"  3. Sensitivity               : {m['Sensitivity_dBm']:.1f} dBm")
        print(f"  4. Avg Intercept Rate         : {m['Avg_Intercept_Rate']:.4f}")
        print(f"     (wall-clock, incl. retunes): {m['Avg_Intercept_Rate_WallClock']:.4f}")
        print(f"  5. Avg Reward / Cost          : {m['Avg_Reward']:.4f}")
        print(f"  6. Correct Predictions        : {m['Pct_Correct_Predictions']:.2f}%")
        print(f"  7. Avg Intercept Time Error   : {m['Avg_Intercept_Time_Error']:.2f} slots")
        print(f"  *  Avg First-Intercept Delay  : {m['Avg_First_Intercept_Delay']:.2f} slots")
        print(f"{'='*60}")
        return m


# ===========================================================================
# 9. BASELINE STRATEGY
# ===========================================================================
class UniformSweepStrategy:
    def __init__(self, num_bands=TOTAL_BANDS):
        self.num_bands = num_bands
        self.current_band = 0

    def reset(self):
        self.current_band = 0

    def select_action(self, state=None, evaluate=True):
        action = self.current_band
        self.current_band = (self.current_band + 1) % self.num_bands
        return action


# ===========================================================================
# 10. EPISODE RUNNER
# ===========================================================================
def run_episode(env, receiver, scheduler, ew_state, periodic_tracker,
                intercept_trainer, fom, steps_per_episode=500,
                is_training=True, record_intercept_data=True,
                episode_seed=None, config_idx=None):

    env.reset(randomize=True, episode_seed=episode_seed, config_idx=config_idx)
    ew_state.reset()
    periodic_tracker.reset()
    receiver.current_band = 0
    if hasattr(scheduler, 'reset'):
        scheduler.reset()

    episode_reward = 0.0
    episode_loss = 0.0
    train_steps = 0
    intercepted_emitters = set()
    state_history = []
    intercept_history = defaultdict(list)
    evaluate = not is_training

    t = 0
    while t < steps_per_episode:
        urgency = periodic_tracker.get_urgency_scores(t)
        state = ew_state.encode(urgency)

        # Select action using current time slot for search/track alternation
        if hasattr(scheduler, 'policy_net'):
            action = scheduler.select_action(state, time_step=t, evaluate=evaluate)
        else:
            action = scheduler.select_action(state=state, evaluate=evaluate)

        # Tuning latency (blind slots)
        band_distance = abs(action - receiver.current_band)
        delay = int(round(receiver.tuning_latency(receiver.current_band, action)))
        for _ in range(delay):
            if t >= steps_per_episode:
                break
            gt, bp, ei = env.step()
            for eid, eband, ep, et in ei:
                fom.record_emitter_event(eid, t, False)
            fom.record_blind_slot()
            ew_state.tick()
            t += 1
        if t >= steps_per_episode:
            break

        # Observation
        ground_truth, band_powers, emitter_info = env.step()
        detected, is_hit, is_fa, sig_pow = receiver.observe(
            action, ground_truth, band_powers)

        threat_tier = 0
        is_first = False
        for eid, eband, ep, et in emitter_info:
            fom.record_emitter_event(eid, t, (is_hit and eband == action))
            if eband == action and is_hit:
                threat_tier = max(threat_tier, et)
                if eid not in intercepted_emitters:
                    intercepted_emitters.add(eid)
                    is_first = True

        reward = compute_reward(is_hit, is_fa, threat_tier, is_first, band_distance)
        episode_reward += reward

        if detected and not is_fa:
            periodic_tracker.record_detection(action, t)

        est_period = periodic_tracker.estimated_periods[action]
        ew_state.update(action, detected, threat_tier, est_period)
        next_urgency = periodic_tracker.get_urgency_scores(t + 1)
        next_state = ew_state.encode(next_urgency)

        band_had_signal = ground_truth[action] > 0.5
        fom.record_observation(is_hit, is_fa, sig_pow, reward, band_had_signal)
        fom.record_prediction(action, ground_truth)

        if record_intercept_data:
            state_history.append((t, state.copy()))
            if is_hit:
                intercept_history[action].append(t)

        if is_training and hasattr(scheduler, 'replay_buffer'):
            done = (t + 1 >= steps_per_episode)
            scheduler.replay_buffer.push(state, action, reward, next_state, done)
            loss = scheduler.train_step()
            if loss > 0:
                episode_loss += loss
                train_steps += 1

        t += 1

    # Intercept-time predictor calculation
    if record_intercept_data and intercept_trainer is not None and state_history:
        max_h = 50.0
        for rec_t, rec_state in state_history[::5]:
            actual = np.full(TOTAL_BANDS, max_h, dtype=np.float32)
            for bi in range(TOTAL_BANDS):
                for it_t in intercept_history.get(bi, []):
                    dt = it_t - rec_t
                    if dt > 0:
                        actual[bi] = min(actual[bi], float(dt))
                        break
            intercept_trainer.record(rec_state, actual)

            # Physics-assisted prediction evaluation
            preds = intercept_trainer.predict(rec_state, tracker=periodic_tracker, current_time=rec_t)
            for bi in range(TOTAL_BANDS):
                if actual[bi] < max_h:
                    fom.record_intercept_time(preds[bi], actual[bi])

        if is_training and hasattr(intercept_trainer, 'train_step'):
            for _ in range(5):
                intercept_trainer.train_step(batch_size=32)

    return episode_reward, episode_loss / max(train_steps, 1)


# ===========================================================================
# 11. MAIN TRAINING & EVALUATION
# ===========================================================================
def train_and_evaluate(num_train_episodes=300, num_eval_episodes=30,
                       steps_per_episode=500, num_emitters=8,
                       data_root=None, data_mode="scan"):

    # Safe CUDA detection with automatic CPU fallback
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        try:
            test_tensor = torch.zeros(1, device=device)
            _ = F.relu(test_tensor)
        except RuntimeError:
            print("[Warning] CUDA device incompatible with current PyTorch build. Falling back to CPU.")
            device = torch.device('cpu')

    print(f"[Init] Device: {device}")

    # Set up environment
    train_loader, val_loader, test_loader = None, None, None
    train_env, val_env, test_env = None, None, None
    train_configs, eval_configs, test_configs = None, None, None

    if data_root:
        from dataset_loader import EWDatasetLoader
        # Load train
        train_loader = EWDatasetLoader(data_root, mode=data_mode, split="train", target_steps=steps_per_episode)
        train_env = RFEnvironment(dataset_loader=train_loader)
        # Load val
        val_loader = EWDatasetLoader(data_root, mode=data_mode, split="val", target_steps=steps_per_episode)
        val_env = RFEnvironment(dataset_loader=val_loader)
        # Load test
        test_loader = EWDatasetLoader(data_root, mode=data_mode, split="test", target_steps=steps_per_episode)
        test_env = RFEnvironment(dataset_loader=test_loader)
        
        recv_params = train_loader.get_receiver_params(0)
        sensitivity = recv_params.get('sensitivity_dbm', -110.0)
        receiver = ReceiverModel(sensitivity_dbm=sensitivity, pfa_rate=0.02, seed=123)

        train_configs = list(range(train_loader.num_configs))
        eval_configs = list(range(val_loader.num_configs))
        test_configs = list(range(test_loader.num_configs))
        
        print(f"[Data] Real data mode: {data_mode}")
        print(f"[Data] {len(train_configs)} train configs, {len(eval_configs)} eval configs, {len(test_configs)} test configs")
        print(f"[Data] Receiver sensitivity: {sensitivity} dBm")
        
        env = train_env # default env for training loop
    else:
        env = RFEnvironment(num_emitters=num_emitters, seed=42)
        val_env = env
        test_env = env
        receiver = ReceiverModel(sensitivity_dbm=-80.0, pfa_rate=0.02, seed=123)
        print(f"[Init] Synthetic mode: {num_emitters} random emitters")

    print(f"[Init] Training for {num_train_episodes} episodes, {steps_per_episode} steps/episode")
    print(f"[Init] State dim: {EWState.STATE_DIM}, Actions: {TOTAL_BANDS}\n")

    dqn = DQNScheduler(state_dim=EWState.STATE_DIM, num_bands=TOTAL_BANDS, device=device)
    ew_state = EWState()
    tracker = PeriodicEmitterTracker()
    itp = HybridInterceptPredictor(device=device)
    train_fom = FiguresOfMerit()

    # ── PHASE 1: TRAINING ────────────────────────────────────────────
    print("=" * 60)
    print("  PHASE 1: TRAINING (DQN Learning from Hits & Misses)")
    print("=" * 60)

    for ep in range(num_train_episodes):
        cfg = train_configs[ep % len(train_configs)] if train_configs else None
        ep_r, ep_l = run_episode(
            env, receiver, dqn, ew_state, tracker, itp, train_fom,
            steps_per_episode=steps_per_episode, is_training=True,
            config_idx=cfg)

        if (ep + 1) % 10 == 0:
            rate = train_fom.total_hits / max(train_fom.total_time_slots, 1)
            print(f"  Episode {ep+1:4d}/{num_train_episodes} | "
                  f"Reward: {ep_r:8.1f} | Loss: {ep_l:.4f} | "
                  f"eps: {dqn.get_epsilon():.3f} | Rate: {rate:.3f}")

    train_fom.print_report("Training Phase — Aggregate Figures of Merit")

    # ── PHASE 2: EVALUATE DQN ────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 2: EVALUATION — TRAINED DQN (epsilon = 0)")
    print("=" * 60)

    eval_indices = eval_configs if train_configs else [900_000 + i for i in range(num_eval_episodes)]
    eval_fom_dqn = FiguresOfMerit()
    for ep in range(num_eval_episodes):
        if train_configs:
            cfg = eval_indices[ep % len(eval_indices)]
            run_episode(val_env, receiver, dqn, ew_state, tracker, itp,
                        eval_fom_dqn, steps_per_episode=steps_per_episode,
                        is_training=False, config_idx=cfg)
        else:
            run_episode(val_env, receiver, dqn, ew_state, tracker, itp,
                        eval_fom_dqn, steps_per_episode=steps_per_episode,
                        is_training=False, episode_seed=eval_indices[ep])

    dqn_m = eval_fom_dqn.print_report("DQN Agent — Evaluation")

    # ── PHASE 3: BASELINE ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 3: BASELINE — UNIFORM SWEEP")
    print("=" * 60)

    baseline = UniformSweepStrategy()
    bl_predictor = UniformSweepTimePredictor()
    eval_fom_bl = FiguresOfMerit()

    for ep in range(num_eval_episodes):
        if train_configs:
            cfg = eval_indices[ep % len(eval_indices)]
            run_episode(val_env, receiver, baseline, ew_state, tracker, bl_predictor,
                        eval_fom_bl, steps_per_episode=steps_per_episode,
                        is_training=False, record_intercept_data=True,
                        config_idx=cfg)
        else:
            run_episode(val_env, receiver, baseline, ew_state, tracker, bl_predictor,
                        eval_fom_bl, steps_per_episode=steps_per_episode,
                        is_training=False, record_intercept_data=True,
                        episode_seed=eval_indices[ep])

    bl_m = eval_fom_bl.print_report("Uniform Sweep — Evaluation")

    # ── PHASE 4: TEST ON TEST DATASET ────────────────────────────────
    print("\n" + "=" * 60)
    print("  PHASE 4: EVALUATION ON TEST SPLIT")
    print("=" * 60)

    if test_configs is not None:
        test_fom_dqn = FiguresOfMerit()
        for ep in range(len(test_configs)):
            cfg = test_configs[ep]
            run_episode(test_env, receiver, dqn, ew_state, tracker, itp,
                        test_fom_dqn, steps_per_episode=steps_per_episode,
                        is_training=False, config_idx=cfg)
        test_fom_dqn.print_report("DQN Agent — Test Set Evaluation")
    else:
        print("  [Skip] No test dataset provided.")

    bl_m = eval_fom_bl.print_report("Uniform Sweep — Evaluation")

    # ── COMPARISON TABLE ─────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  COMPARISON: DQN Agent vs. Uniform Sweep Baseline")
    print("=" * 60)
    print(f"  {'Metric':<30s} {'DQN':>12s} {'Baseline':>12s} {'Delta':>12s}")
    print(f"  {'-'*66}")
    lower_better = {"P_fa", "Avg_Intercept_Time_Error", "Avg_First_Intercept_Delay"}
    for key in ["P_d", "P_fa", "Avg_Intercept_Rate", "Avg_Intercept_Rate_WallClock",
                "Avg_Reward", "Pct_Correct_Predictions", "Avg_Intercept_Time_Error",
                "Avg_First_Intercept_Delay"]:
        dv, bv = dqn_m.get(key, 0), bl_m.get(key, 0)
        delta = (bv - dv) if key in lower_better else (dv - bv)
        tag = "better" if delta > 0 else "worse"
        print(f"  {key:<30s} {dv:>12.4f} {bv:>12.4f} {delta:>+8.4f} {tag}")
    print(f"  {'='*66}")

    # ── Save model weights ───────────────────────────────────────────
    # Only save if this was a meaningful training run (≥50 episodes) to avoid
    # short test/debug runs overwriting a properly trained model.
    save_path = "ew_smart_scan_dqn.pth"
    if num_train_episodes >= 50:
        torch.save({
            'policy_net': dqn.policy_net.state_dict(),
            'target_net': dqn.target_net.state_dict(),
            'intercept_predictor': itp.predictor.state_dict(),
        }, save_path)
        print(f"\n[Saved] Model weights -> {save_path}")
    else:
        print(f"\n[Skipped Save] Only {num_train_episodes} episodes — need ≥50 to overwrite model. "
              f"Existing {save_path} preserved.")

    return dqn, itp


# ===========================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="EW Smart Scan — DQN RL Training & Evaluation")
    parser.add_argument("--data-root", type=str, default=None,
                        help=r"Path to real dataset (e.g. F:\SIH\newProcessed)")
    parser.add_argument("--data-mode", type=str, default="scan",
                        choices=["scan", "stare"],
                        help="Dataset receiver mode (default: scan)")
    parser.add_argument("--train-episodes", type=int, default=300)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--emitters", type=int, default=8,
                        help="Number of synthetic emitters (ignored with --data-root)")
    args = parser.parse_args()

    train_and_evaluate(
        num_train_episodes=args.train_episodes,
        num_eval_episodes=args.eval_episodes,
        steps_per_episode=args.steps,
        num_emitters=args.emitters,
        data_root=args.data_root,
        data_mode=args.data_mode,
    )