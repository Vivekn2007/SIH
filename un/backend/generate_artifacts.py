"""
===========================================================================
EW Smart Scan — Frontend-Identical Artifact Generator
===========================================================================
Generates graphs using EXACTLY the same simulation stack the frontend sees:
  - environment.py  → RFEnvironment  (same synthetic + CSV replay)
  - schedulers/smart_scan.py → SmartScanScheduler (loaded DQN model)
  - schedulers/open_loop.py  → OpenLoopScheduler  (round-robin baseline)
  - main.py → EngineStats   (identical detection_prob / intercept_rate /
                              false_alarms / reward / avg_delay logic)

Outputs:
  1. frontend_kpis.png          — All 5 KPI cards over time (as the frontend shows)
  2. comparison_chart.png       — Smart Scan vs Open Loop (mirrors the live graph)
  3. spectrum_heatmap.png       — Band power & scheduler dwell heatmap
  4. figures_of_merit.csv       — KPI table (real numbers, not hardcoded)
  5. summary_report.json        — Full metadata + FoM payload

Usage:
  cd E:\\newSIH\\ff\\un\\backend
  python generate_artifacts.py
  python generate_artifacts.py --steps 2000 --compare-every 50 --out-dir artifacts/
===========================================================================
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque

# ── Use the EXACT same imports the frontend's backend uses ──────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from environment import RFEnvironment
from schedulers import OpenLoopScheduler, SmartScanScheduler

# ── EngineStats — copied verbatim from main.py so metrics are identical ─────
class EngineStats:
    WINDOW = 200

    def __init__(self):
        self.scans = self.hits = self.false_alarms = self.reward = 0
        self._hit_streak = 0
        self.intercept_delays = []
        self.emitter_first_seen = {}
        self._recent = []

    def update(self, band, observation):
        self.scans += 1
        step     = observation["step"]
        emitters = observation["active_emitters"]
        powers   = observation["bands"]["power_dbm"]

        for e in emitters:
            self.emitter_first_seen.setdefault(e["id"], step)

        hit_emitters = [e for e in emitters if e["band_id"] == band]
        hit = bool(hit_emitters)

        self._recent.append(hit)
        if len(self._recent) > self.WINDOW:
            self._recent.pop(0)

        if hit:
            self.hits += 1
            self._hit_streak += 1
            emitter  = hit_emitters[0]
            eid      = emitter["id"]
            first    = self.emitter_first_seen.pop(eid, step)
            delay    = step - first
            self.intercept_delays.append(delay)
            threat_bonus = {"High": 10, "Medium": 5, "Low": 0}.get(
                emitter.get("threat_level", "Low"), 0)
            early_bonus  = max(0, 15 - delay // 3)
            streak_bonus = min(self._hit_streak * 2, 12)
            self.reward += 20 + early_bonus + streak_bonus + threat_bonus
            return True, eid

        self._hit_streak = 0
        if powers[band] > -98:
            self.false_alarms += 1
            self.reward -= 2
        return False, None

    def cumulative(self):
        det_prob  = sum(self._recent) / len(self._recent) * 100 if self._recent else 0.0
        intercept = self.hits / self.scans * 100 if self.scans else 0.0
        avg_delay = (round(sum(self.intercept_delays) / len(self.intercept_delays), 2)
                     if self.intercept_delays else 0.0)
        return {
            "detection_prob":      round(det_prob, 1),
            "intercept_rate":      round(intercept, 1),
            "false_alarms":        self.false_alarms,
            "avg_delay":           avg_delay,
            "reward":              self.reward,
        }


# ── Matplotlib theme matching the frontend dark style ───────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "figure.facecolor":  "#0b1220",
    "axes.facecolor":    "#0f1a2e",
    "axes.edgecolor":    "#27445b",
    "axes.labelcolor":   "#94a3b8",
    "xtick.color":       "#64748b",
    "ytick.color":       "#64748b",
    "text.color":        "#e5e7eb",
    "grid.color":        "#1e3a4f",
    "grid.linestyle":    ":",
    "legend.facecolor":  "#0f1a2e",
    "legend.edgecolor":  "#27445b",
    "legend.labelcolor": "#cbd5e1",
})

CYAN   = "#22d3ee"
RED    = "#ef4444"
GREEN  = "#10b981"
PURPLE = "#a855f7"
AMBER  = "#f59e0b"
SLATE  = "#64748b"


# ══════════════════════════════════════════════════════════════════════════════
# Core simulation loop — same as Simulation.tick() in main.py
# ══════════════════════════════════════════════════════════════════════════════
def run_simulation(total_steps: int, compare_every: int):
    """
    Runs the exact same tick loop as the live backend.
    Returns per-step logs and the comparison_history list the frontend charts.
    """
    env         = RFEnvironment()
    open_sched  = OpenLoopScheduler()
    smart_sched = SmartScanScheduler()
    open_stats  = EngineStats()
    smart_stats = EngineStats()

    history = deque(maxlen=500)   # mirrors main.py deque(maxlen=80) but larger for graphs

    # Per-step logs
    log = {
        "step":              [],
        "smart_det_prob":    [],
        "open_det_prob":     [],
        "smart_intercept":   [],
        "open_intercept":    [],
        "smart_false_alarms":[],
        "open_false_alarms": [],
        "smart_reward":      [],
        "open_reward":       [],
        "smart_avg_delay":   [],
        "open_avg_delay":    [],
        "smart_band":        [],
        "open_band":         [],
        "band_powers":       [],   # list of 36-element arrays
        "sinr_db":           [],
    }

    print(f"  Running {total_steps} ticks (same as frontend simulation)...")
    for _ in range(total_steps):
        obs         = env.advance()
        open_band   = open_sched.select_band(obs)
        smart_band  = smart_sched.select_band(obs)
        open_stats.update(open_band, obs)
        smart_stats.update(smart_band, obs)

        open_c  = open_stats.cumulative()
        smart_c = smart_stats.cumulative()

        step = obs["step"]
        if step % compare_every == 0:
            history.append({
                "step":       step,
                "open_loop":  open_c,
                "smart_scan": smart_c,
            })

        # Compute SINR exactly as main.py does
        active = obs["active_emitters"]
        if active:
            sig = max(obs["bands"]["power_dbm"][e["band_id"]] for e in active)
        else:
            sig = -104.0
        sinr = max(sig - (-104.0), 0.0) * (smart_c["detection_prob"] / 100.0 + 0.1)

        log["step"].append(step)
        log["smart_det_prob"].append(smart_c["detection_prob"])
        log["open_det_prob"].append(open_c["detection_prob"])
        log["smart_intercept"].append(smart_c["intercept_rate"])
        log["open_intercept"].append(open_c["intercept_rate"])
        log["smart_false_alarms"].append(smart_c["false_alarms"])
        log["open_false_alarms"].append(open_c["false_alarms"])
        log["smart_reward"].append(smart_c["reward"])
        log["open_reward"].append(open_c["reward"])
        log["smart_avg_delay"].append(smart_c["avg_delay"])
        log["open_avg_delay"].append(open_c["avg_delay"])
        log["smart_band"].append(smart_band)
        log["open_band"].append(open_band)
        log["band_powers"].append(obs["bands"]["power_dbm"].copy())
        log["sinr_db"].append(round(sinr, 1))

        if step % 200 == 0:
            print(f"    Step {step:4d}/{total_steps} | "
                  f"Smart det={smart_c['detection_prob']:5.1f}% "
                  f"Open det={open_c['detection_prob']:5.1f}% | "
                  f"Smart FA={smart_c['false_alarms']} "
                  f"Open FA={open_c['false_alarms']} | "
                  f"SINR={sinr:.1f} dB")

    return log, list(history), smart_stats.cumulative(), open_stats.cumulative()


# ══════════════════════════════════════════════════════════════════════════════
# Plot 1 — Frontend KPI Cards over time
# ══════════════════════════════════════════════════════════════════════════════
def plot_kpis(log, out):
    steps = log["step"]

    fig = plt.figure(figsize=(16, 10), dpi=160)
    fig.suptitle("EW Smart Scan — Live KPI Metrics Over Time\n"
                 "(Identical data to what the frontend dashboard shows)",
                 fontsize=13, fontweight="bold", color="#bae6fd", y=0.98)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.55, wspace=0.35)

    def ax_style(ax, title, ylabel, ylim=None):
        ax.set_title(title, fontsize=10, fontweight="bold", color="#bae6fd", pad=6)
        ax.set_xlabel("Simulation Step", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.grid(True, alpha=0.4)
        if ylim:
            ax.set_ylim(*ylim)

    # ── Detection Probability (rolling 200-step window = same as frontend) ──
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(steps, log["smart_det_prob"], color=CYAN,  lw=1.8, label="Smart Scan RL")
    ax1.plot(steps, log["open_det_prob"],  color=SLATE, lw=1.2, linestyle="--", label="Open Loop")
    ax1.legend(fontsize=8)
    ax_style(ax1, "Detection Probability (%) — Rolling 200-step window", "Det Prob (%)", (0, 100))

    # ── Intercept Rate ───────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(steps, log["smart_intercept"], color=CYAN,  lw=1.8, label="Smart Scan RL")
    ax2.plot(steps, log["open_intercept"],  color=SLATE, lw=1.2, linestyle="--", label="Open Loop")
    ax2.legend(fontsize=8)
    ax_style(ax2, "Intercept Rate (%) — Lifetime hits / total scans", "Intercept Rate (%)")

    # ── False Alarms (cumulative) ─────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(steps, log["smart_false_alarms"], color=RED,   lw=1.8, label="Smart Scan RL")
    ax3.plot(steps, log["open_false_alarms"],  color=AMBER, lw=1.2, linestyle="--", label="Open Loop")
    ax3.legend(fontsize=8)
    ax_style(ax3, "False Alarms (cumulative count)", "False Alarms")

    # ── Cumulative Reward ─────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.plot(steps, log["smart_reward"], color=GREEN, lw=1.8, label="Smart Scan RL")
    ax4.plot(steps, log["open_reward"],  color=AMBER, lw=1.2, linestyle="--", label="Open Loop")
    ax4.axhline(0, color="#ffffff", lw=0.5, alpha=0.3)
    ax4.legend(fontsize=8)
    ax_style(ax4, "Cumulative Reward", "Reward")

    # ── Avg Intercept Delay ───────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 0])
    ax5.plot(steps, log["smart_avg_delay"], color=CYAN,  lw=1.8, label="Smart Scan RL")
    ax5.plot(steps, log["open_avg_delay"],  color=SLATE, lw=1.2, linestyle="--", label="Open Loop")
    ax5.legend(fontsize=8)
    ax_style(ax5, "Avg Intercept Delay (steps — lower is better)", "Avg Delay (steps)")

    # ── SINR ──────────────────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.plot(steps, log["sinr_db"], color=PURPLE, lw=1.6, label="SINR (dB)")
    ax6.legend(fontsize=8)
    ax_style(ax6, "SINR (dB) — Signal vs Noise Floor", "SINR (dB)")

    plt.savefig(out, bbox_inches="tight", facecolor="#0b1220")
    plt.close()
    print(f"[Saved] {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 2 — Comparison Chart (mirrors the live frontend chart exactly)
# ══════════════════════════════════════════════════════════════════════════════
def plot_comparison_chart(history, out):
    if not history:
        print("[Skip] No comparison history recorded.")
        return

    hist_steps    = [h["step"] for h in history]
    smart_dp      = [h["smart_scan"]["detection_prob"]  for h in history]
    open_dp       = [h["open_loop"]["detection_prob"]   for h in history]
    smart_ir      = [h["smart_scan"]["intercept_rate"]  for h in history]
    open_ir       = [h["open_loop"]["intercept_rate"]   for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=160)
    fig.suptitle("Comparison Chart — Smart Scan RL vs Open Loop\n"
                 "(Same data as the live frontend comparison graph)",
                 fontsize=12, fontweight="bold", color="#bae6fd")
    fig.patch.set_facecolor("#0b1220")

    for ax in (ax1, ax2):
        ax.set_facecolor("#0f1a2e")
        ax.tick_params(colors="#64748b")
        ax.spines[:].set_color("#27445b")
        ax.grid(True, color="#1e3a4f", linestyle=":")

    ax1.plot(hist_steps, smart_dp, color=CYAN,  lw=2.2, label="Smart Scan RL")
    ax1.plot(hist_steps, open_dp,  color=SLATE, lw=1.6, linestyle="--", label="Open Loop")
    ax1.fill_between(hist_steps, smart_dp, open_dp,
                     where=[s > o for s, o in zip(smart_dp, open_dp)],
                     alpha=0.15, color=CYAN)
    ax1.set_title("Detection Probability (%)", color="#bae6fd", fontweight="bold")
    ax1.set_xlabel("Step", color="#94a3b8")
    ax1.set_ylabel("Detection Prob (%)", color="#94a3b8")
    ax1.set_ylim(0, 100)
    ax1.legend(fontsize=9)

    ax2.plot(hist_steps, smart_ir, color=CYAN,  lw=2.2, label="Smart Scan RL")
    ax2.plot(hist_steps, open_ir,  color=SLATE, lw=1.6, linestyle="--", label="Open Loop")
    ax2.fill_between(hist_steps, smart_ir, open_ir,
                     where=[s > o for s, o in zip(smart_ir, open_ir)],
                     alpha=0.15, color=CYAN)
    ax2.set_title("Intercept Rate (%)", color="#bae6fd", fontweight="bold")
    ax2.set_xlabel("Step", color="#94a3b8")
    ax2.set_ylabel("Intercept Rate (%)", color="#94a3b8")
    ax2.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", facecolor="#0b1220")
    plt.close()
    print(f"[Saved] {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 3 — Spectrum Heatmap (band powers + scheduler dwell)
# ══════════════════════════════════════════════════════════════════════════════
def plot_spectrum_heatmap(log, out):
    steps      = len(log["step"])
    powers_mat = np.array(log["band_powers"]).T   # shape (36, steps)
    smart_acts = np.array(log["smart_band"])
    open_acts  = np.array(log["open_band"])

    # Subsample for readability if very long run
    max_cols = 1000
    if steps > max_cols:
        idx = np.linspace(0, steps - 1, max_cols, dtype=int)
        powers_mat = powers_mat[:, idx]
        smart_acts = smart_acts[idx]
        open_acts  = open_acts[idx]
        x_range    = max_cols
    else:
        x_range = steps

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 8), dpi=160, sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1]})
    fig.patch.set_facecolor("#0b1220")
    fig.suptitle("Spectrum Power Heatmap + Scheduler Dwell Pattern\n"
                 "(Power map identical to the SpectrumArray panel on the frontend)",
                 fontsize=12, fontweight="bold", color="#bae6fd")

    cmap_power = plt.cm.plasma
    for ax in (ax1, ax2):
        ax.set_facecolor("#0b1626")

    im = ax1.imshow(powers_mat, aspect="auto", cmap=cmap_power,
                    origin="lower", extent=[0, x_range, 0, 36],
                    vmin=-110, vmax=-30)
    ax1.scatter(range(x_range), smart_acts + 0.5,
                color=CYAN, s=5, alpha=0.7, marker="x", label="Smart Scan RL dwell")
    ax1.set_title("Smart Scan RL — Band Power Map + Dwell Selections",
                  color="#bae6fd", fontweight="bold", fontsize=11)
    ax1.set_ylabel("Band (0–35)", color="#94a3b8")
    ax1.legend(fontsize=8, loc="upper right")
    ax1.tick_params(colors="#64748b")

    ax2.imshow(powers_mat, aspect="auto", cmap=cmap_power,
               origin="lower", extent=[0, x_range, 0, 36],
               vmin=-110, vmax=-30)
    ax2.scatter(range(x_range), open_acts + 0.5,
                color=AMBER, s=4, alpha=0.65, marker="o", label="Open Loop dwell")
    ax2.set_title("Open Loop — Band Power Map + Dwell Pattern (Round-Robin)",
                  color="#bae6fd", fontweight="bold", fontsize=11)
    ax2.set_xlabel("Time Step (subsampled)", color="#94a3b8")
    ax2.set_ylabel("Band (0–35)", color="#94a3b8")
    ax2.legend(fontsize=8, loc="upper right")
    ax2.tick_params(colors="#64748b")

    cbar = fig.colorbar(im, ax=[ax1, ax2], orientation="vertical",
                        fraction=0.015, pad=0.01)
    cbar.set_label("Power (dBm)", color="#94a3b8", fontsize=9)
    cbar.ax.yaxis.set_tick_params(color="#64748b", labelcolor="#64748b")

    plt.tight_layout()
    plt.savefig(out, bbox_inches="tight", facecolor="#0b1220")
    plt.close()
    print(f"[Saved] {out}")


# ══════════════════════════════════════════════════════════════════════════════
# Export CSV + JSON
# ══════════════════════════════════════════════════════════════════════════════
def export_metrics(smart_final, open_final, total_steps, out_csv, out_json):
    def verdict(s, o, lower=False):
        if lower:
            return "Smart Scan Superior" if s < o else ("Open Loop Superior" if s > o else "Tied")
        return "Smart Scan Superior" if s > o else ("Open Loop Superior" if s < o else "Tied")

    def delta(s, o):
        diff = s - o
        pct  = (diff / abs(o) * 100) if o != 0 else 0
        sign = "+" if diff >= 0 else ""
        return f"{sign}{diff:.2f} ({sign}{pct:.1f}%)"

    rows = [
        {"Metric": "Detection Probability (%)",
         "Smart_Scan_RL": smart_final["detection_prob"],
         "Open_Loop":     open_final["detection_prob"],
         "Delta":         delta(smart_final["detection_prob"], open_final["detection_prob"]),
         "Verdict":       verdict(smart_final["detection_prob"], open_final["detection_prob"])},

        {"Metric": "Intercept Rate (%)",
         "Smart_Scan_RL": smart_final["intercept_rate"],
         "Open_Loop":     open_final["intercept_rate"],
         "Delta":         delta(smart_final["intercept_rate"], open_final["intercept_rate"]),
         "Verdict":       verdict(smart_final["intercept_rate"], open_final["intercept_rate"])},

        {"Metric": "False Alarms (cumulative)",
         "Smart_Scan_RL": smart_final["false_alarms"],
         "Open_Loop":     open_final["false_alarms"],
         "Delta":         delta(smart_final["false_alarms"], open_final["false_alarms"]),
         "Verdict":       verdict(smart_final["false_alarms"], open_final["false_alarms"], lower=True)},

        {"Metric": "Cumulative Reward",
         "Smart_Scan_RL": smart_final["reward"],
         "Open_Loop":     open_final["reward"],
         "Delta":         delta(smart_final["reward"], open_final["reward"]),
         "Verdict":       verdict(smart_final["reward"], open_final["reward"])},

        {"Metric": "Avg Intercept Delay (steps)",
         "Smart_Scan_RL": smart_final["avg_delay"],
         "Open_Loop":     open_final["avg_delay"],
         "Delta":         delta(smart_final["avg_delay"], open_final["avg_delay"]),
         "Verdict":       verdict(smart_final["avg_delay"], open_final["avg_delay"], lower=True)},
    ]

    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"[Saved] {out_csv}")

    report = {
        "project":          "EW Smart Scan — RL-Based Spectrum Scheduler",
        "architecture":     "DQN + Hybrid Urgency Scheduler (SmartScanScheduler)",
        "simulation_steps": total_steps,
        "data_source":      "Same RFEnvironment as live frontend (rf_dataset.csv or synthetic)",
        "band_count":       config.BAND_COUNT,
        "freq_range_ghz":   f"{config.BAND_BASE_GHZ} – {config.BAND_BASE_GHZ + config.BAND_COUNT * config.BAND_STEP_GHZ}",
        "smart_scan_final": smart_final,
        "open_loop_final":  open_final,
        "figures_of_merit": rows,
    }
    with open(out_json, "w") as f:
        json.dump(report, f, indent=4)
    print(f"[Saved] {out_json}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Generate frontend-identical artifacts from the live simulation stack")
    parser.add_argument("--steps",         type=int, default=2000,
                        help="Simulation steps to run (default: 2000 = 200s at 10Hz)")
    parser.add_argument("--compare-every", type=int, default=50,
                        help="History snapshot interval (matches frontend config)")
    parser.add_argument("--out-dir",       type=str, default=".",
                        help="Output directory for all generated files")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    print("=" * 62)
    print("  EW Smart Scan — Frontend-Identical Artifact Generator")
    print("=" * 62)
    print(f"  Steps:         {args.steps}")
    print(f"  Compare every: {args.compare_every} steps")
    print(f"  Output dir:    {args.out_dir}")
    print()

    # ── Run the simulation (same as main.py Simulation.run) ─────────────────
    print("[1/5] Running simulation (SmartScanScheduler + OpenLoopScheduler)...")
    log, history, smart_final, open_final = run_simulation(
        args.steps, args.compare_every)

    # ── Plots ────────────────────────────────────────────────────────────────
    print("[2/5] Generating KPI time-series graph...")
    plot_kpis(log, os.path.join(args.out_dir, "frontend_kpis.png"))

    print("[3/5] Generating comparison chart (mirrors frontend live graph)...")
    plot_comparison_chart(history, os.path.join(args.out_dir, "comparison_chart.png"))

    print("[4/5] Generating spectrum heatmap...")
    plot_spectrum_heatmap(log, os.path.join(args.out_dir, "spectrum_heatmap.png"))

    print("[5/5] Exporting figures of merit...")
    export_metrics(smart_final, open_final, args.steps,
                   os.path.join(args.out_dir, "figures_of_merit.csv"),
                   os.path.join(args.out_dir, "summary_report.json"))

    # ── Final summary ────────────────────────────────────────────────────────
    print()
    print("=" * 62)
    print("  ARTIFACTS GENERATED — all from live simulation stack")
    print("=" * 62)
    print(f"  Smart Scan  det prob : {smart_final['detection_prob']:5.1f}%")
    print(f"  Open Loop   det prob : {open_final['detection_prob']:5.1f}%")
    print(f"  Smart Scan  intercept: {smart_final['intercept_rate']:5.1f}%")
    print(f"  Open Loop   intercept: {open_final['intercept_rate']:5.1f}%")
    print(f"  Smart Scan  FA count : {smart_final['false_alarms']}")
    print(f"  Open Loop   FA count : {open_final['false_alarms']}")
    print(f"  Smart Scan  reward   : {smart_final['reward']}")
    print(f"  Open Loop   reward   : {open_final['reward']}")
    print("─" * 62)
    print(f"  frontend_kpis.png        : All 5 KPI cards over time")
    print(f"  comparison_chart.png     : Smart Scan vs Open Loop")
    print(f"  spectrum_heatmap.png     : Band powers + dwell pattern")
    print(f"  figures_of_merit.csv     : Final KPI table")
    print(f"  summary_report.json      : Full simulation metadata")
    print("=" * 62)


if __name__ == "__main__":
    main()