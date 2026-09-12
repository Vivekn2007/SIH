export type Cumulative = { detection_prob: number; intercept_rate: number; false_alarms: number; avg_delay: number; reward: number; correct_predictions: number }
export type Emitter = { id: string; band_id: number; freq_ghz: number; power_dbm: number; aoa_deg: number; pulse_width_us: number; class: string; threat_level: 'Low' | 'Medium' | 'High' }
export type Telemetry = {
  step: number; total_steps: number; sim_time: string; ground_truth: { active_emitters: Emitter[] };
  open_loop: { dwelling_band: number; hit: boolean; cumulative: Cumulative };
  smart_scan: { dwelling_band: number; hit: boolean; cumulative: Cumulative };
  event: { text: string; type: 'detection' | 'tracking' | 'alert' | 'system'; severity: 'ok' | 'warn' | 'threat' };
  comparison_history: { step: number; open_loop: Cumulative; smart_scan: Cumulative }[]; compare_every: number;
  data_source: { name: string; mode: string }; sinr_db: number;
}
