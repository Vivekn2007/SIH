export type Cumulative = { detection_prob: number; intercept_rate: number; false_alarms: number; avg_delay: number; reward: number }
export type Emitter = { id: string; band_id: number; freq_ghz: number; lat: number; lon: number; range_km: number; power_dbm: number; aoa_deg: number; pulse_width_us: number; class: string; threat_level: 'Low' | 'Medium' | 'High' }
export type BandConfig = { band_id: number; min_ghz: number; max_ghz: number }
export type Receiver = { lat: number; lon: number; label: string }
export type Telemetry = {
  step: number; total_steps: number; sim_time: string; ground_truth: { active_emitters: Emitter[] };
  open_loop: { dwelling_band: number; hit: boolean; cumulative: Cumulative };
  smart_scan: { dwelling_band: number; hit: boolean; cumulative: Cumulative };
  event: { text: string; type: 'detection' | 'tracking' | 'alert' | 'system'; severity: 'ok' | 'warn' | 'threat' };
  comparison_history: { step: number; open_loop: Cumulative; smart_scan: Cumulative }[]; compare_every: number;
  data_source: { name: string; mode: string }; sinr_db: number | null;
  band_config: BandConfig[]; receiver: Receiver;
  bands: { power_dbm: number[]; is_transmitting: boolean[] };
}
