import os
import gc
import ast
import math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict

TOTAL_BANDS = 36

def _compute_threat_score(freq_mhz, pw_us=5.0, pri_us=200.0, pri_jitter_ratio=0.0):
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

def _parse_value(val):
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    val = str(val).strip()
    if val.startswith('['):
        try:
            return ast.literal_eval(val)
        except (ValueError, SyntaxError):
            return val
    try:
        return float(val)
    except ValueError:
        return val

class DataDrivenScenario:
    __slots__ = (
        'band_occupancy', 'band_powers', 'emitter_info_per_step',
        'emitter_metadata', 'receiver_params', 'num_steps',
        'dwell_centres_mhz',
    )
    def __init__(self, band_occupancy, band_powers, emitter_info_per_step,
                 emitter_metadata, receiver_params, num_steps, dwell_centres_mhz=None):
        self.band_occupancy = band_occupancy
        self.band_powers = band_powers
        self.emitter_info_per_step = emitter_info_per_step
        self.emitter_metadata = emitter_metadata
        self.receiver_params = receiver_params
        self.num_steps = num_steps
        self.dwell_centres_mhz = dwell_centres_mhz

class EWDatasetLoader:
    def __init__(self, data_root, mode="scan", split="val", target_steps=500, max_cache_size=8):
        self.data_root = Path(data_root)
        self.mode = mode
        self.split = split
        self.target_steps = target_steps
        self.max_cache_size = max_cache_size
        self._config_paths = self._discover_configs()
        self._cache = OrderedDict()
        print(f"[DatasetLoader] Found {len(self._config_paths)} '{mode}' configs in {self.data_root} for split {self.split}")
        print(f"[Memory Guard] LRU cache capped at {max_cache_size} scenarios to preserve RAM.")

    @property
    def num_configs(self):
        return len(self._config_paths)

    def load_scenario(self, config_idx):
        if config_idx in self._cache:
            self._cache.move_to_end(config_idx)
            return self._cache[config_idx]
        
        if len(self._cache) >= self.max_cache_size:
            self._cache.popitem(last=False)
            gc.collect()

        scenario = self._load_and_process(config_idx)
        self._cache[config_idx] = scenario
        return scenario

    def get_receiver_params(self, config_idx=0):
        config_path = self._config_paths[config_idx]
        return self._parse_receiver_metadata(config_path)

    def _discover_configs(self):
        processed_dir = self.data_root / self.mode / self.split / "processed"
        if not processed_dir.exists():
            raise FileNotFoundError(f"Dataset directory not found: {processed_dir}")
        configs = []
        for d in sorted(processed_dir.iterdir(),
                        key=lambda p: int(p.name.split('_')[-1]) if p.name.startswith('config_') else 0):
            if d.is_dir() and d.name.startswith("config_"):
                configs.append(d)
        return configs

    def _load_and_process(self, config_idx):
        config_path = self._config_paths[config_idx]
        emitter_metadata = self._parse_emitter_truth(config_path)
        pdw_df = self._load_pdw(config_path)
        if pdw_df is None or len(pdw_df) == 0:
            return self._empty_scenario(emitter_metadata, config_path)

        num_raw_steps = int(pdw_df['time_slot'].max()) + 1
        chunk_size = max(1, num_raw_steps // self.target_steps)
        actual_steps = max(1, min(self.target_steps, num_raw_steps // max(1, chunk_size)))

        band_occ_raw, band_pow_raw = self._build_band_data(pdw_df, num_raw_steps)
        ds_occ, ds_pow = self._downsample_band_data(
            band_occ_raw, band_pow_raw, chunk_size, actual_steps, num_raw_steps)
        ds_emitter_info = self._build_ds_emitter_info(
            pdw_df, emitter_metadata, chunk_size, actual_steps, num_raw_steps)
        receiver_params = self._parse_receiver_metadata(config_path)

        del pdw_df, band_occ_raw, band_pow_raw
        gc.collect()

        return DataDrivenScenario(
            band_occupancy=ds_occ,
            band_powers=ds_pow,
            emitter_info_per_step=ds_emitter_info,
            emitter_metadata=emitter_metadata,
            receiver_params=receiver_params,
            num_steps=actual_steps,
            dwell_centres_mhz=receiver_params.get('dwell_centres_mhz'),
        )

    def _parse_emitter_truth(self, config_path):
        csv_path = config_path / "emitter_behavior_truth.csv"
        if not csv_path.exists():
            csv_path = config_path / "emitter_behavior_truth_stare.csv"
        if not csv_path.exists():
            return []

        df = pd.read_csv(csv_path)
        emitters = []
        for _, row in df.iterrows():
            try:
                tid = str(row.get('transmitter_id', ''))
                eid = int(tid.split('_')[-1]) if '_' in tid else len(emitters)
                freqs = _parse_value(row.get('frequency_config//freqs_mhz', 5000.0))
                all_freqs = freqs if isinstance(freqs, list) else [float(freqs)]
                primary_freq = all_freqs[0]

                pws = _parse_value(row.get('pulse_width_config//pws_us', 5.0))
                primary_pw = pws[0] if isinstance(pws, list) else float(pws)

                pris = _parse_value(row.get('pri_config//pris_us', 200.0))
                primary_pri = pris[0] if isinstance(pris, list) else float(pris)

                pri_mode = str(row.get('pri_config/pri_mode', 'Fixed'))
                jitter_map = {'Fixed': 0.01, 'Jittered': 0.20, 'Staggered': 0.15, 'SwitchDwell': 0.10}
                pri_jitter = jitter_map.get(pri_mode, 0.05)

                threat = _compute_threat_score(primary_freq, primary_pw, primary_pri, pri_jitter)

                emitters.append({
                    'emitter_id': eid,
                    'transmitter_id': tid,
                    'function': str(row.get('function', 'Unknown')),
                    'freq_mode': str(row.get('frequency_config/freq_mode', 'FixedSingle')),
                    'primary_freq_mhz': primary_freq,
                    'all_freqs_mhz': all_freqs,
                    'pw_us': primary_pw,
                    'pri_us': primary_pri,
                    'pri_jitter_ratio': pri_jitter,
                    'pri_mode': pri_mode,
                    'threat_tier': threat,
                    'scan_type': str(row.get('scan_config/scan_type', 'Circular')),
                    'scan_rate_rpm': float(row.get('scan_config/scan_rate_rpm', 6.0)),
                    'power_w': float(row.get('power_config/power_w', 100.0)),
                    'gain_db': float(row.get('power_config/gain', 30.0)),
                })
            except Exception:
                continue
        return emitters

    def _load_pdw(self, config_path):
        parquet_files = sorted(config_path.glob("pdw_with_band_timeslot_*.parquet"))
        if not parquet_files:
            return None
        cols = ['RF', 'PulseWidth', 'PA', 'emitter_label', 'band', 'time_slot']
        dfs = [pd.read_parquet(pf, columns=cols) for pf in parquet_files]
        return pd.concat(dfs, ignore_index=True) if len(dfs) > 1 else dfs[0]

    def _build_band_data(self, pdw_df, num_raw_steps):
        band_occ = np.zeros((TOTAL_BANDS, num_raw_steps), dtype=np.int8)
        band_pow = np.full((TOTAL_BANDS, num_raw_steps), -200.0, dtype=np.float32)

        valid = ((pdw_df['band'] >= 0) & (pdw_df['band'] < TOTAL_BANDS) &
                 (pdw_df['time_slot'] >= 0) & (pdw_df['time_slot'] < num_raw_steps) &
                 (pdw_df['emitter_label'] >= 0))
        df = pdw_df[valid]

        bands = df['band'].values.astype(np.intp)
        ts = df['time_slot'].values.astype(np.intp)
        band_occ[bands, ts] = 1

        grouped = df.groupby(['band', 'time_slot'])['PA'].max()
        for (b, t), pa in grouped.items():
            band_pow[int(b), int(t)] = float(pa)
        return band_occ, band_pow

    def _downsample_band_data(self, band_occ, band_pow, chunk_size, actual_steps, num_raw_steps):
        limit = actual_steps * chunk_size
        occ_view = band_occ[:, :limit].reshape(TOTAL_BANDS, actual_steps, chunk_size)
        pow_view = band_pow[:, :limit].reshape(TOTAL_BANDS, actual_steps, chunk_size)
        ds_occ = occ_view.max(axis=2).astype(np.float32)
        ds_pow = pow_view.max(axis=2).astype(np.float32)
        return ds_occ, ds_pow

    def _build_ds_emitter_info(self, pdw_df, emitter_metadata, chunk_size, actual_steps, num_raw_steps):
        label_to_threat = {em['emitter_id']: em['threat_tier'] for em in emitter_metadata}
        valid = ((pdw_df['band'] >= 0) & (pdw_df['band'] < TOTAL_BANDS) &
                 (pdw_df['time_slot'] >= 0) & (pdw_df['time_slot'] < num_raw_steps) &
                 (pdw_df['emitter_label'] >= 0))
        df = pdw_df[valid].copy()
        df['ds_step'] = (df['time_slot'] // chunk_size).clip(upper=actual_steps - 1)

        grouped = (df.groupby(['ds_step', 'emitter_label', 'band'])['PA']
                     .max().reset_index())

        result = [[] for _ in range(actual_steps)]
        for ds, eid, band, pa in zip(grouped['ds_step'].values,
                                     grouped['emitter_label'].values,
                                     grouped['band'].values,
                                     grouped['PA'].values):
            threat = label_to_threat.get(int(eid), 0)
            result[int(ds)].append((int(eid), int(band), float(pa), threat))
        return result

    def _parse_receiver_metadata(self, config_path):
        csv_path = config_path / "receiver_metadata.csv"
        if not csv_path.exists():
            csv_path = config_path / "receiver_metadata_stare.csv"
        if not csv_path.exists():
            return {'sensitivity_dbm': -110.0, 'bandwidth_mhz': 500.0,
                    'dwell_centres_mhz': [], 'dwell_times_s': [],
                    'freq_range_mhz': [500.0, 18000.0]}

        row = pd.read_csv(csv_path).iloc[0]
        params = {
            'bandwidth_mhz': float(row.get('bandwidth_mhz', 500.0)),
            'sensitivity_dbm': float(row.get('sensitivity_dbm', -110.0)),
            'scan_mode': str(row.get('scan_mode', 'Scanning')),
            'gain_db': float(row.get('gain_db', 10.0)),
            'collection_time_s': float(row.get('collection_time_s', 30.0)),
        }
        for key, col in [('dwell_centres_mhz', 'dwell_centres_mhz'),
                         ('dwell_times_s', 'dwell_times_s'),
                         ('freq_range_mhz', 'freq_range_mhz')]:
            val = _parse_value(row.get(col, '[]'))
            params[key] = val if isinstance(val, list) else []
        return params

    def _empty_scenario(self, emitter_metadata, config_path):
        receiver_params = self._parse_receiver_metadata(config_path)
        return DataDrivenScenario(
            band_occupancy=np.zeros((TOTAL_BANDS, self.target_steps), dtype=np.float32),
            band_powers=np.full((TOTAL_BANDS, self.target_steps), -200.0, dtype=np.float32),
            emitter_info_per_step=[[] for _ in range(self.target_steps)],
            emitter_metadata=emitter_metadata,
            receiver_params=receiver_params,
            num_steps=self.target_steps,
            dwell_centres_mhz=receiver_params.get('dwell_centres_mhz'),
        )