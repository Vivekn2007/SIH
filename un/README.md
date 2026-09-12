# Smart Scan Command Center

Real-time Electronic Support receiver-scheduler dashboard for **Team Perceptrons · SIH26055**. It compares a baseline open-loop scheduler and a Smart Scan integration seam against exactly the same 36-band RF ground truth. The project contains only the requested scheduler integration seams.

## Run it

Requirements: Python 3.10+ and Node.js 20+.

```powershell
# Terminal 1 — from the project root
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r backend\requirements.txt
uvicorn backend.main:app --reload --port 8000
```

```powershell
# Terminal 2 — from the project root
cd frontend
npm install
npm run dev
```

Open the Vite URL (normally `http://localhost:5173`). The FastAPI server publishes telemetry at 10 Hz over `ws://localhost:8000/ws/telemetry`.

## Data source

By default the NumPy environment generates a plausible, vectorized 36-band RF stream. To replay data, place `rf_dataset.csv` in the repository root or set `DATA_SOURCE` before starting the server. Required CSV columns are:

```text
timestep,band_id,freq_ghz,is_transmitting,power_dbm,aoa_deg,pulse_width_us,emitter_id,emitter_class,threat_level
```

The footer automatically identifies `SYNTHETIC` versus `LIVE REPLAY`. Every KPI and chart value comes from the environment state, policy band selection, hit/miss check, and cumulative aggregator in `backend/main.py`.

## Plug in your own algorithms

Only edit these two integration points:

1. `backend/schedulers/open_loop.py` — replace the fixed round-robin `select_band` placeholder with your open-loop policy.
2. `backend/schedulers/smart_scan.py` — replace the strongest-observed-band placeholder with your model invocation. The commented model-loading seam is deliberately the only model reference.

Set integration configuration in `backend/config.py` or via environment variables:

| Setting | Purpose |
| --- | --- |
| `OPEN_LOOP_IMPL` | Label/config for your open-loop integration |
| `SMART_SCAN_MODEL_PATH` | Path your own smart-policy integration will consume |
| `DATA_SOURCE` | Optional CSV replay file path |
| `COMPARE_EVERY_N_STEPS` | Initial sampling interval for the comparison chart |

The UI can change `COMPARE_EVERY_N_STEPS` live through the WebSocket; the backend owns the history used by the Recharts chart.

## Architecture

- `backend/environment.py`: CSV replay or NumPy ground truth only.
- `backend/schedulers/base.py`: stable policy contract.
- `backend/main.py`: scheduler-agnostic simulation/aggregation, FastAPI, WebSockets.
- `frontend/src/hooks/useTelemetry.ts`: the sole source of React telemetry state.
- `frontend/src/components`: one component per reviewed dashboard panel, including a 60 FPS Canvas radar PPI and a distinct Leaflet GEO MAP tab.
