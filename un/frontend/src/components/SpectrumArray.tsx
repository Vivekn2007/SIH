import { useEffect, useState } from 'react'
import { Panel } from './Panel'
import type { Telemetry } from '../types'
type Filter = 'open' | 'smart' | 'both'
const formatRange = (min: number, max: number) => `${min.toFixed(1)} – ${max.toFixed(1)} GHz`
function Wave({ power, step }: { power: number | undefined; step: number | undefined }) {
  const [samples, setSamples] = useState<number[]>([])
  useEffect(() => {
    if (power === undefined) { setSamples([]); return }
    setSamples(previous => [...previous.slice(-21), power])
  }, [power, step])
  const points = samples.map((value, i) => `${i*3},${19-Math.max(0,Math.min(1,(value+120)/120))*18}`).join(' ')
  return <svg className="wave" viewBox="0 0 66 20" aria-label="Received power history"><polyline points={points}/></svg>
}
export function SpectrumArray({ telemetry, filter }: { telemetry: Telemetry | null; filter: Filter }) {
 const emitters=telemetry?.ground_truth.active_emitters ?? [];const open=telemetry?.open_loop.dwelling_band,smart=telemetry?.smart_scan.dwelling_band
 const bandConfig = new Map((telemetry?.band_config ?? []).map(band => [band.band_id, band]))
 // The IDs make the 6 × 6 layout stable; range text is shown only when supplied by the backend simulator.
 return <Panel title="36-BAND RF SPECTRUM ARRAY" className="spectrum-panel" action={<small>REAL-TIME SPECTRAL MONITORING</small>}><div className="bands">{Array.from({length:36},(_,id)=>{
   const band=bandConfig.get(id)
   const threat=emitters.some(e=>e.band_id===id&&e.threat_level==='High')
   const active=emitters.some(e=>e.band_id===id)
   const selected=(filter==='open'&&id===open)||(filter==='smart'&&id===smart)||(filter==='both'&&(id===open||id===smart))
   // Purple = scanner is dwelling on a band that has an active emitter (confirmed intercept)
   const intercepted = selected && active
   const cls = intercepted ? 'intercepted' : threat ? 'threat' : selected ? 'scanning' : ''
   const label = intercepted ? <em className="label-intercepted">INTERCEPT</em> : threat ? <em>THREAT</em> : selected ? <em>SCANNING</em> : <em>{band ? 'IDLE' : 'WAIT'}</em>
   return <div className={`band ${cls}`} key={id}><b>B{id}</b><span>{band ? formatRange(band.min_ghz, band.max_ghz) : 'SIMULATOR WAITING'}</span><Wave power={telemetry?.bands?.power_dbm[id]} step={telemetry?.step}/>{label}</div>
 })}</div></Panel>
}
