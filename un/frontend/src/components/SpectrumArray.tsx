import { Panel } from './Panel'
import type { Telemetry } from '../types'
type Filter = 'open' | 'smart' | 'both'
const range=(i:number)=>`${(0.5+i*.5).toFixed(1)} – ${(1+i*.5).toFixed(1)} GHz`
function Wave({ seed, active }: { seed:number; active:boolean }) { const points=Array.from({length:22},(_,i)=>{const x=i*3;const y=17-(Math.sin(i*(.65+seed*.07)+seed)*5+(active?Math.sin(i*2.4)*5:0)+((i*seed)%5));return `${x},${y}`}).join(' '); return <svg className="wave" viewBox="0 0 66 20"><polyline points={points}/></svg> }
export function SpectrumArray({ telemetry, filter }: { telemetry: Telemetry | null; filter: Filter }) {
 const emitters=telemetry?.ground_truth.active_emitters ?? [];const open=telemetry?.open_loop.dwelling_band,smart=telemetry?.smart_scan.dwelling_band
 return <Panel title="36-BAND RF SPECTRUM ARRAY" className="spectrum-panel" action={<small>REAL-TIME SPECTRAL MONITORING</small>}><div className="bands">{Array.from({length:36},(_,id)=>{const threat=emitters.some(e=>e.band_id===id&&e.threat_level==='High');const selected=(filter==='open'&&id===open)||(filter==='smart'&&id===smart)||(filter==='both'&&(id===open||id===smart));return <div className={`band ${threat?'threat':''} ${selected?'scanning':''}`} key={id}><b>B{id}</b><span>{range(id)}</span><Wave seed={id} active={selected||threat}/>{threat?<em>THREAT</em>:selected?<em>SCANNING</em>:<em>IDLE</em>}</div>})}</div></Panel>
}
