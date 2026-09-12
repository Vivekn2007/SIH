import { useEffect, useRef } from 'react'
import { Panel } from './Panel'
import type { Emitter, Telemetry } from '../types'
type Filter = 'open' | 'smart' | 'both'

function renderRadar(canvas: HTMLCanvasElement, emitters: Emitter[], smartBand: number, openBand: number, filter: Filter, phase: number) {
  const ctx = canvas.getContext('2d')!; const rect = canvas.getBoundingClientRect(); const dpr = devicePixelRatio || 1
  if(canvas.width !== Math.round(rect.width*dpr) || canvas.height !== Math.round(rect.height*dpr)){canvas.width = Math.round(rect.width*dpr); canvas.height = Math.round(rect.height*dpr)} ctx.setTransform(dpr,0,0,dpr,0,0)
  const w = rect.width, h = rect.height, r = Math.min(w * .62, h * .78) / 2, cx = w * .54, cy = h * .50
  ctx.clearRect(0, 0, w, h); ctx.save(); ctx.translate(cx, cy)
  ctx.strokeStyle = 'rgba(34,211,238,.32)'; ctx.lineWidth = 1
  for (let i=1;i<=6;i++) { ctx.beginPath();ctx.arc(0,0,r*i/6,0,Math.PI*2);ctx.stroke() }
  for (let deg=0;deg<360;deg+=15) { const a=(deg-90)*Math.PI/180;ctx.beginPath();ctx.moveTo(0,0);ctx.lineTo(Math.cos(a)*r,Math.sin(a)*r);ctx.strokeStyle=deg%45===0?'rgba(34,211,238,.35)':'rgba(34,211,238,.13)';ctx.stroke() }
  ctx.font='10px "JetBrains Mono", monospace';ctx.textAlign='center';ctx.fillStyle='#cbd5e1'
  for (let deg=0;deg<360;deg+=30) { const a=(deg-90)*Math.PI/180;ctx.fillText(`${deg}°`, Math.cos(a)*(r+15), Math.sin(a)*(r+15)+3) }
  // thin sweep with a deliberately soft trailing arc
  const a=phase-Math.PI/2; ctx.save();ctx.rotate(a); const gradient=ctx.createLinearGradient(0,0,r,0);gradient.addColorStop(0,'rgba(34,211,238,.95)');gradient.addColorStop(1,'rgba(34,211,238,.25)');ctx.strokeStyle=gradient;ctx.lineWidth=2;ctx.beginPath();ctx.moveTo(0,0);ctx.lineTo(r,0);ctx.stroke();
  for(let i=1;i<=12;i++){ctx.strokeStyle=`rgba(34,211,238,${.13*(1-i/13)})`;ctx.lineWidth=1;ctx.beginPath();ctx.arc(0,0,r, -i*.055, -i*.055+.01);ctx.stroke()}ctx.restore()
  emitters.forEach((e, index) => { const distance=r*Math.min(e.range_km/300,1); const angle=(e.aoa_deg-90)*Math.PI/180; const x=Math.cos(angle)*distance,y=Math.sin(angle)*distance; const threat=e.threat_level==='High'; const locked=((filter !== 'open' && e.band_id===smartBand) || (filter !== 'smart' && e.band_id===openBand)); ctx.save();ctx.translate(x,y)
    if(locked){ctx.strokeStyle='#22d3ee';ctx.shadowColor='#22d3ee';ctx.shadowBlur=10;ctx.lineWidth=2;[8,12].forEach(rr=>{ctx.beginPath();ctx.arc(0,0,rr,0,Math.PI*2);ctx.stroke()});ctx.shadowBlur=0}
    if(threat){ctx.fillStyle='#ef4444';ctx.beginPath();ctx.moveTo(0,-7);ctx.lineTo(6,6);ctx.lineTo(-6,6);ctx.closePath();ctx.fill()} else if(e.class==='Unknown'){ctx.fillStyle='#e5e7eb';ctx.strokeStyle='#22d3ee';ctx.lineWidth=1;ctx.fillRect(-4,-4,8,8);ctx.strokeRect(-4,-4,8,8)} else {ctx.fillStyle='#22d3ee';ctx.beginPath();ctx.arc(0,0,4,0,Math.PI*2);ctx.fill()}
    if(locked){ctx.fillStyle='#22d3ee';ctx.font='10px "JetBrains Mono"';ctx.fillText('LOCKED',0,-17)}ctx.restore()
  });ctx.restore()
}
export function RadarPpi({ telemetry, filter }: { telemetry: Telemetry | null; filter: Filter }) {
  const ref=useRef<HTMLCanvasElement>(null), data=useRef({ emitters: [] as Emitter[], band: -1, openBand: -1, filter }); data.current={emitters:telemetry?.ground_truth.active_emitters ?? [],band:telemetry?.smart_scan.dwelling_band ?? -1,openBand:telemetry?.open_loop.dwelling_band ?? -1,filter}
  useEffect(()=>{let frame=0;const start=performance.now();const draw=(now:number)=>{if(ref.current)renderRadar(ref.current,data.current.emitters,data.current.band,data.current.openBand,data.current.filter,(now-start)/1100);frame=requestAnimationFrame(draw)};frame=requestAnimationFrame(draw);return()=>cancelAnimationFrame(frame)},[])
  const es=telemetry?.ground_truth.active_emitters ?? []; const threats=es.filter(e=>e.threat_level==='High').length;const locked=es.filter(e=>(filter!=='open' && e.band_id===telemetry?.smart_scan.dwelling_band)||(filter!=='smart' && e.band_id===telemetry?.open_loop.dwelling_band)).length
  const receiver=telemetry?.receiver
  const latitude=receiver?`${Math.abs(receiver.lat).toFixed(4)}° ${receiver.lat>=0?'N':'S'}`:'—'
  const longitude=receiver?`${Math.abs(receiver.lon).toFixed(4)}° ${receiver.lon>=0?'E':'W'}`:'—'
  return <Panel title="RADAR PPI — 360° SITUATIONAL AWARENESS" className="radar-panel"><div className="radar-body"><div className="radar-info"><p>RANGE MAX: <b>300 km</b></p><p>RINGS: <b>6 (50 km)</b></p><p>MODE: <b>ES SURVEY</b></p><p>ANTENNA: <b>4-AXIS</b></p><p>LAT: <b>{latitude}</b></p><p>LON: <b>{longitude}</b></p></div><canvas className="radar-canvas" ref={ref}/><div className="radar-readout"><p>TRACKS: <b>{es.length}</b></p><p>LOCKED: <b>{locked}</b></p><p>THREATS: <b className="red">{threats}</b></p><hr/><p>JAMMING: <b>OFF</b></p><p>GEOFENCE: <b className="green">ON</b></p><p>AUTO-CLASS: <b className="green">ON</b></p><p>TRACK FUSION: <b className="green">ON</b></p></div></div><div className="legend"><span className="dot cyan"/>Friendly / Unknown <span className="triangle"/>Hostile / Threat <span className="square"/>Unclassified <span className="lock-ring"/>Locked Target</div></Panel>
}
