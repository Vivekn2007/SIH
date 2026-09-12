import { useState } from 'react'
import { Map, Radar } from 'lucide-react'
import { Header } from './components/Header'
import { KpiStrip } from './components/KpiStrip'
import { RadarPpi } from './components/RadarPpi'
import { SpectrumArray } from './components/SpectrumArray'
import { EventLog } from './components/EventLog'
import { RecentEmitters } from './components/RecentEmitters'
import { ComparisonChart } from './components/ComparisonChart'
import { GeoMap } from './components/GeoMap'
import { Footer } from './components/Footer'
import { useTelemetry } from './hooks/useTelemetry'

export default function App() { const {telemetry,connected,setCompareEvery}=useTelemetry();const [tab,setTab]=useState<'radar'|'map'>('radar');const [filter,setFilter]=useState<'open'|'smart'|'both'>('smart');return <main><Header online={connected}/><KpiStrip telemetry={telemetry}/><nav className="view-tabs"><button className={tab==='radar'?'active':''} onClick={()=>setTab('radar')}><Radar size={15}/>RADAR PPI</button><button className={tab==='map'?'active':''} onClick={()=>setTab('map')}><Map size={15}/>GEO MAP</button></nav>{tab==='radar'?<div className="dashboard-grid"><RadarPpi telemetry={telemetry} filter={filter}/><div className="right-stack"><SpectrumArray telemetry={telemetry} filter={filter}/><EventLog telemetry={telemetry}/><RecentEmitters telemetry={telemetry}/></div><ComparisonChart telemetry={telemetry} setCompareEvery={setCompareEvery}/></div>:<GeoMap telemetry={telemetry}/>}<Footer telemetry={telemetry} filter={filter} setFilter={setFilter}/></main> }
