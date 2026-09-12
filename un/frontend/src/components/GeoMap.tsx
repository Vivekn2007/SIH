import { useEffect, useRef } from 'react'
import L from 'leaflet'
import { Panel } from './Panel'
import type { Telemetry } from '../types'
export function GeoMap({ telemetry }: { telemetry: Telemetry | null }) {
  const node=useRef<HTMLDivElement>(null),map=useRef<L.Map>(),layers=useRef<L.LayerGroup>()
  const receiver=telemetry?.receiver
  const lastCenter=useRef('')
  useEffect(()=>{if(!node.current||map.current)return;map.current=L.map(node.current,{zoomControl:false}).setView([34.0837,74.7973],9);L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',{attribution:'© OpenStreetMap'}).addTo(map.current);layers.current=L.layerGroup().addTo(map.current);return()=>{map.current?.remove();map.current=undefined;layers.current=undefined;lastCenter.current=''}},[])
  useEffect(()=>{const layer=layers.current;if(!layer)return;layer.clearLayers();if(!receiver)return;const centerKey = receiver.lat + ',' + receiver.lon; if(lastCenter.current !== centerKey){ map.current?.setView([receiver.lat,receiver.lon],9,{animate:false}); lastCenter.current=centerKey; }const center=[receiver.lat,receiver.lon] as L.LatLngExpression;L.circleMarker(center,{radius:9,color:'#22d3ee',fillOpacity:.8}).bindTooltip(Object.assign(document.createElement('span'),{textContent:receiver.label})).addTo(layer);(telemetry?.ground_truth.active_emitters??[]).forEach((e)=>{if(!Number.isFinite(e.lat)||!Number.isFinite(e.lon))return;const pos:[number,number]=[e.lat,e.lon];L.circleMarker(pos,{radius:6,color:e.threat_level==='High'?'#ef4444':'#22d3ee',fillOpacity:.85}).bindTooltip(Object.assign(document.createElement('span'),{textContent:`${e.id} · ${e.class}`})).addTo(layer)})},[telemetry,receiver])
  return <Panel title="GEO MAP — RECEIVER & EMITTER TRACKS (SIMULATED RANGE)" className="map-panel"><div ref={node} className="map"/></Panel>
}
