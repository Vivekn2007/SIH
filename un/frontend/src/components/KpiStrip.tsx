import { Activity, Crosshair, Gauge, Radar, ScanLine, Timer } from 'lucide-react'
import type { Telemetry } from '../types'
const fmt = (v: number | undefined, suffix = '') => v === undefined ? '—' : `${v}${suffix}`
export function KpiStrip({ telemetry }: { telemetry: Telemetry | null }) {
  const smart = telemetry?.smart_scan.cumulative, open = telemetry?.open_loop.cumulative
  const cards = [
    ['Detection Probability', <Crosshair/>, fmt(smart?.detection_prob, '%'), `vs Open-Loop: ${fmt(open?.detection_prob, '%')}`],
    ['False Alarms', <Activity/>, fmt(smart?.false_alarms), `vs Open-Loop: ${fmt(open?.false_alarms)}`],
    ['Avg Intercept Delay', <Timer/>, fmt(smart?.avg_delay, ' steps'), `vs Open-Loop: ${fmt(open?.avg_delay, ' steps')}`],
    ['Intercept Rate', <Radar/>, fmt(smart?.intercept_rate, '%'), `vs Open-Loop: ${fmt(open?.intercept_rate, '%')}`],
    ['Active Emitters', <ScanLine/>, fmt(telemetry?.ground_truth.active_emitters.length), 'from ground truth'],
    ['Reward / Cost', <Gauge/>, fmt(smart?.reward), `vs Open-Loop: ${fmt(open?.reward)}`],
  ]
  return <div className="kpi-strip">{cards.map(([label, icon, value, secondary]) => <div className="kpi" key={String(label)}><span className="kpi-icon">{icon}</span><div><label>{label}</label><strong>{value}</strong><small>{secondary}</small></div><svg viewBox="0 0 76 22" aria-hidden="true"><path d="M1 18 L10 15 L16 17 L25 8 L31 14 L39 10 L48 13 L57 5 L64 9 L75 2"/></svg></div>)}</div>
}
