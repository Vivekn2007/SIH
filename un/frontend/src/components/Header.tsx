import { RadioTower } from 'lucide-react'
export function Header({ online }: { online: boolean }) {
  const utc = new Date().toISOString().replace('T', ' ').slice(0, 19) + 'Z'
  return <header className="header"><div className="brand"><RadioTower size={31}/><div><h1>SMART SCAN COMMAND CENTER</h1><p>AI-Powered RF/EW Spectrum Scheduler</p></div><span className={`online ${online ? '' : 'offline'}`}><i/>SYSTEM ONLINE</span></div><div className="meta"><span>MISSION&nbsp; SIH26055</span><span>ENVIRONMENT&nbsp; RF REPLAY</span><span>MODE&nbsp; EVALUATION</span><span>{utc}</span><b>DETECT / CLASSIFY / ADAPT</b></div></header>
}
