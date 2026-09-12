import { RadioTower } from 'lucide-react'
export function Header({ online }: { online: boolean }) {
  return <header className="header"><div className="brand"><RadioTower size={31}/><div><h1>SMART SCAN COMMAND CENTER</h1><p>AI-Powered RF/EW Spectrum Scheduler</p></div><span className={`online ${online ? '' : 'offline'}`}><i/>{online ? 'SYSTEM ONLINE' : 'SIMULATOR OFFLINE'}</span></div></header>
}
