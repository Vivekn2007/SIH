import type { ReactNode } from 'react'
export function Panel({ title, action, className = '', children }: { title?: string; action?: ReactNode; className?: string; children: ReactNode }) {
  return <section className={`panel ${className}`}>{title && <div className="panel-head"><h2>{title}</h2>{action}</div>}{children}</section>
}
