import { useCallback, useEffect, useRef, useState } from 'react'
import type { Telemetry } from '../types'

export function useTelemetry() {
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null)
  const [connected, setConnected] = useState(false)
  const socket = useRef<WebSocket | null>(null)
  useEffect(() => {
    let timer: number
    const connect = () => {
      const protocol = location.protocol === 'https:' ? 'wss' : 'ws'
      const ws = new WebSocket(`${protocol}://${location.hostname}:8000/ws/telemetry`)
      socket.current = ws
      ws.onopen = () => setConnected(true)
      ws.onmessage = (event) => setTelemetry(JSON.parse(event.data) as Telemetry)
      ws.onclose = () => { setConnected(false); timer = window.setTimeout(connect, 1200) }
    }
    connect()
    return () => { window.clearTimeout(timer); socket.current?.close() }
  }, [])
  const setCompareEvery = useCallback((value: number) => socket.current?.readyState === WebSocket.OPEN && socket.current.send(JSON.stringify({ type: 'set_compare_every', value })), [])
  return { telemetry, connected, setCompareEvery }
}
