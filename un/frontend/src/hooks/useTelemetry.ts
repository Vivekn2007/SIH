import { useCallback, useEffect, useRef, useState } from 'react'
import type { Telemetry } from '../types'

export function useTelemetry() {
  const [telemetry, setTelemetry] = useState<Telemetry | null>(null)
  const [connected, setConnected] = useState(false)
  const socket = useRef<WebSocket | null>(null)
  useEffect(() => {
    let stopped = false
    let timer: number | undefined
    let lastMessage = 0
    const connect = () => {
      if (stopped) return
      const protocol = location.protocol === 'https:' ? 'wss' : 'ws'
      const ws = new WebSocket(`${protocol}://${location.hostname}:8000/ws/telemetry`)
      socket.current = ws
      lastMessage = Date.now()
      ws.onmessage = (event) => {
        if (stopped) return
        try {
          const payload = JSON.parse(event.data)
          if (!payload.ground_truth || !payload.open_loop || !payload.smart_scan || !Array.isArray(payload.band_config)) return
          lastMessage = Date.now()
          setTelemetry(payload as Telemetry)
          setConnected(true)
        } catch { setConnected(false) }
      }
      ws.onerror = () => setConnected(false)
      ws.onclose = () => {
        if (stopped) return
        setConnected(false)
        setTelemetry(null)
        timer = window.setTimeout(connect, 1200)
      }
    }
    connect()
    const watchdog = window.setInterval(() => {
      if (Date.now() - lastMessage > 5000) socket.current?.close()
    }, 1000)
    return () => {
      stopped = true
      window.clearTimeout(timer)
      window.clearInterval(watchdog)
      socket.current?.close()
    }
  }, [])
  const setCompareEvery = useCallback((value: number) => {
    if (Number.isInteger(value) && value >= 10 && value <= 500 && socket.current?.readyState === WebSocket.OPEN)
      socket.current.send(JSON.stringify({ type: 'set_compare_every', value }))
  }, [])
  return { telemetry, connected, setCompareEvery }
}
