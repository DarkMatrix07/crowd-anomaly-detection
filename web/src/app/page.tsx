'use client'

import { useCallback, useEffect, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { Activity, Camera, LayoutDashboard, RefreshCw, Shield, Video, Wifi, WifiOff } from 'lucide-react'
import clsx from 'clsx'

import MetricCard    from '@/components/MetricCard'
import RiskTimeline  from '@/components/RiskTimeline'
import AlertFeed     from '@/components/AlertFeed'
import SidePanel     from '@/components/SidePanel'
import LiveDetection from '@/components/LiveDetection'

import {
  checkHealth, fetchSummary, fetchThresholds,
  updateThresholds, acknowledgeAlert,
  type Alert, type LevelCounts, type ThresholdProfile, type TimelinePoint,
} from '@/lib/api'

const REFRESH_INTERVAL = 10_000
type Tab = 'dashboard' | 'detection'

export default function Dashboard() {
  const [activeTab, setActiveTab]         = useState<Tab>('dashboard')
  const [connected, setConnected]         = useState<boolean | null>(null)
  const [alerts, setAlerts]               = useState<Alert[]>([])
  const [timeline, setTimeline]           = useState<TimelinePoint[]>([])
  const [levelCounts, setLevelCounts]     = useState<LevelCounts>({ low: 0, medium: 0, high: 0 })
  const [thresholds, setThresholds]       = useState<ThresholdProfile>({
    profile_name: 'default', low: 0.30, medium: 0.60, high: 0.85,
  })
  const [cameras, setCameras]             = useState<string[]>([])
  const [selectedCamera, setSelectedCamera] = useState('ALL')
  const [lastRefresh, setLastRefresh]     = useState<Date | null>(null)
  const [isRefreshing, setIsRefreshing]   = useState(false)
  const [prefillAckId, setPrefillAckId]   = useState<number | null>(null)

  const refresh = useCallback(async () => {
    setIsRefreshing(true)
    try {
      const [healthy, summary, thresh] = await Promise.all([
        checkHealth(),
        fetchSummary(200, selectedCamera === 'ALL' ? undefined : selectedCamera),
        fetchThresholds(),
      ])
      setConnected(healthy)
      setAlerts(summary.alerts)
      setTimeline(summary.timeline)
      setLevelCounts(summary.level_counts)
      setThresholds(thresh)
      setCameras([...new Set(summary.alerts.map((a) => a.camera_id))].sort())
      setLastRefresh(new Date())
    } catch {
      setConnected(false)
    } finally {
      setIsRefreshing(false)
    }
  }, [selectedCamera])

  useEffect(() => {
    refresh()
    const id = setInterval(refresh, REFRESH_INTERVAL)
    return () => clearInterval(id)
  }, [refresh])

  async function handleUpdateThresholds(profile: ThresholdProfile) {
    const updated = await updateThresholds(profile)
    setThresholds(updated)
  }

  async function handleAcknowledge(id: number, operator: string, note: string) {
    await acknowledgeAlert(id, operator, note)
    await refresh()
  }

  function handleAckFromTable(id: number) {
    setPrefillAckId(id)
    document.getElementById('side-panel')?.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <div className="min-h-[100dvh] bg-bg text-text">

      {/* ── Header ─────────────────────────────────────────────────── */}
      <header className="sticky top-0 z-50 border-b border-border bg-bg/80 backdrop-blur-md">
        <div className="max-w-[1400px] mx-auto px-6 h-14 flex items-center gap-6">

          {/* Logo */}
          <div className="flex items-center gap-2.5 shrink-0">
            <div className="w-7 h-7 rounded-lg bg-accent/10 border border-accent/20 flex items-center justify-center">
              <Shield size={14} className="text-accent" />
            </div>
            <span className="font-semibold text-sm tracking-tight">Crowd Monitor</span>
          </div>

          {/* Tab navigation */}
          <nav className="flex items-center gap-1 bg-surface border border-border rounded-lg p-1">
            {([
              { id: 'dashboard', icon: LayoutDashboard, label: 'Dashboard' },
              { id: 'detection', icon: Video,           label: 'Live Detection' },
            ] as const).map(({ id, icon: Icon, label }) => (
              <button
                key={id}
                onClick={() => setActiveTab(id)}
                className={clsx(
                  'flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-all',
                  activeTab === id
                    ? 'bg-accent/10 text-accent border border-accent/20'
                    : 'text-muted hover:text-text',
                )}
              >
                <Icon size={12} />
                {label}
              </button>
            ))}
          </nav>

          {/* Right controls */}
          <div className="flex items-center gap-3 flex-1 justify-end">
            {activeTab === 'dashboard' && cameras.length > 0 && (
              <div className="flex items-center gap-2">
                <Camera size={13} className="text-muted" />
                <select
                  value={selectedCamera}
                  onChange={(e) => setSelectedCamera(e.target.value)}
                  className="bg-surface border border-border rounded-lg px-3 py-1.5 text-xs text-text focus:outline-none focus:border-accent/50 transition-colors cursor-pointer"
                >
                  <option value="ALL">All cameras</option>
                  {cameras.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </div>
            )}

            {lastRefresh && activeTab === 'dashboard' && (
              <span className="text-xs text-muted hidden sm:block">
                Updated {lastRefresh.toLocaleTimeString()}
              </span>
            )}

            {activeTab === 'dashboard' && (
              <button
                onClick={refresh}
                disabled={isRefreshing}
                className="flex items-center gap-1.5 text-xs text-muted hover:text-text border border-border hover:border-border/80 rounded-lg px-3 py-1.5 transition-all disabled:opacity-50"
              >
                <RefreshCw size={12} className={clsx(isRefreshing && 'animate-spin')} />
                Refresh
              </button>
            )}

            <div className={clsx(
              'flex items-center gap-1.5 text-xs font-medium rounded-full px-3 py-1 border',
              connected === true  && 'bg-low/10 text-low border-low/30',
              connected === false && 'bg-high/10 text-high border-high/30',
              connected === null  && 'bg-muted/10 text-muted border-muted/20',
            )}>
              {connected === true ? <Wifi size={11} /> : <WifiOff size={11} />}
              {connected === true ? 'Live' : connected === false ? 'Offline' : 'Connecting…'}
            </div>
          </div>
        </div>
      </header>

      {/* ── Content ────────────────────────────────────────────────── */}
      <main className="max-w-[1400px] mx-auto px-6 py-6">

        {/* Offline banner */}
        <AnimatePresence>
          {connected === false && (
            <motion.div
              initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }} exit={{ opacity: 0, height: 0 }}
              className="mb-4 bg-high/10 border border-high/30 rounded-xl px-5 py-3 text-sm text-high"
            >
              Cannot reach API at <code className="font-mono text-xs bg-high/10 px-1.5 py-0.5 rounded">http://127.0.0.1:8000</code> — start it with{' '}
              <code className="font-mono text-xs bg-high/10 px-1.5 py-0.5 rounded">uvicorn src.api.app:app --reload --port 8000</code>
            </motion.div>
          )}
        </AnimatePresence>

        {/* Dashboard tab */}
        <AnimatePresence mode="wait">
          {activeTab === 'dashboard' && (
            <motion.div
              key="dashboard"
              initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.25 }}
              className="space-y-4"
            >
              <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                <MetricCard level="LOW"    count={levelCounts.low}    />
                <MetricCard level="MEDIUM" count={levelCounts.medium} />
                <MetricCard level="HIGH"   count={levelCounts.high}   />
              </div>

              <RiskTimeline data={timeline} />

              <div className="grid grid-cols-1 lg:grid-cols-[1fr_320px] gap-4 items-start">
                <AlertFeed alerts={alerts} onAck={handleAckFromTable} />
                <div id="side-panel">
                  <SidePanel
                    thresholds={thresholds}
                    onUpdateThresholds={handleUpdateThresholds}
                    onAcknowledge={handleAcknowledge}
                    prefillAckId={prefillAckId}
                  />
                </div>
              </div>
            </motion.div>
          )}

          {/* Live Detection tab */}
          {activeTab === 'detection' && (
            <motion.div
              key="detection"
              initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -6 }}
              transition={{ duration: 0.25 }}
            >
              <LiveDetection />
            </motion.div>
          )}
        </AnimatePresence>

        {/* Footer */}
        <div className="flex items-center justify-between pt-6 mt-6 border-t border-[#21262d]">
          <div className="flex items-center gap-2 text-xs text-muted">
            <Activity size={11} />
            <span>Abnormal Crowd Behaviour Detection · SRM University AP · Capstone 2026</span>
          </div>
          {activeTab === 'dashboard' && (
            <span className="text-xs text-muted">Auto-refresh {REFRESH_INTERVAL / 1000}s</span>
          )}
        </div>
      </main>
    </div>
  )
}
