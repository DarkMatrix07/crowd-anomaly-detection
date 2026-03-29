'use client'

import { AnimatePresence, motion } from 'framer-motion'
import { CheckCircle, Clock } from 'lucide-react'
import clsx from 'clsx'
import type { Alert } from '@/lib/api'

const LEVEL_STYLES = {
  LOW:    { badge: 'bg-low/10 text-low border-low/30',    bar: 'bg-low'    },
  MEDIUM: { badge: 'bg-medium/10 text-medium border-medium/30', bar: 'bg-medium' },
  HIGH:   { badge: 'bg-high/10 text-high border-high/30',  bar: 'bg-high'   },
}

function formatTime(ts: number) {
  return new Date(ts * 1000).toLocaleString([], {
    month: 'short', day: '2-digit',
    hour: '2-digit', minute: '2-digit', second: '2-digit',
  })
}

interface Props {
  alerts: Alert[]
  onAck: (id: number) => void
}

export default function AlertFeed({ alerts, onAck }: Props) {
  return (
    <div className="bg-surface border border-border rounded-xl overflow-hidden">
      <div className="flex items-center justify-between px-5 py-4 border-b border-border">
        <h2 className="text-sm font-semibold text-text">Alert History</h2>
        <span className="text-xs text-muted">{alerts.length} records</span>
      </div>

      <div className="overflow-auto max-h-[420px]">
        {alerts.length === 0 ? (
          <div className="py-16 text-center text-muted text-sm">No alerts recorded yet.</div>
        ) : (
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-border">
                {['ID', 'Camera', 'Level', 'Score', 'Time', 'Status'].map((h) => (
                  <th
                    key={h}
                    className="px-4 py-3 text-left text-xs font-semibold text-muted uppercase tracking-wider"
                  >
                    {h}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              <AnimatePresence initial={false}>
                {alerts.map((alert) => {
                  const styles = LEVEL_STYLES[alert.risk_level] ?? LEVEL_STYLES.LOW
                  return (
                    <motion.tr
                      key={alert.id}
                      initial={{ opacity: 0, x: 10 }}
                      animate={{ opacity: 1, x: 0 }}
                      exit={{ opacity: 0 }}
                      transition={{ duration: 0.2 }}
                      className="border-b border-[#21262d] hover:bg-white/[0.02] transition-colors"
                    >
                      {/* Left accent bar */}
                      <td className="px-4 py-3 relative">
                        <div
                          className={clsx('absolute left-0 top-2 bottom-2 w-0.5 rounded-full', styles.bar)}
                        />
                        <span className="font-mono text-muted text-xs">#{alert.id}</span>
                      </td>
                      <td className="px-4 py-3">
                        <span className="font-mono text-xs text-accent">{alert.camera_id}</span>
                      </td>
                      <td className="px-4 py-3">
                        <span
                          className={clsx(
                            'inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-semibold border',
                            styles.badge,
                          )}
                        >
                          {alert.risk_level}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        <div className="flex items-center gap-2">
                          <div className="w-16 h-1.5 bg-[#21262d] rounded-full overflow-hidden">
                            <div
                              className={clsx('h-full rounded-full', styles.bar)}
                              style={{ width: `${alert.score * 100}%` }}
                            />
                          </div>
                          <span className="font-mono text-xs text-muted">
                            {alert.score.toFixed(3)}
                          </span>
                        </div>
                      </td>
                      <td className="px-4 py-3">
                        <span className="flex items-center gap-1.5 text-xs text-muted">
                          <Clock size={11} />
                          {formatTime(alert.timestamp)}
                        </span>
                      </td>
                      <td className="px-4 py-3">
                        {alert.acknowledged_by ? (
                          <span className="flex items-center gap-1.5 text-xs text-low">
                            <CheckCircle size={13} />
                            {alert.acknowledged_by}
                          </span>
                        ) : (
                          <button
                            onClick={() => onAck(alert.id)}
                            className="text-xs text-muted hover:text-accent border border-border hover:border-accent/40 rounded-md px-2.5 py-1 transition-all"
                          >
                            Acknowledge
                          </button>
                        )}
                      </td>
                    </motion.tr>
                  )
                })}
              </AnimatePresence>
            </tbody>
          </table>
        )}
      </div>
    </div>
  )
}
