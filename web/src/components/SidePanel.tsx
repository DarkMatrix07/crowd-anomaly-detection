'use client'

import { useState } from 'react'
import { Settings, MessageSquare, CheckCircle } from 'lucide-react'
import clsx from 'clsx'
import type { ThresholdProfile } from '@/lib/api'

interface Props {
  thresholds: ThresholdProfile
  onUpdateThresholds: (p: ThresholdProfile) => Promise<void>
  onAcknowledge: (id: number, operator: string, note: string) => Promise<void>
  prefillAckId?: number | null
}

const PROFILES = ['default', 'strict', 'relaxed', 'custom'] as const

const THRESHOLD_PRESETS: Record<string, Omit<ThresholdProfile, 'profile_name'>> = {
  default: { low: 0.30, medium: 0.60, high: 0.85 },
  strict:  { low: 0.20, medium: 0.45, high: 0.70 },
  relaxed: { low: 0.40, medium: 0.65, high: 0.90 },
}

const SLIDER_CONFIG = {
  low:    { label: 'Low',    color: '#3fb950' },
  medium: { label: 'Medium', color: '#f0883e' },
  high:   { label: 'High',   color: '#ff7b72' },
} as const

export default function SidePanel({
  thresholds,
  onUpdateThresholds,
  onAcknowledge,
  prefillAckId,
}: Props) {
  const [local, setLocal] = useState<ThresholdProfile>(thresholds)
  const [saving, setSaving] = useState(false)
  const [saved, setSaved] = useState(false)

  const [ackId, setAckId] = useState(String(prefillAckId ?? ''))
  const [operator, setOperator] = useState('operator-1')
  const [note, setNote] = useState('')
  const [acking, setAcking] = useState(false)
  const [ackDone, setAckDone] = useState(false)

  function applyPreset(name: string) {
    const preset = THRESHOLD_PRESETS[name]
    if (preset) setLocal({ ...preset, profile_name: name })
    else setLocal((prev) => ({ ...prev, profile_name: name }))
  }

  async function handleSaveThresholds() {
    setSaving(true)
    try {
      await onUpdateThresholds(local)
      setSaved(true)
      setTimeout(() => setSaved(false), 2000)
    } finally {
      setSaving(false)
    }
  }

  async function handleAck(e: React.FormEvent) {
    e.preventDefault()
    setAcking(true)
    try {
      await onAcknowledge(Number(ackId), operator, note)
      setAckDone(true)
      setAckId('')
      setNote('')
      setTimeout(() => setAckDone(false), 2500)
    } finally {
      setAcking(false)
    }
  }

  return (
    <div className="flex flex-col gap-4">
      {/* Thresholds */}
      <div className="bg-surface border border-border rounded-xl p-5">
        <div className="flex items-center gap-2 mb-4">
          <Settings size={14} className="text-muted" />
          <h2 className="text-sm font-semibold text-text">Alert Thresholds</h2>
        </div>

        {/* Profile selector */}
        <div className="flex gap-1.5 mb-5 flex-wrap">
          {PROFILES.map((p) => (
            <button
              key={p}
              onClick={() => applyPreset(p)}
              className={clsx(
                'px-3 py-1 rounded-md text-xs font-medium border transition-all capitalize',
                local.profile_name === p
                  ? 'bg-accent/10 border-accent/40 text-accent'
                  : 'border-border text-muted hover:text-text hover:border-border/80',
              )}
            >
              {p}
            </button>
          ))}
        </div>

        {/* Sliders */}
        <div className="space-y-4">
          {(Object.keys(SLIDER_CONFIG) as Array<keyof typeof SLIDER_CONFIG>).map((key) => {
            const cfg = SLIDER_CONFIG[key]
            return (
              <div key={key}>
                <div className="flex justify-between items-center mb-1.5">
                  <label className="text-xs font-medium" style={{ color: cfg.color }}>
                    {cfg.label}
                  </label>
                  <span className="text-xs font-mono text-muted">
                    {local[key].toFixed(2)}
                  </span>
                </div>
                <input
                  type="range"
                  min={0}
                  max={1}
                  step={0.01}
                  value={local[key]}
                  onChange={(e) =>
                    setLocal((prev) => ({ ...prev, [key]: parseFloat(e.target.value) }))
                  }
                  className="w-full"
                  style={{ accentColor: cfg.color }}
                />
              </div>
            )
          })}
        </div>

        <button
          onClick={handleSaveThresholds}
          disabled={saving}
          className={clsx(
            'mt-5 w-full py-2 rounded-lg text-sm font-semibold transition-all',
            saved
              ? 'bg-low/15 text-low border border-low/30'
              : 'bg-accent/10 text-accent border border-accent/30 hover:bg-accent/20',
          )}
        >
          {saved ? 'Saved' : saving ? 'Saving…' : 'Apply Thresholds'}
        </button>
      </div>

      {/* Acknowledge */}
      <div className="bg-surface border border-border rounded-xl p-5">
        <div className="flex items-center gap-2 mb-4">
          <MessageSquare size={14} className="text-muted" />
          <h2 className="text-sm font-semibold text-text">Acknowledge Alert</h2>
        </div>

        <form onSubmit={handleAck} className="space-y-3">
          <div>
            <label className="block text-xs text-muted mb-1">Alert ID</label>
            <input
              type="number"
              min={1}
              value={ackId}
              onChange={(e) => setAckId(e.target.value)}
              placeholder="e.g. 42"
              required
              className="w-full bg-bg border border-border rounded-lg px-3 py-2 text-sm text-text placeholder-muted/50 focus:outline-none focus:border-accent/50 transition-colors"
            />
          </div>
          <div>
            <label className="block text-xs text-muted mb-1">Operator</label>
            <input
              type="text"
              value={operator}
              onChange={(e) => setOperator(e.target.value)}
              required
              className="w-full bg-bg border border-border rounded-lg px-3 py-2 text-sm text-text placeholder-muted/50 focus:outline-none focus:border-accent/50 transition-colors"
            />
          </div>
          <div>
            <label className="block text-xs text-muted mb-1">Note</label>
            <textarea
              value={note}
              onChange={(e) => setNote(e.target.value)}
              rows={2}
              placeholder="Checked and escalated."
              className="w-full bg-bg border border-border rounded-lg px-3 py-2 text-sm text-text placeholder-muted/50 focus:outline-none focus:border-accent/50 transition-colors resize-none"
            />
          </div>
          <button
            type="submit"
            disabled={acking || !ackId}
            className={clsx(
              'w-full py-2 rounded-lg text-sm font-semibold transition-all',
              ackDone
                ? 'bg-low/15 text-low border border-low/30'
                : 'bg-[#21262d] text-text border border-border hover:border-accent/40 hover:text-accent disabled:opacity-40',
            )}
          >
            {ackDone ? (
              <span className="flex items-center justify-center gap-2">
                <CheckCircle size={14} /> Acknowledged
              </span>
            ) : acking ? 'Submitting…' : 'Submit'}
          </button>
        </form>
      </div>
    </div>
  )
}
