'use client'

import { motion } from 'framer-motion'
import clsx from 'clsx'

const LEVEL_CONFIG = {
  LOW: {
    label: 'Low Risk',
    color: '#3fb950',
    bg: 'rgba(63, 185, 80, 0.08)',
    border: 'rgba(63, 185, 80, 0.25)',
    dot: 'bg-low',
  },
  MEDIUM: {
    label: 'Medium Risk',
    color: '#f0883e',
    bg: 'rgba(240, 136, 62, 0.08)',
    border: 'rgba(240, 136, 62, 0.25)',
    dot: 'bg-medium',
  },
  HIGH: {
    label: 'High Risk',
    color: '#ff7b72',
    bg: 'rgba(255, 123, 114, 0.08)',
    border: 'rgba(255, 123, 114, 0.25)',
    dot: 'bg-high',
  },
}

interface MetricCardProps {
  level: 'LOW' | 'MEDIUM' | 'HIGH'
  count: number
}

export default function MetricCard({ level, count }: MetricCardProps) {
  const cfg = LEVEL_CONFIG[level]
  const isPulsing = level === 'HIGH' && count > 0

  return (
    <motion.div
      initial={{ opacity: 0, y: 12 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
      className={clsx(
        'relative rounded-xl p-5 border overflow-hidden',
        isPulsing && 'animate-pulse-red',
      )}
      style={{
        background: cfg.bg,
        borderColor: cfg.border,
      }}
    >
      {/* Glow top edge */}
      <div
        className="absolute top-0 left-0 right-0 h-px"
        style={{ background: `linear-gradient(90deg, transparent, ${cfg.color}50, transparent)` }}
      />

      <div className="flex items-start justify-between">
        <div>
          <p className="text-xs font-semibold uppercase tracking-widest text-muted mb-3">
            {cfg.label}
          </p>
          <motion.p
            key={count}
            initial={{ scale: 0.8, opacity: 0 }}
            animate={{ scale: 1, opacity: 1 }}
            transition={{ type: 'spring', stiffness: 300, damping: 25 }}
            className="text-4xl font-bold tracking-tight"
            style={{ color: cfg.color }}
          >
            {count}
          </motion.p>
          <p className="text-xs text-muted mt-1">alerts</p>
        </div>

        {/* Status dot */}
        <div className="flex items-center gap-1.5 mt-1">
          <span
            className={clsx('w-2 h-2 rounded-full', cfg.dot, isPulsing && 'animate-ping')}
          />
          <span className="text-xs text-muted">{level}</span>
        </div>
      </div>
    </motion.div>
  )
}
