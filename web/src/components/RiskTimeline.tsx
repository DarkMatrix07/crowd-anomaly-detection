'use client'

import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts'
import type { TimelinePoint } from '@/lib/api'

interface Props {
  data: TimelinePoint[]
}

function formatTime(ts: number) {
  return new Date(ts * 1000).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  })
}

function scoreColor(score: number) {
  if (score >= 0.85) return '#ff7b72'
  if (score >= 0.6)  return '#f0883e'
  return '#3fb950'
}

const CustomTooltip = ({ active, payload }: any) => {
  if (!active || !payload?.length) return null
  const d = payload[0].payload as TimelinePoint
  return (
    <div className="bg-surface border border-border rounded-lg px-3 py-2 text-xs shadow-xl">
      <p className="text-muted mb-1">{formatTime(d.timestamp)}</p>
      <p className="font-semibold" style={{ color: scoreColor(d.score) }}>
        Score: {d.score.toFixed(3)}
      </p>
      <p className="text-muted">{d.camera_id}</p>
    </div>
  )
}

export default function RiskTimeline({ data }: Props) {
  const sorted = [...data].sort((a, b) => a.timestamp - b.timestamp)

  const chartData = sorted.map((p) => ({
    ...p,
    time: formatTime(p.timestamp),
    score: parseFloat(p.score.toFixed(4)),
  }))

  return (
    <div className="bg-surface border border-border rounded-xl p-5">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm font-semibold text-text">Risk Score Timeline</h2>
        <div className="flex items-center gap-4 text-xs text-muted">
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-low inline-block" /> Normal
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-medium inline-block" /> Medium
          </span>
          <span className="flex items-center gap-1.5">
            <span className="w-2 h-2 rounded-full bg-high inline-block" /> High
          </span>
        </div>
      </div>

      {chartData.length === 0 ? (
        <div className="h-48 flex items-center justify-center text-muted text-sm">
          No data yet — alerts will appear here in real time
        </div>
      ) : (
        <ResponsiveContainer width="100%" height={220}>
          <AreaChart data={chartData} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
            <defs>
              <linearGradient id="scoreGradient" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor="#58a6ff" stopOpacity={0.25} />
                <stop offset="95%" stopColor="#58a6ff" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#21262d" vertical={false} />
            <XAxis
              dataKey="time"
              tick={{ fill: '#8b949e', fontSize: 10 }}
              axisLine={false}
              tickLine={false}
              interval="preserveStartEnd"
            />
            <YAxis
              domain={[0, 1]}
              tick={{ fill: '#8b949e', fontSize: 10 }}
              axisLine={false}
              tickLine={false}
              tickCount={5}
            />
            <Tooltip content={<CustomTooltip />} />
            <ReferenceLine y={0.85} stroke="#ff7b72" strokeDasharray="4 4" strokeOpacity={0.6} />
            <ReferenceLine y={0.60} stroke="#f0883e" strokeDasharray="4 4" strokeOpacity={0.6} />
            <ReferenceLine y={0.30} stroke="#3fb950" strokeDasharray="4 4" strokeOpacity={0.6} />
            <Area
              type="monotone"
              dataKey="score"
              stroke="#58a6ff"
              strokeWidth={2}
              fill="url(#scoreGradient)"
              dot={false}
              activeDot={{ r: 4, fill: '#58a6ff', strokeWidth: 0 }}
            />
          </AreaChart>
        </ResponsiveContainer>
      )}
    </div>
  )
}
