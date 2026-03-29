'use client'

import { useCallback, useEffect, useRef, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Play, Square, RotateCcw, Cpu, ChevronDown,
  AlertTriangle, CheckCircle, TrendingUp, Activity,
} from 'lucide-react'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ReferenceLine, ResponsiveContainer,
} from 'recharts'
import clsx from 'clsx'
import { fetchClips, analyzeClip, frameUrl, type ClipInfo, type AnalysisResult } from '@/lib/api'

// ── Constants ─────────────────────────────────────────────────────────────────

const MODELS = [
  { id: 'resnet', label: 'ResNet18 + MLP', badge: '0.97 AUC', color: '#58a6ff' },
  { id: 'rf',     label: 'Random Forest',  badge: '0.83 AUC', color: '#3fb950' },
]

const THRESH_MEDIUM = 0.65
const THRESH_HIGH   = 0.85

function scoreLevel(s: number) {
  if (s >= THRESH_HIGH)   return 'HIGH'
  if (s >= THRESH_MEDIUM) return 'MEDIUM'
  return 'LOW'
}

const LEVEL_COLOR = { LOW: '#3fb950', MEDIUM: '#f0883e', HIGH: '#ff7b72' }
const LEVEL_BG    = {
  LOW:    'rgba(63,185,80,0.12)',
  MEDIUM: 'rgba(240,136,62,0.12)',
  HIGH:   'rgba(255,123,114,0.12)',
}


// ── Video overlay ─────────────────────────────────────────────────────────────

interface OverlayProps {
  score: number
  frameIdx: number
  totalFrames: number
  gtLabel: number | null
  model: string
}

function VideoOverlay({ score, frameIdx, totalFrames, gtLabel, model }: OverlayProps) {
  const level = scoreLevel(score)
  const color = LEVEL_COLOR[level]
  const pct   = (frameIdx / Math.max(totalFrames - 1, 1)) * 100

  return (
    <div className="absolute inset-0 pointer-events-none select-none">
      {/* Top banner */}
      <div className="absolute top-0 left-0 right-0 bg-gradient-to-b from-black/70 to-transparent px-4 py-3 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <span
            className="text-xs font-bold px-2.5 py-1 rounded-full border"
            style={{ color, borderColor: `${color}40`, background: `${color}15` }}
          >
            {level}
          </span>
          <span className="text-white font-mono text-sm font-semibold">
            {score.toFixed(3)}
          </span>
        </div>
        <div className="flex items-center gap-2">
          {gtLabel !== null && (
            <span
              className="text-xs px-2 py-0.5 rounded border font-medium"
              style={
                gtLabel === 1
                  ? { color: '#ff7b72', borderColor: '#ff7b7240', background: '#ff7b7215' }
                  : { color: '#8b949e', borderColor: '#30363d', background: 'transparent' }
              }
            >
              GT: {gtLabel === 1 ? 'ANOMALY' : 'normal'}
            </span>
          )}
          <span className="text-xs text-white/50 font-mono">
            {frameIdx + 1} / {totalFrames}
          </span>
        </div>
      </div>

      {/* Score bar */}
      <div className="absolute bottom-0 left-0 right-0">
        {/* Progress bar */}
        <div className="h-1 bg-white/10">
          <div
            className="h-full transition-all duration-100"
            style={{ width: `${pct}%`, background: color }}
          />
        </div>
        {/* Score fill */}
        <div className="h-1.5 bg-black/40">
          <div
            className="h-full transition-all duration-150"
            style={{ width: `${score * 100}%`, background: color }}
          />
        </div>
      </div>

      {/* HIGH alert pulsing border */}
      {level === 'HIGH' && (
        <div
          className="absolute inset-0 rounded-xl animate-pulse-red"
          style={{ boxShadow: 'inset 0 0 0 2px rgba(255,123,114,0.6)' }}
        />
      )}

      {/* Model badge */}
      <div className="absolute bottom-8 right-3 text-xs text-white/30 font-mono">
        {model === 'resnet' ? 'ResNet18+MLP' : 'RF'}
      </div>
    </div>
  )
}


// ── Summary ───────────────────────────────────────────────────────────────────

function AnalysisSummary({ result, onReset }: { result: AnalysisResult; onReset: () => void }) {
  const chartData = result.scores.map((s, i) => ({
    frame: i,
    score: s,
    gt:    result.gt_labels ? result.gt_labels[i] * 0.5 : undefined,
  }))

  const stats = [
    { label: 'Peak Score',     value: result.peak_score.toFixed(3),                        color: LEVEL_COLOR[scoreLevel(result.peak_score)] },
    { label: 'Mean Score',     value: result.mean_score.toFixed(3),                        color: LEVEL_COLOR[scoreLevel(result.mean_score)] },
    { label: 'Anomaly Frames', value: `${result.anomaly_frames} / ${result.frame_count}`,  color: '#f0883e' },
    { label: 'ROC-AUC',        value: result.roc_auc?.toFixed(4) ?? 'N/A',                 color: '#58a6ff' },
    { label: 'Accuracy',       value: result.accuracy ? `${(result.accuracy * 100).toFixed(1)}%` : 'N/A', color: '#3fb950' },
    { label: 'GT Anomaly %',   value: result.gt_anomaly_pct != null ? `${result.gt_anomaly_pct}%` : 'N/A', color: '#8b949e' },
  ]

  return (
    <motion.div
      initial={{ opacity: 0, y: 20 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.5, ease: [0.16, 1, 0.3, 1] }}
      className="mt-6 space-y-4"
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <CheckCircle size={16} className="text-low" />
          <h3 className="text-sm font-semibold text-text">Analysis Complete — {result.clip_id}</h3>
        </div>
        <button
          onClick={onReset}
          className="flex items-center gap-1.5 text-xs text-muted hover:text-text border border-border hover:border-border/80 rounded-lg px-3 py-1.5 transition-all"
        >
          <RotateCcw size={11} /> New Analysis
        </button>
      </div>

      {/* Stat grid */}
      <div className="grid grid-cols-3 lg:grid-cols-6 gap-3">
        {stats.map(({ label, value, color }) => (
          <div key={label} className="bg-surface border border-border rounded-xl p-4">
            <p className="text-xs text-muted mb-1 truncate">{label}</p>
            <p className="text-lg font-bold font-mono" style={{ color }}>{value}</p>
          </div>
        ))}
      </div>

      {/* Score timeline chart */}
      <div className="bg-surface border border-border rounded-xl p-5">
        <div className="flex items-center gap-2 mb-4">
          <TrendingUp size={14} className="text-muted" />
          <h4 className="text-sm font-semibold text-text">Frame-by-Frame Score</h4>
          {result.gt_labels && (
            <span className="text-xs text-muted ml-auto flex items-center gap-1.5">
              <span className="w-3 h-0.5 bg-high/60 inline-block rounded" /> Ground truth anomaly regions
            </span>
          )}
        </div>
        <ResponsiveContainer width="100%" height={180}>
          <AreaChart data={chartData} margin={{ top: 4, right: 4, left: -20, bottom: 0 }}>
            <defs>
              <linearGradient id="sg" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor="#58a6ff" stopOpacity={0.3} />
                <stop offset="95%" stopColor="#58a6ff" stopOpacity={0.02} />
              </linearGradient>
              <linearGradient id="gtg" x1="0" y1="0" x2="0" y2="1">
                <stop offset="5%"  stopColor="#ff7b72" stopOpacity={0.25} />
                <stop offset="95%" stopColor="#ff7b72" stopOpacity={0.02} />
              </linearGradient>
            </defs>
            <CartesianGrid strokeDasharray="3 3" stroke="#21262d" vertical={false} />
            <XAxis dataKey="frame" tick={{ fill: '#8b949e', fontSize: 10 }} axisLine={false} tickLine={false} />
            <YAxis domain={[0, 1]} tick={{ fill: '#8b949e', fontSize: 10 }} axisLine={false} tickLine={false} tickCount={5} />
            <Tooltip
              contentStyle={{ background: '#161b22', border: '1px solid #30363d', borderRadius: 8, fontSize: 12 }}
              labelStyle={{ color: '#8b949e' }}
            />
            <ReferenceLine y={THRESH_HIGH}   stroke="#ff7b72" strokeDasharray="4 4" strokeOpacity={0.5} />
            <ReferenceLine y={THRESH_MEDIUM} stroke="#f0883e" strokeDasharray="4 4" strokeOpacity={0.5} />
            {result.gt_labels && (
              <Area type="step" dataKey="gt" stroke="#ff7b72" strokeWidth={0} fill="url(#gtg)" dot={false} />
            )}
            <Area type="monotone" dataKey="score" stroke="#58a6ff" strokeWidth={2} fill="url(#sg)" dot={false}
              activeDot={{ r: 3, fill: '#58a6ff', strokeWidth: 0 }} />
          </AreaChart>
        </ResponsiveContainer>
      </div>

      {/* Model comparison badge */}
      <div className="bg-surface border border-border rounded-xl px-5 py-4 flex items-center justify-between">
        <div className="text-xs text-muted">
          Model used: <span className="text-text font-medium">
            {result.model === 'resnet' ? 'ResNet18 + MLP' : 'Random Forest (W=30)'}
          </span>
        </div>
        {result.roc_auc && (
          <div className="text-xs text-muted">
            Clip ROC-AUC:
            <span className="text-accent font-mono font-semibold ml-1">{result.roc_auc.toFixed(4)}</span>
            <span className="text-muted ml-2">vs RF baseline 0.8313</span>
          </div>
        )}
      </div>
    </motion.div>
  )
}


// ── Main component ─────────────────────────────────────────────────────────────

export default function LiveDetection() {
  const [clips, setClips]             = useState<ClipInfo[]>([])
  const [clipsError, setClipsError]   = useState<string | null>(null)
  const [selectedClip, setSelectedClip] = useState<string>('')
  const [selectedModel, setSelectedModel] = useState<string>('resnet')

  const [analyzing, setAnalyzing]     = useState(false)
  const [analyzeError, setAnalyzeError] = useState<string | null>(null)
  const [result, setResult]           = useState<AnalysisResult | null>(null)

  const [playing, setPlaying]         = useState(false)
  const [currentFrame, setCurrentFrame] = useState(0)
  const [fps, setFps]                 = useState(12)
  const [done, setDone]               = useState(false)

  const intervalRef = useRef<ReturnType<typeof setInterval> | null>(null)
  const imgRef      = useRef<HTMLImageElement>(null)

  // Load clips on mount
  useEffect(() => {
    fetchClips()
      .then((c) => { setClips(c); setSelectedClip(c[0]?.id ?? '') })
      .catch((e) => setClipsError(e.message))
  }, [])

  // Playback loop
  useEffect(() => {
    if (!playing || !result) return
    intervalRef.current = setInterval(() => {
      setCurrentFrame((prev) => {
        const next = prev + 1
        if (next >= result.frame_count) {
          setPlaying(false)
          setDone(true)
          return prev
        }
        return next
      })
    }, 1000 / fps)
    return () => { if (intervalRef.current) clearInterval(intervalRef.current) }
  }, [playing, fps, result])

  const handleAnalyze = useCallback(async () => {
    if (!selectedClip) return
    setAnalyzing(true)
    setAnalyzeError(null)
    setResult(null)
    setCurrentFrame(0)
    setDone(false)
    setPlaying(false)
    try {
      const r = await analyzeClip(selectedClip, selectedModel)
      setResult(r)
    } catch (e: any) {
      setAnalyzeError(e.message)
    } finally {
      setAnalyzing(false)
    }
  }, [selectedClip, selectedModel])

  function handlePlay() {
    if (done) { setCurrentFrame(0); setDone(false) }
    setPlaying(true)
  }

  function handleStop() { setPlaying(false) }

  function handleReset() {
    setResult(null)
    setCurrentFrame(0)
    setPlaying(false)
    setDone(false)
    setAnalyzeError(null)
  }

  const currentScore = result?.scores[currentFrame] ?? 0
  const currentLevel = scoreLevel(currentScore) as keyof typeof LEVEL_COLOR
  const currentGt    = result?.gt_labels ? result.gt_labels[currentFrame] : null

  const clipInfo = clips.find((c) => c.id === selectedClip)

  return (
    <div className="space-y-5">

      {/* ── Config bar ────────────────────────────────────────────── */}
      <div className="bg-surface border border-border rounded-xl p-5">
        <div className="grid grid-cols-1 sm:grid-cols-[1fr_200px_140px_auto] gap-3 items-end">

          {/* Clip selector */}
          <div>
            <label className="block text-xs text-muted mb-1.5">Video Clip</label>
            {clipsError ? (
              <p className="text-xs text-high">{clipsError}</p>
            ) : (
              <div className="relative">
                <select
                  value={selectedClip}
                  onChange={(e) => { setSelectedClip(e.target.value); handleReset() }}
                  disabled={analyzing}
                  className="w-full appearance-none bg-bg border border-border rounded-lg px-3 py-2.5 text-sm text-text focus:outline-none focus:border-accent/50 transition-colors cursor-pointer pr-8"
                >
                  {clips.map((c) => (
                    <option key={c.id} value={c.id}>
                      {c.id} — {c.description}
                    </option>
                  ))}
                </select>
                <ChevronDown size={13} className="absolute right-3 top-1/2 -translate-y-1/2 text-muted pointer-events-none" />
              </div>
            )}
          </div>

          {/* Model selector */}
          <div>
            <label className="block text-xs text-muted mb-1.5">Model</label>
            <div className="flex gap-2">
              {MODELS.map((m) => (
                <button
                  key={m.id}
                  onClick={() => { setSelectedModel(m.id); handleReset() }}
                  disabled={analyzing}
                  className={clsx(
                    'flex-1 py-2.5 rounded-lg text-xs font-semibold border transition-all',
                    selectedModel === m.id
                      ? 'text-white border-transparent'
                      : 'border-border text-muted hover:text-text',
                  )}
                  style={selectedModel === m.id ? { background: `${m.color}20`, borderColor: `${m.color}40`, color: m.color } : {}}
                >
                  {m.label}
                  <span className="block text-[10px] opacity-60 font-mono">{m.badge}</span>
                </button>
              ))}
            </div>
          </div>

          {/* FPS */}
          <div>
            <label className="block text-xs text-muted mb-1.5">Speed — {fps} fps</label>
            <input
              type="range" min={1} max={30} value={fps}
              onChange={(e) => setFps(Number(e.target.value))}
              className="w-full mt-1.5"
            />
          </div>

          {/* Analyze button */}
          <button
            onClick={handleAnalyze}
            disabled={analyzing || !selectedClip}
            className={clsx(
              'px-5 py-2.5 rounded-lg text-sm font-semibold border transition-all whitespace-nowrap',
              analyzing
                ? 'border-border text-muted cursor-wait'
                : 'bg-accent/10 text-accent border-accent/30 hover:bg-accent/20',
            )}
          >
            {analyzing ? (
              <span className="flex items-center gap-2">
                <Cpu size={14} className="animate-pulse" /> Analyzing…
              </span>
            ) : result ? 'Re-analyze' : 'Analyze Clip'}
          </button>
        </div>

        {/* Clip info row */}
        {clipInfo && (
          <div className="mt-3 flex items-center gap-4 text-xs text-muted border-t border-[#21262d] pt-3">
            <span>{clipInfo.frame_count} frames</span>
            {clipInfo.has_ground_truth && (
              <span className="text-low flex items-center gap-1">
                <CheckCircle size={10} /> Ground truth available
              </span>
            )}
            {analyzeError && (
              <span className="text-high flex items-center gap-1 ml-auto">
                <AlertTriangle size={11} /> {analyzeError}
              </span>
            )}
          </div>
        )}
      </div>

      {/* ── Analyzing spinner ─────────────────────────────────────── */}
      <AnimatePresence>
        {analyzing && (
          <motion.div
            initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            className="bg-surface border border-border rounded-xl p-8 flex flex-col items-center gap-3"
          >
            <div className="w-10 h-10 rounded-full border-2 border-accent/20 border-t-accent animate-spin" />
            <p className="text-sm text-muted">Running inference on all frames…</p>
            <p className="text-xs text-muted/60">
              {selectedModel === 'resnet'
                ? 'ResNet18 feature extraction in progress — may take 30–60 seconds'
                : 'Random Forest inference — should be quick'}
            </p>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Player ───────────────────────────────────────────────── */}
      <AnimatePresence>
        {result && !analyzing && (
          <motion.div
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ duration: 0.4, ease: [0.16, 1, 0.3, 1] }}
            className="grid grid-cols-1 lg:grid-cols-[1fr_340px] gap-4 items-start"
          >
            {/* Video panel */}
            <div className="bg-surface border border-border rounded-xl overflow-hidden">
              <div className="relative bg-black aspect-video">
                {/* Frame image */}
                <img
                  ref={imgRef}
                  src={frameUrl(result.clip_id, currentFrame)}
                  alt={`Frame ${currentFrame}`}
                  className="w-full h-full object-contain"
                  key={currentFrame}
                />
                {/* Overlay */}
                <VideoOverlay
                  score={currentScore}
                  frameIdx={currentFrame}
                  totalFrames={result.frame_count}
                  gtLabel={currentGt}
                  model={selectedModel}
                />
              </div>

              {/* Playback controls */}
              <div className="flex items-center gap-3 px-4 py-3 border-t border-border">
                {!playing ? (
                  <button
                    onClick={handlePlay}
                    className="flex items-center gap-2 px-4 py-2 bg-accent/10 text-accent border border-accent/30 rounded-lg text-sm font-semibold hover:bg-accent/20 transition-all"
                  >
                    <Play size={13} fill="currentColor" />
                    {done ? 'Replay' : 'Play'}
                  </button>
                ) : (
                  <button
                    onClick={handleStop}
                    className="flex items-center gap-2 px-4 py-2 bg-high/10 text-high border border-high/30 rounded-lg text-sm font-semibold hover:bg-high/20 transition-all"
                  >
                    <Square size={13} fill="currentColor" /> Stop
                  </button>
                )}

                {/* Mini score bars */}
                <div className="flex-1 flex items-center gap-2">
                  <div className="flex-1 h-1.5 bg-[#21262d] rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full transition-all duration-150"
                      style={{
                        width: `${currentScore * 100}%`,
                        background: LEVEL_COLOR[currentLevel],
                      }}
                    />
                  </div>
                  <span
                    className="text-xs font-mono font-semibold w-12 text-right"
                    style={{ color: LEVEL_COLOR[currentLevel] }}
                  >
                    {currentScore.toFixed(3)}
                  </span>
                </div>

                <span
                  className="text-xs px-2.5 py-1 rounded-full font-semibold border"
                  style={{
                    color: LEVEL_COLOR[currentLevel],
                    borderColor: `${LEVEL_COLOR[currentLevel]}40`,
                    background: LEVEL_BG[currentLevel],
                  }}
                >
                  {currentLevel}
                </span>
              </div>
            </div>

            {/* Live stats sidebar */}
            <div className="space-y-3">
              {/* Live score card */}
              <div
                className="bg-surface border rounded-xl p-5 transition-colors duration-300"
                style={{ borderColor: `${LEVEL_COLOR[currentLevel]}30` }}
              >
                <p className="text-xs text-muted mb-1 uppercase tracking-widest">Current Score</p>
                <motion.p
                  key={currentFrame}
                  className="text-5xl font-bold font-mono tracking-tight"
                  style={{ color: LEVEL_COLOR[currentLevel] }}
                >
                  {currentScore.toFixed(3)}
                </motion.p>
                <p className="text-xs text-muted mt-2">
                  Frame {currentFrame + 1} of {result.frame_count}
                </p>
              </div>

              {/* Mini metrics */}
              <div className="grid grid-cols-2 gap-3">
                {[
                  { label: 'Peak',      value: result.peak_score.toFixed(3), color: LEVEL_COLOR[scoreLevel(result.peak_score)] },
                  { label: 'Anomalous', value: `${result.anomaly_frames}`,   color: '#f0883e' },
                  { label: 'ROC-AUC',   value: result.roc_auc?.toFixed(3) ?? '—', color: '#58a6ff' },
                  { label: 'Accuracy',  value: result.accuracy ? `${(result.accuracy * 100).toFixed(0)}%` : '—', color: '#3fb950' },
                ].map(({ label, value, color }) => (
                  <div key={label} className="bg-surface border border-border rounded-xl p-3">
                    <p className="text-xs text-muted mb-0.5">{label}</p>
                    <p className="text-xl font-bold font-mono" style={{ color }}>{value}</p>
                  </div>
                ))}
              </div>

              {/* Mini score strip */}
              <div className="bg-surface border border-border rounded-xl p-4">
                <div className="flex items-center gap-2 mb-2">
                  <Activity size={12} className="text-muted" />
                  <p className="text-xs text-muted">Score history</p>
                </div>
                <div className="flex gap-0.5 h-10 items-end">
                  {result.scores
                    .filter((_, i) => i % Math.max(1, Math.floor(result.scores.length / 60)) === 0)
                    .map((s, i) => {
                      const lv = scoreLevel(s) as keyof typeof LEVEL_COLOR
                      const isCurrent = i === Math.floor(currentFrame / Math.max(1, Math.floor(result.scores.length / 60)))
                      return (
                        <div
                          key={i}
                          className="flex-1 rounded-sm transition-all"
                          style={{
                            height: `${Math.max(4, s * 40)}px`,
                            background: LEVEL_COLOR[lv],
                            opacity: isCurrent ? 1 : 0.4,
                          }}
                        />
                      )
                    })}
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* ── Summary (shown when playback completes) ───────────────── */}
      <AnimatePresence>
        {done && result && (
          <AnalysisSummary result={result} onReset={handleReset} />
        )}
      </AnimatePresence>
    </div>
  )
}
