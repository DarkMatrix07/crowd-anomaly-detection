const API = '/api'

export interface Alert {
  id: number
  timestamp: number
  camera_id: string
  risk_level: 'LOW' | 'MEDIUM' | 'HIGH'
  score: number
  evidence_window: [number, number]
  acknowledged_by: string | null
  ack_note: string | null
  acknowledged_at: number | null
}

export interface TimelinePoint {
  timestamp: number
  score: number
  risk_level: string
  camera_id: string
}

export interface LevelCounts {
  low: number
  medium: number
  high: number
}

export interface DashboardSummary {
  alerts: Alert[]
  timeline: TimelinePoint[]
  level_counts: LevelCounts
}

export interface ThresholdProfile {
  profile_name: string
  low: number
  medium: number
  high: number
}

export interface ClipInfo {
  id: string
  description: string
  frame_count: number
  has_ground_truth: boolean
}

export interface AnalysisResult {
  clip_id: string
  model: string
  frame_count: number
  scores: number[]
  predictions: number[]
  gt_labels: number[] | null
  peak_score: number
  mean_score: number
  anomaly_frames: number
  accuracy?: number
  gt_anomaly_pct?: number
  roc_auc?: number
}

// ── Dashboard ─────────────────────────────────────────────────────────────────

export async function checkHealth(): Promise<boolean> {
  try {
    const res = await fetch(`${API}/health`, {
      signal: AbortSignal.timeout(3000),
      cache: 'no-store',
    })
    const data = await res.json()
    return data.status === 'ok'
  } catch {
    return false
  }
}

export async function fetchSummary(
  limit = 100,
  cameraId?: string,
): Promise<DashboardSummary> {
  const params = new URLSearchParams({ limit: String(limit) })
  if (cameraId && cameraId !== 'ALL') params.append('camera_id', cameraId)
  const res = await fetch(`${API}/dashboard/summary?${params}`, { cache: 'no-store' })
  if (!res.ok) throw new Error(`API ${res.status}`)
  return res.json()
}

export async function fetchThresholds(): Promise<ThresholdProfile> {
  const res = await fetch(`${API}/config/thresholds`, { cache: 'no-store' })
  if (!res.ok) throw new Error(`API ${res.status}`)
  return res.json()
}

export async function updateThresholds(profile: ThresholdProfile): Promise<ThresholdProfile> {
  const res = await fetch(`${API}/config/thresholds`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(profile),
  })
  if (!res.ok) throw new Error(`API ${res.status}`)
  return res.json()
}

export async function acknowledgeAlert(
  alertId: number,
  operatorName: string,
  note: string,
): Promise<Alert> {
  const res = await fetch(`${API}/alerts/${alertId}/ack`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ operator_name: operatorName, note }),
  })
  if (!res.ok) throw new Error(`API ${res.status}`)
  return res.json()
}

// ── Demo / Live Detection ─────────────────────────────────────────────────────

export async function fetchClips(): Promise<ClipInfo[]> {
  const res = await fetch(`${API}/demo/clips`, { cache: 'no-store' })
  if (!res.ok) throw new Error(`API ${res.status}`)
  const data = await res.json()
  return data.clips
}

export function frameUrl(clipId: string, frameIdx: number): string {
  return `${API}/demo/clips/${clipId}/frame/${frameIdx}`
}

export async function analyzeClip(
  clipId: string,
  model: string,
): Promise<AnalysisResult> {
  const res = await fetch(`${API}/demo/clips/${clipId}/analyze`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ model, window_size: 30, stride: 10 }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    throw new Error(err.detail ?? `API ${res.status}`)
  }
  return res.json()
}
