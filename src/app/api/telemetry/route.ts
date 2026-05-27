import { NextResponse } from 'next/server'

interface TelemetryEvent {
  type: string
  timestamp: number
  session_id: string
  [key: string]: unknown
}

interface TelemetryPayload {
  events: TelemetryEvent[]
}

const HF_API = 'https://huggingface.co/api'

// CORS headers for cross-origin requests
const CORS_HEADERS = {
  'Access-Control-Allow-Origin': '*',
  'Access-Control-Allow-Methods': 'POST, OPTIONS',
  'Access-Control-Allow-Headers': 'Content-Type',
  'Access-Control-Max-Age': '86400',
}

// ── Rate Limiter (in-memory, per-isolate) ────────────────────────────
const RATE_LIMIT_WINDOW_MS = 60_000 // 1 minute
const RATE_LIMIT_MAX = 10           // max requests per window per session_id
const rateLimitMap = new Map<string, number[]>()

function isRateLimited(sessionId: string): boolean {
  const now = Date.now()
  let timestamps = rateLimitMap.get(sessionId)

  if (!timestamps) {
    timestamps = []
    rateLimitMap.set(sessionId, timestamps)
  }

  const cutoff = now - RATE_LIMIT_WINDOW_MS
  while (timestamps.length > 0 && timestamps[0] <= cutoff) {
    timestamps.shift()
  }

  if (timestamps.length >= RATE_LIMIT_MAX) {
    return true
  }

  timestamps.push(now)
  return false
}

function deriveSessionKey(event: TelemetryEvent): string {
  const raw = JSON.stringify(event)
  let h = 0
  for (let i = 0; i < raw.length; i++) {
    h = ((h << 5) - h + raw.charCodeAt(i)) | 0
  }
  return `__derived_${Math.abs(h).toString(36)}`
}

// ── Event Schema Validation ──────────────────────────────────────────
function validateEvent(event: unknown): event is TelemetryEvent {
  if (typeof event !== 'object' || event === null || Array.isArray(event)) {
    return false
  }
  const e = event as Record<string, unknown>

  if (typeof e.type !== 'string' || e.type.length === 0) return false
  if (typeof e.timestamp !== 'number' || !Number.isFinite(e.timestamp)) return false
  if (typeof e.session_id !== 'string') return false
  if (JSON.stringify(e).length > 65_536) return false

  return true
}

export async function OPTIONS() {
  return new NextResponse(null, { status: 204, headers: CORS_HEADERS })
}

export async function POST(request: Request) {
  const HF_TOKEN = process.env.HF_TOKEN
  const HF_DATASET_REPO = process.env.HF_DATASET_REPO
  const HF_DATASET_BRANCH = process.env.HF_DATASET_BRANCH || 'main'

  if (!HF_TOKEN || !HF_DATASET_REPO) {
    const missing = []
    if (!HF_TOKEN) missing.push('HF_TOKEN')
    if (!HF_DATASET_REPO) missing.push('HF_DATASET_REPO')
    console.error(`[Telemetry] Missing env vars: ${missing.join(', ')} — set these in Vercel Dashboard`)
    return jsonResponse({ error: `Telemetry not configured (missing: ${missing.join(', ')})` }, 503)
  }

  let payload: TelemetryPayload
  try {
    payload = await request.json() as TelemetryPayload
  } catch {
    return jsonResponse({ error: 'Invalid JSON' }, 400)
  }

  if (!payload.events || !Array.isArray(payload.events) || payload.events.length === 0) {
    return jsonResponse({ error: 'No events provided' }, 400)
  }

  const MAX_BATCH = 500
  const events = payload.events.slice(0, MAX_BATCH)

  const invalid = events.filter(e => !validateEvent(e))
  if (invalid.length > 0) {
    return jsonResponse(
      { error: `${invalid.length} event(s) failed schema validation` },
      400,
    )
  }

  const firstEvent = events[0]
  const sessionKey = firstEvent.session_id
    ? firstEvent.session_id
    : deriveSessionKey(firstEvent)

  if (isRateLimited(sessionKey)) {
    return jsonResponse({ error: 'Rate limit exceeded — try again later' }, 429)
  }

  const sanitized = events.map(stripPII)
  const jsonl = sanitized.map(e => JSON.stringify(e)).join('\n')

  const ts = new Date().toISOString().replace(/[:.]/g, '-')
  const hash = shortHash(jsonl)
  const filePath = `telemetry/batch_${ts}_${hash}.jsonl`

  const ok = await commitToHF(HF_TOKEN, HF_DATASET_REPO, HF_DATASET_BRANCH, filePath, jsonl)

  if (ok) {
    return jsonResponse({
      accepted: sanitized.length,
      file: filePath,
    }, 200)
  }

  return jsonResponse({ error: 'Failed to publish to HuggingFace — check function logs for details' }, 502)
}

// ── HuggingFace Hub Commit ───────────────────────────────────────────
async function commitToHF(
  token: string,
  repo: string,
  branch: string,
  filePath: string,
  content: string,
): Promise<boolean> {
  const url = `${HF_API}/datasets/${repo}/commit/${branch}`

  const contentBase64 = Buffer.from(content).toString('base64')
  const ndjson = [
    JSON.stringify({ key: 'header', value: { summary: `[telemetry] ${filePath}` } }),
    JSON.stringify({ key: 'file', value: { content: contentBase64, path: filePath, encoding: 'base64' } }),
  ].join('\n')

  try {
    const res = await fetch(url, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${token}`,
        'Content-Type': 'application/x-ndjson',
      },
      body: ndjson,
    })

    if (!res.ok) {
      const err = await res.text().catch(() => '')
      if (res.status === 401 || res.status === 403) {
        console.error(`[Telemetry] HF AUTH FAILED (${res.status}) — HF_TOKEN is invalid or lacks write access to "${repo}"`)
      } else if (res.status === 404) {
        console.error(`[Telemetry] HF REPO NOT FOUND (404) — "${repo}" does not exist on HuggingFace`)
      } else {
        console.error(`[Telemetry] HF commit failed (${res.status}): ${err.slice(0, 300)}`)
      }
    }

    return res.ok
  } catch (err) {
    console.error(`[Telemetry] Network error:`, err)
    return false
  }
}

// ── Helpers ──────────────────────────────────────────────────────────
const ALLOWED_FIELDS = new Set<string>([
  'type', 'timestamp', 'session_id', 'mode', 'model', 'duration_ms', 'response_length', 'success', 'error_type',
  'pipeline', 'autotune', 'detected_context', 'confidence', 'parseltongue', 'triggers_found', 'technique', 'intensity',
  'ultraplinian', 'tier', 'models_queried', 'models_succeeded', 'models_refused', 'early_stop', 'early_threshold',
  'winner_model', 'winner_score', 'winner_content_length', 'winner_duration_ms', 'winner_template', 'total_duration_ms',
  'judge_model', 'model_results', 'attempts', 'content_length', 'temperature', 'top_p', 'parseltongue_transform',
  'stm_modules', 'strategy', 'godmode', 'auto_retry', 'improve_mode', 'liquid_mode', 'autotune_context', 'autotune_confidence',
  'classification', 'persona', 'prompt_length', 'conversation_depth', 'memory_count', 'no_log', 'parseltongue_transformed',
  'has_image', 'winner_combo', 'combos_attempted', 'combos_succeeded', 'combos_failed', 'all_scores', 'encoding',
  'encoding_rounds', 'liquid_upgrades', 'combo', 'stream', 'fast_stream', 'liquid_upgraded', 'winner_source', 'race_result',
  'fallback_reason'
])

function stripPII(event: TelemetryEvent): TelemetryEvent {
  const clean: Record<string, unknown> = {}
  for (const key of Object.keys(event)) {
    if (ALLOWED_FIELDS.has(key)) {
      clean[key] = event[key]
    }
  }
  return clean as TelemetryEvent
}

function shortHash(str: string): string {
  let h = 0
  for (let i = 0; i < str.length; i++) {
    h = ((h << 5) - h + str.charCodeAt(i)) | 0
  }
  return Math.abs(h).toString(36).slice(0, 6)
}

function jsonResponse(data: unknown, status: number): NextResponse {
  return new NextResponse(JSON.stringify(data), {
    status,
    headers: {
      'Content-Type': 'application/json',
      ...CORS_HEADERS,
    },
  })
}
