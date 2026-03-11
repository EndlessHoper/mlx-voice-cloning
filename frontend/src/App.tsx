import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { AnimatePresence, motion } from 'framer-motion'

const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

interface ScriptOption { text: string }

interface GenerationSettings {
  max_new_tokens: number | null
  repetition_penalty: number | null
  temperature: number | null
  top_p: number | null
}

interface ActivityLog {
  id: string
  time: string
  message: string
  tone: 'info' | 'success' | 'warning' | 'error'
}

interface GenerationResult {
  id: string
  url: string
  text: string
  elapsed: number
  createdAt: string
}

const defaultSettings: GenerationSettings = {
  max_new_tokens: null,
  repetition_penalty: null,
  temperature: null,
  top_p: null,
}

function api(path: string): string {
  return API_BASE ? `${API_BASE}${path}` : path
}

function asset(path: string | undefined): string {
  if (!path) return ''
  if (/^https?:\/\//.test(path)) return path
  return API_BASE ? `${API_BASE}${path}` : path
}

function isObj(v: unknown): v is Record<string, unknown> {
  return Boolean(v) && typeof v === 'object' && !Array.isArray(v)
}

function str(r: Record<string, unknown>, k: string): string | undefined {
  const v = r[k]; return typeof v === 'string' ? v : undefined
}

function errMsg(p: unknown, fallback: string): string {
  if (typeof p === 'string' && p.trim()) return p
  if (!isObj(p)) return fallback
  return str(p, 'detail') || str(p, 'message') || fallback
}

async function readBody(r: Response): Promise<unknown> {
  const ct = r.headers.get('content-type') || ''
  if (ct.includes('application/json')) { try { return await r.json() } catch { return null } }
  return (await r.text()) || null
}

function mkLog(message: string, tone: ActivityLog['tone'] = 'info'): ActivityLog {
  return {
    id: `${Date.now()}_${Math.random().toString(36).slice(2, 9)}`,
    time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
    message,
    tone,
  }
}

export default function App() {
  const [loading, setLoading] = useState(true)
  const [scripts, setScripts] = useState<ScriptOption[]>([])
  const [languages, setLanguages] = useState<string[]>(['English'])
  const [models, setModels] = useState<Record<string, string>>({ '1.7B': '' })
  const [modelKey, setModelKey] = useState('1.7B')
  const [backend, setBackend] = useState('mlx')
  const [scriptText, setScriptText] = useState('')

  const [device, setDevice] = useState('auto')
  const [xVectorOnly, setXVectorOnly] = useState(false)
  const [cacheVoice, setCacheVoice] = useState(true)

  const [language, setLanguage] = useState('English')
  const [synthText, setSynthText] = useState(
    'You should probably set up a pass phrase with your loved ones, especially elderly family members. Something only you would know. If you got a phone call from this voice right now, asking for money, or asking for help, would you be able to tell it was not real?',
  )
  const [settings, setSettings] = useState<GenerationSettings>(defaultSettings)

  const [mics, setMics] = useState<MediaDeviceInfo[]>([])
  const [micId, setMicId] = useState('')

  const [blob, setBlob] = useState<Blob | null>(null)
  const [recUrl, setRecUrl] = useState('')
  const [recSecs, setRecSecs] = useState(0)
  const [recording, setRecording] = useState(false)

  const [training, setTraining] = useState(false)
  const [generating, setGenerating] = useState(false)

  const [profileId, setProfileId] = useState('')
  const [refUrl, setRefUrl] = useState('')
  const [gens, setGens] = useState<GenerationResult[]>([])
  const [elapsed, setElapsed] = useState(0)

  const [trainStatus, setTrainStatus] = useState('')
  const [synthStatus, setSynthStatus] = useState('')
  const [warning, setWarning] = useState('')
  const [error, setError] = useState('')
  const [logs, setLogs] = useState<ActivityLog[]>(() => [mkLog('waiting for backend...')])

  const [playbackRate, setPlaybackRate] = useState(1.0)
  const [useStreaming, setUseStreaming] = useState(false)
  const [showTrainSettings, setShowTrainSettings] = useState(false)
  const [showGenSettings, setShowGenSettings] = useState(false)

  const recorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const timerRef = useRef<number | null>(null)
  const genTimerRef = useRef<number | null>(null)
  const recUrlRef = useRef('')
  const audioCtxRef = useRef<AudioContext | null>(null)
  const nextStartRef = useRef<number>(0)

  const words = useMemo(() => (scriptText || '').trim().split(/\s+/).filter(Boolean).length, [scriptText])
  const estSecs = useMemo(() => Math.max(1, Math.floor(words / 2.8)), [words])
  const canTrain = Boolean(blob) && !training && !loading
  const canGen = Boolean(profileId) && !generating && !loading

  const spinR = useMemo(() => ({ rotate: 360 }), [])
  const spinT = useMemo(() => ({ repeat: Infinity, duration: 0.85, ease: 'linear' as const }), [])

  const scriptRef = useRef<HTMLTextAreaElement>(null)
  const synthRef = useRef<HTMLTextAreaElement>(null)

  const autoSize = useCallback((el: HTMLTextAreaElement | null) => {
    if (!el) return
    el.style.height = '0px'
    el.style.height = `${el.scrollHeight}px`
  }, [])

  useEffect(() => { autoSize(scriptRef.current) }, [scriptText, autoSize])
  useEffect(() => { autoSize(synthRef.current) }, [synthText, autoSize])

  function log(msg: string, tone: ActivityLog['tone'] = 'info') {
    setLogs(prev => [mkLog(msg, tone), ...prev].slice(0, 20))
  }

  // ── Config ──
  useEffect(() => {
    let ok = true
    async function load() {
      try {
        const r = await fetch(api('/api/config'))
        const p = await readBody(r)
        if (!r.ok) throw new Error(errMsg(p, 'Failed to load config.'))
        if (!ok) return
        const c = isObj(p) ? p : {}
        const s = Array.isArray(c.scripts)
          ? c.scripts.map(e => { if (isObj(e)) { const t = str(e, 'text'); if (t) return { text: t } } return null }).filter((e): e is ScriptOption => e !== null)
          : []
        const l = Array.isArray(c.languages)
          ? c.languages.filter((e): e is string => typeof e === 'string' && Boolean(String(e).trim()))
          : []
        const m = isObj(c.models) ? (c.models as Record<string, string>) : { '1.7B': '' }
        const dk = typeof c.default_model === 'string' ? c.default_model : '1.7B'
        const eng = typeof c.backend === 'string' ? c.backend : 'mlx'
        setScripts(s)
        setLanguages(l.length > 0 ? l : ['English'])
        setModels(m)
        setModelKey(dk)
        setBackend(eng)
        setScriptText(s[0]?.text || '')
        setLanguage(l.includes('English') ? 'English' : l[0] || 'English')
        log('ready', 'success')
      } catch (e) {
        if (!ok) return
        const m = e instanceof Error ? e.message : 'Config load failed.'
        setError(m); log(m, 'error')
      } finally { if (ok) setLoading(false) }
    }
    void load()
    return () => {
      ok = false
      stopRec(true)
      if (recUrlRef.current) URL.revokeObjectURL(recUrlRef.current)
      if (genTimerRef.current !== null) window.clearInterval(genTimerRef.current)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // ── Mic enumeration ──
  async function refreshMics() {
    try {
      const devs = await navigator.mediaDevices.enumerateDevices()
      const m = devs.filter(d => d.kind === 'audioinput' && d.deviceId)
      setMics(m)
      if (m.length > 0 && !micId) setMicId(m[0].deviceId)
    } catch {}
  }

  useEffect(() => {
    void refreshMics()
    navigator.mediaDevices?.addEventListener('devicechange', refreshMics)
    return () => { navigator.mediaDevices?.removeEventListener('devicechange', refreshMics) }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // ── Recording ──
  async function startRec() {
    setError(''); setWarning('')
    if (!navigator.mediaDevices?.getUserMedia) {
      const m = 'Browser does not support mic capture.'
      setError(m); log(m, 'error'); return
    }
    try {
      if (recUrl) URL.revokeObjectURL(recUrl)
      setRecUrl(''); recUrlRef.current = ''; setBlob(null); setSynthStatus('')
      const constraints: MediaTrackConstraints = micId ? { deviceId: { exact: micId } } : {}
      const stream = await navigator.mediaDevices.getUserMedia({ audio: constraints })
      streamRef.current = stream
      const types = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4']
      let opts: MediaRecorderOptions | undefined
      for (const t of types) { if (window.MediaRecorder?.isTypeSupported?.(t)) { opts = { mimeType: t }; break } }
      const rec = opts ? new MediaRecorder(stream, opts) : new MediaRecorder(stream)
      chunksRef.current = []
      rec.ondataavailable = (e: BlobEvent) => { if (e.data?.size > 0) chunksRef.current.push(e.data) }
      rec.onstop = () => {
        const b = new Blob(chunksRef.current, { type: rec.mimeType || 'audio/webm' })
        const u = URL.createObjectURL(b)
        setBlob(b); setRecUrl(u); recUrlRef.current = u; chunksRef.current = []
        log('recording captured', 'success')
      }
      rec.start(); recorderRef.current = rec; setRecSecs(0); setRecording(true)
      log('recording...')
      void refreshMics()
      timerRef.current = window.setInterval(() => setRecSecs(p => p + 1), 1000)
    } catch (e) {
      const m = e instanceof Error ? e.message : 'Unable to start recording.'
      setError(m); log(m, 'error'); stopRec(true)
    }
  }

  function stopRec(silent = false) {
    const rec = recorderRef.current
    const was = Boolean(rec && rec.state !== 'inactive')
    if (timerRef.current !== null) { window.clearInterval(timerRef.current); timerRef.current = null }
    if (was && rec) rec.stop()
    recorderRef.current = null
    if (streamRef.current) { streamRef.current.getTracks().forEach(t => t.stop()); streamRef.current = null }
    setRecording(false)
    if (!silent && was) log('stopped')
  }

  function clearRec() {
    if (recUrl) URL.revokeObjectURL(recUrl)
    setRecUrl(''); recUrlRef.current = ''; setBlob(null); setRecSecs(0)
    setProfileId(''); setRefUrl(''); setGens([]); setTrainStatus(''); setSynthStatus('')
    log('cleared', 'warning')
  }

  // ── Train ──
  async function handleTrain() {
    setError(''); setWarning(''); setSynthStatus('')
    if (!blob) { const m = 'Record first.'; setError(m); log(m, 'error'); return }
    const fd = new FormData()
    fd.append('audio', blob, `rec_${Date.now()}.webm`)
    fd.append('script_text', scriptText)
    fd.append('device', device)
    fd.append('x_vector_only', String(xVectorOnly))
    fd.append('cache_voice', String(cacheVoice))
    fd.append('model_key', modelKey)
    try {
      setTraining(true); setTrainStatus('training...')
      log('training started')
      const r = await fetch(api('/api/train'), { method: 'POST', body: fd })
      const p = await readBody(r)
      if (!r.ok) throw new Error(errMsg(p, 'Training failed.'))
      const d = isObj(p) ? p : {}
      const pid = str(d, 'profile_id')
      if (!pid) throw new Error('No profile id.')
      setProfileId(pid)
      setRefUrl(asset(str(d, 'reference_audio_url')))
      const s = str(d, 'status') || 'voice profile ready'
      const w = str(d, 'warning') || ''
      setTrainStatus(s); setWarning(w)
      log(s, 'success')
      if (w) log(w, 'warning')
    } catch (e) {
      const m = e instanceof Error ? e.message : 'Training failed.'
      setError(m); setTrainStatus('failed'); log(m, 'error')
    } finally { setTraining(false) }
  }

  // ── Synthesize (streaming) ──
  async function handleSynthStream() {
    setError(''); setWarning('')
    if (!profileId) { const m = 'Train a profile first.'; setError(m); log(m, 'error'); return }
    if (!synthText.trim()) { const m = 'Enter text.'; setError(m); log(m, 'error'); return }

    if (!audioCtxRef.current) audioCtxRef.current = new AudioContext()
    const ctx = audioCtxRef.current
    if (ctx.state === 'suspended') await ctx.resume()
    nextStartRef.current = ctx.currentTime

    setElapsed(0)
    const t0 = Date.now()
    genTimerRef.current = window.setInterval(() => setElapsed(Math.floor((Date.now() - t0) / 1000)), 1000)

    const overrides: Record<string, number> = {}
    if (settings.max_new_tokens !== null) overrides.max_new_tokens = settings.max_new_tokens
    if (settings.repetition_penalty !== null) overrides.repetition_penalty = settings.repetition_penalty
    if (settings.temperature !== null) overrides.temperature = settings.temperature
    if (settings.top_p !== null) overrides.top_p = settings.top_p

    try {
      setGenerating(true); setSynthStatus('streaming...')
      log('stream synthesis started')
      const r = await fetch(api('/api/synthesize-stream'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ profile_id: profileId, text: synthText, language, ...overrides }),
      })
      if (!r.ok) { const p = await readBody(r); throw new Error(errMsg(p, 'Stream failed.')) }

      const reader = r.body!.getReader()
      let buf = new Uint8Array(0)
      let outputUrl = ''
      let firstChunk = true

      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        const merged = new Uint8Array(buf.length + value.length)
        merged.set(buf); merged.set(value, buf.length); buf = merged

        while (buf.length >= 4) {
          const len = new DataView(buf.buffer, buf.byteOffset).getUint32(0, true)
          if (buf.length < 4 + len) break
          const payload = buf.slice(4, 4 + len)
          buf = buf.slice(4 + len)

          if (payload[0] === 0x7B) { // JSON frame
            try {
              const json = JSON.parse(new TextDecoder().decode(payload))
              if (json.error) throw new Error(json.error)
              if (json.output_audio_url) outputUrl = json.output_audio_url
            } catch (e) { throw e }
          } else { // PCM float32 frame
            const floats = new Float32Array(payload.buffer, payload.byteOffset, payload.byteLength / 4)
            const ab = ctx.createBuffer(1, floats.length, 24000)
            ab.copyToChannel(floats, 0)
            const src = ctx.createBufferSource()
            src.buffer = ab
            src.playbackRate.value = playbackRate
            src.connect(ctx.destination)
            const when = Math.max(nextStartRef.current, ctx.currentTime)
            src.start(when)
            nextStartRef.current = when + ab.duration
            if (firstChunk) { setSynthStatus('playing...'); firstChunk = false }
          }
        }
      }

      const secs = Math.floor((Date.now() - t0) / 1000)
      if (outputUrl) {
        setGens(prev => [{
          id: `${Date.now()}_${Math.random().toString(36).slice(2, 9)}`,
          url: asset(outputUrl),
          text: synthText.length > 80 ? synthText.slice(0, 80) + '...' : synthText,
          elapsed: secs,
          createdAt: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
        }, ...prev])
      }
      setSynthStatus(`done in ${secs}s`)
      log(`streamed in ${secs}s`, 'success')
    } catch (e) {
      const m = e instanceof Error ? e.message : 'Stream failed.'
      setError(m); setSynthStatus('failed'); log(m, 'error')
    } finally {
      if (genTimerRef.current !== null) { window.clearInterval(genTimerRef.current); genTimerRef.current = null }
      setGenerating(false)
    }
  }

  // ── Synthesize ──
  async function handleSynth() {
    setError(''); setWarning('')
    if (!profileId) { const m = 'Train a profile first.'; setError(m); log(m, 'error'); return }
    if (!synthText.trim()) { const m = 'Enter text.'; setError(m); log(m, 'error'); return }

    setElapsed(0)
    const t0 = Date.now()
    genTimerRef.current = window.setInterval(() => setElapsed(Math.floor((Date.now() - t0) / 1000)), 1000)

    const overrides: Record<string, number> = {}
    if (settings.max_new_tokens !== null) overrides.max_new_tokens = settings.max_new_tokens
    if (settings.repetition_penalty !== null) overrides.repetition_penalty = settings.repetition_penalty
    if (settings.temperature !== null) overrides.temperature = settings.temperature
    if (settings.top_p !== null) overrides.top_p = settings.top_p

    try {
      setGenerating(true); setSynthStatus('generating...')
      log('synthesis started')
      const r = await fetch(api('/api/synthesize'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ profile_id: profileId, text: synthText, language, ...overrides }),
      })
      const p = await readBody(r)
      if (!r.ok) throw new Error(errMsg(p, 'Generation failed.'))
      const d = isObj(p) ? p : {}
      const url = str(d, 'output_audio_url')
      if (!url) throw new Error('No audio URL.')
      const secs = Math.floor((Date.now() - t0) / 1000)
      setGens(prev => [{
        id: `${Date.now()}_${Math.random().toString(36).slice(2, 9)}`,
        url: asset(url),
        text: synthText.length > 80 ? synthText.slice(0, 80) + '...' : synthText,
        elapsed: secs,
        createdAt: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      }, ...prev])
      setSynthStatus(`done in ${secs}s`)
      log(`generated in ${secs}s`, 'success')
    } catch (e) {
      const m = e instanceof Error ? e.message : 'Generation failed.'
      setError(m); setSynthStatus('failed'); log(m, 'error')
    } finally {
      if (genTimerRef.current !== null) { window.clearInterval(genTimerRef.current); genTimerRef.current = null }
      setGenerating(false)
    }
  }

  return (
    <>
      <div className="noise" />
      <div className="shell">
        <header>
          <div className="logo">voice clone</div>
          <span className="backend-badge">{backend}</span>
        </header>

        <div className="workspace">
          {/* ── Left: Record + Train ── */}
          <div className="pane pane-left">
            <div className="pane-title">01 — Record</div>

            <div className="script-label">read this aloud</div>
            <textarea
              ref={scriptRef}
              className="script-area"
              value={scriptText}
              onChange={e => setScriptText(e.target.value)}
              aria-label="Training script"
            />
            <div className="meta">
              <span>{words} words</span>
              <span>/</span>
              <span>~{estSecs}s</span>
            </div>

            {mics.length > 1 && (
              <div className="mic-row">
                <span>mic</span>
                <select value={micId} onChange={e => setMicId(e.target.value)} disabled={recording}>
                  {mics.map(d => (
                    <option key={d.deviceId} value={d.deviceId}>
                      {d.label || `mic ${d.deviceId.slice(0, 8)}`}
                    </option>
                  ))}
                </select>
              </div>
            )}

            <div className="btn-row">
              {!recording ? (
                <button className="btn" onClick={() => void startRec()} disabled={loading}>
                  Record
                </button>
              ) : (
                <button className="btn btn-danger" onClick={() => stopRec(false)}>
                  Stop
                </button>
              )}
              <button className="btn-ghost" onClick={clearRec} disabled={recording}>clear</button>

              <AnimatePresence mode="wait">
                {recording ? (
                  <motion.span key="live" className="rec-badge live"
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                    <span className="rec-dot" />{recSecs}s
                  </motion.span>
                ) : blob ? (
                  <motion.span key="done" className="rec-badge"
                    initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
                    {recSecs}s captured
                  </motion.span>
                ) : null}
              </AnimatePresence>
            </div>

            {recUrl && <audio controls src={recUrl} className="audio-el" />}

            <button className="settings-toggle" onClick={() => setShowTrainSettings(s => !s)}>
              {showTrainSettings ? 'hide' : 'settings'}
            </button>

            {showTrainSettings && (
              <div className="settings-grid">
                <label>
                  model
                  <select value={modelKey} onChange={e => setModelKey(e.target.value)}>
                    {Object.keys(models).map(k => (
                      <option key={k} value={k}>{k === '1.7B' ? '1.7B (quality)' : '0.6B (fast)'}</option>
                    ))}
                  </select>
                </label>
                <label>
                  device
                  <select value={device} onChange={e => setDevice(e.target.value)}>
                    <option value="auto">auto</option>
                    <option value="cpu">cpu</option>
                    <option value="cuda">cuda</option>
                  </select>
                </label>
                <label>
                  cache
                  <select value={String(cacheVoice)} onChange={e => setCacheVoice(e.target.value === 'true')}>
                    <option value="true">on</option>
                    <option value="false">off</option>
                  </select>
                </label>
                <label className="check-row">
                  <input type="checkbox" checked={xVectorOnly} onChange={e => setXVectorOnly(e.target.checked)} />
                  x_vector_only (faster, lower quality)
                </label>
              </div>
            )}

            <button
              className="btn btn-solid btn-full"
              style={{ marginTop: 16 }}
              disabled={!canTrain}
              onClick={() => void handleTrain()}
            >
              {training ? 'training...' : 'train voice profile'}
            </button>

            <AnimatePresence>
              {training && (
                <motion.div className="spinner-row" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  <motion.span className="spin-ring" aria-hidden animate={spinR} transition={spinT} />
                  <span className="spin-text">{trainStatus || 'training...'}</span>
                </motion.div>
              )}
            </AnimatePresence>

            {!training && profileId && (
              <div>
                <p className="status status-ok">{trainStatus}</p>
                {refUrl && <audio controls src={refUrl} className="audio-el" />}
              </div>
            )}

            {/* Logs at bottom */}
            <div className="logs">
              {logs.slice(0, 6).map(l => (
                <div key={l.id} className={`log-entry log-${l.tone}`}>
                  <span className="log-time">{l.time}</span>
                  <span>{l.message}</span>
                </div>
              ))}
            </div>
          </div>

          {/* ── Right: Generate ── */}
          <div className="pane pane-right">
            <div className="pane-title">02 — Generate</div>

            <div className="script-label">type anything</div>
            <textarea
              ref={synthRef}
              className="synth-area"
              value={synthText}
              onChange={e => setSynthText(e.target.value)}
              aria-label="Synthesis text"
            />

            <div className="btn-row">
              <button
                className="btn btn-solid"
                disabled={!canGen}
                onClick={() => void (useStreaming ? handleSynthStream() : handleSynth())}
              >
                {generating ? (useStreaming ? 'streaming...' : 'generating...') : 'generate'}
              </button>
              <label className="check-row" style={{ margin: 0 }}>
                <input type="checkbox" checked={useStreaming} onChange={e => setUseStreaming(e.target.checked)} />
                stream
              </label>
            </div>

            <AnimatePresence>
              {generating && (
                <motion.div className="spinner-row" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  <motion.span className="spin-ring" aria-hidden animate={spinR} transition={spinT} />
                  <span className="spin-text">generating... {elapsed}s</span>
                </motion.div>
              )}
            </AnimatePresence>

            {!generating && synthStatus && <p className="status status-ok">{synthStatus}</p>}

            {gens.length > 0 ? (
              <div>
                {gens.map(g => (
                  <motion.div key={g.id} className="gen-item"
                    initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.2 }}>
                    <div className="gen-meta">
                      <span>{g.createdAt}</span>
                      <span>{g.elapsed}s</span>
                    </div>
                    <p className="gen-text">{g.text}</p>
                    <audio controls src={g.url} className="audio-el"
                      ref={el => { if (el) el.playbackRate = playbackRate }} />
                  </motion.div>
                ))}
              </div>
            ) : (
              !generating && <p className="status">output will appear here</p>
            )}

            <button className="settings-toggle" onClick={() => setShowGenSettings(s => !s)}>
              {showGenSettings ? 'hide' : 'generation settings'}
            </button>

            {showGenSettings && (
              <div className="settings-grid">
                <label style={{ gridColumn: '1 / -1' }}>
                  playback speed — {playbackRate.toFixed(2)}x
                  <input type="range" min={0.5} max={1.5} step={0.01}
                    value={playbackRate}
                    onChange={e => setPlaybackRate(Number(e.target.value))}
                    style={{ width: '100%' }}
                  />
                </label>
                <label>
                  language
                  <select value={language} onChange={e => setLanguage(e.target.value)}>
                    {languages.map(l => <option key={l} value={l}>{l}</option>)}
                  </select>
                </label>
                <label>
                  max tokens
                  <input type="number" min={200} max={4096} placeholder="default"
                    value={settings.max_new_tokens ?? ''}
                    onChange={e => setSettings(p => ({ ...p, max_new_tokens: e.target.value ? Number(e.target.value) : null }))} />
                </label>
                <label>
                  rep. penalty
                  <input type="number" min={1} max={2} step={0.05} placeholder="default"
                    value={settings.repetition_penalty ?? ''}
                    onChange={e => setSettings(p => ({ ...p, repetition_penalty: e.target.value ? Number(e.target.value) : null }))} />
                </label>
                <label>
                  temperature
                  <input type="number" min={0.01} max={2} step={0.05} placeholder="default"
                    value={settings.temperature ?? ''}
                    onChange={e => setSettings(p => ({ ...p, temperature: e.target.value ? Number(e.target.value) : null }))} />
                </label>
                <label>
                  top-p
                  <input type="number" min={0.01} max={1} step={0.05} placeholder="default"
                    value={settings.top_p ?? ''}
                    onChange={e => setSettings(p => ({ ...p, top_p: e.target.value ? Number(e.target.value) : null }))} />
                </label>
              </div>
            )}
          </div>
        </div>
      </div>

      {/* ── Notices ── */}
      <div className="notice-stack" role="status" aria-live="polite">
        <AnimatePresence>
          {error && (
            <motion.p key="error" className="notice notice-error"
              initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 8 }}
              transition={{ duration: 0.15 }}>
              {error}
            </motion.p>
          )}
        </AnimatePresence>
        <AnimatePresence>
          {warning && (
            <motion.p key="warning" className="notice notice-warning"
              initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 8 }}
              transition={{ duration: 0.15 }}>
              {warning}
            </motion.p>
          )}
        </AnimatePresence>
      </div>
    </>
  )
}
