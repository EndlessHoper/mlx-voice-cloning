import { useEffect, useMemo, useRef, useState, type RefObject } from 'react'
import { AnimatePresence, motion, useReducedMotion, type Variants } from 'framer-motion'

const API_BASE = (import.meta.env.VITE_API_BASE_URL || '').replace(/\/$/, '')

type LogTone = 'info' | 'success' | 'warning' | 'error'

interface ScriptOption {
  text: string
}

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
  tone: LogTone
}

interface GenerationResult {
  id: string
  url: string
  text: string
  elapsed: number
  createdAt: string
}

type SectionRef = RefObject<HTMLElement | null>

// null = let the SDK use its own internal defaults
const defaultSettings: GenerationSettings = {
  max_new_tokens: null,
  repetition_penalty: null,
  temperature: null,
  top_p: null,
}

function apiUrl(path: string): string {
  return API_BASE ? `${API_BASE}${path}` : path
}

function assetUrl(path: string | undefined): string {
  if (!path) return ''
  if (/^https?:\/\//.test(path)) return path
  return API_BASE ? `${API_BASE}${path}` : path
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value) && typeof value === 'object' && !Array.isArray(value)
}

function getStringField(record: Record<string, unknown>, key: string): string | undefined {
  const value = record[key]
  return typeof value === 'string' ? value : undefined
}

function getErrorMessage(payload: unknown, fallback: string): string {
  if (typeof payload === 'string' && payload.trim()) return payload
  if (!isRecord(payload)) return fallback
  const detail = getStringField(payload, 'detail')
  if (detail) return detail
  const message = getStringField(payload, 'message')
  if (message) return message
  return fallback
}

async function readResponsePayload(response: Response): Promise<unknown> {
  const contentType = response.headers.get('content-type') || ''
  if (contentType.includes('application/json')) {
    try { return await response.json() } catch { return null }
  }
  const text = await response.text()
  return text || null
}

function createLog(message: string, tone: LogTone = 'info'): ActivityLog {
  return {
    id: `${Date.now()}_${Math.random().toString(36).slice(2, 9)}`,
    time: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' }),
    message,
    tone,
  }
}

function autoSizeTextarea(textarea: HTMLTextAreaElement | null, minHeight = 120): void {
  if (!textarea) return
  textarea.style.height = '0px'
  textarea.style.height = `${Math.max(minHeight, textarea.scrollHeight)}px`
}

export default function App() {
  const [loadingConfig, setLoadingConfig] = useState(true)
  const [scripts, setScripts] = useState<ScriptOption[]>([])
  const [languages, setLanguages] = useState<string[]>(['English'])
  const [availableModels, setAvailableModels] = useState<Record<string, string>>({ '1.7B': '' })
  const [modelKey, setModelKey] = useState('1.7B')
  const [backendEngine, setBackendEngine] = useState('mlx')
  const [scriptText, setScriptText] = useState('')

  const [device, setDevice] = useState('auto')
  const [xVectorOnly, setXVectorOnly] = useState(false)
  const [cacheVoice, setCacheVoice] = useState(true)

  const [language, setLanguage] = useState('English')
  const [synthText, setSynthText] = useState(
    'You should set up a pass phrase with your loved ones, especially elderly family members. Something only you would know. Because if you got a phone call from this voice right now, asking for money, asking for help, would you be able to tell it was not real?',
  )
  const [settings, setSettings] = useState<GenerationSettings>(defaultSettings)

  const [audioDevices, setAudioDevices] = useState<MediaDeviceInfo[]>([])
  const [selectedDeviceId, setSelectedDeviceId] = useState('')

  const [recordingBlob, setRecordingBlob] = useState<Blob | null>(null)
  const [recordingUrl, setRecordingUrl] = useState('')
  const [recordingSeconds, setRecordingSeconds] = useState(0)
  const [isRecording, setIsRecording] = useState(false)

  const [training, setTraining] = useState(false)
  const [generating, setGenerating] = useState(false)

  const [profileId, setProfileId] = useState('')
  const [referenceAudioUrl, setReferenceAudioUrl] = useState('')
  const [generations, setGenerations] = useState<GenerationResult[]>([])
  const [genElapsed, setGenElapsed] = useState(0)

  const [trainStatus, setTrainStatus] = useState('')
  const [synthStatus, setSynthStatus] = useState('')
  const [warning, setWarning] = useState('')
  const [error, setError] = useState('')
  const [activityLogs, setActivityLogs] = useState<ActivityLog[]>(() => [createLog('Waiting for setup...')])

  const prefersReducedMotion = useReducedMotion()

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const timerRef = useRef<number | null>(null)
  const genTimerRef = useRef<number | null>(null)
  const recordingUrlRef = useRef('')
  // const audioCtxRef = useRef<AudioContext | null>(null)  // Reserved for future streaming playback
  // const streamAbortRef = useRef<AbortController | null>(null)

  const scriptInputRef = useRef<HTMLTextAreaElement | null>(null)
  const synthInputRef = useRef<HTMLTextAreaElement | null>(null)
  const recordSectionRef = useRef<HTMLElement | null>(null)
  const generateSectionRef = useRef<HTMLElement | null>(null)

  const wordCount = useMemo(() => (scriptText || '').trim().split(/\s+/).filter(Boolean).length, [scriptText])
  const estimatedSeconds = useMemo(() => Math.max(1, Math.floor(wordCount / 2.8)), [wordCount])
  const canTrain = Boolean(recordingBlob) && !training && !loadingConfig
  const canGenerate = Boolean(profileId) && !generating && !loadingConfig

  const sectionReveal: Variants = prefersReducedMotion
    ? { hidden: { opacity: 1 }, visible: { opacity: 1 } }
    : {
        hidden: { opacity: 0 },
        visible: {
          opacity: 1,
          transition: { duration: 0.9, ease: [0.16, 1, 0.3, 1], staggerChildren: 0.08 },
        },
      }

  const itemReveal: Variants = prefersReducedMotion
    ? { hidden: { opacity: 1 }, visible: { opacity: 1 } }
    : {
        hidden: { opacity: 0, y: 24 },
        visible: {
          opacity: 1,
          y: 0,
          transition: { duration: 0.65, ease: [0.16, 1, 0.3, 1] },
        },
      }

  function appendLog(message: string, tone: LogTone = 'info'): void {
    setActivityLogs((prev) => [createLog(message, tone), ...prev].slice(0, 24))
  }

  function scrollTo(ref: SectionRef): void {
    ref.current?.scrollIntoView({ behavior: 'smooth', block: 'start' })
  }

  // ── Config Load ──
  useEffect(() => {
    let isMounted = true
    async function loadConfig() {
      try {
        const response = await fetch(apiUrl('/api/config'))
        const payload = await readResponsePayload(response)
        if (!response.ok) throw new Error(getErrorMessage(payload, 'Failed to load configuration.'))
        if (!isMounted) return
        const config = isRecord(payload) ? payload : {}
        const loadedScripts = Array.isArray(config.scripts)
          ? config.scripts.map((e) => { if (isRecord(e)) { const t = getStringField(e, 'text'); if (t) return { text: t } } return null }).filter((e): e is ScriptOption => e !== null)
          : []
        const loadedLanguages = Array.isArray(config.languages)
          ? config.languages.filter((e): e is string => typeof e === 'string' && Boolean(String(e).trim()))
          : []
        const models = isRecord(config.models) ? (config.models as Record<string, string>) : { '1.7B': '' }
        const defaultModel = typeof config.default_model === 'string' ? config.default_model : '1.7B'
        const engine = typeof config.backend === 'string' ? config.backend : 'mlx'
        setScripts(loadedScripts)
        setLanguages(loadedLanguages.length > 0 ? loadedLanguages : ['English'])
        setAvailableModels(models)
        setModelKey(defaultModel)
        setBackendEngine(engine)
        setScriptText(loadedScripts[0]?.text || '')
        setLanguage(loadedLanguages.includes('English') ? 'English' : loadedLanguages[0] || 'English')
        appendLog('Ready.', 'success')
      } catch (err) {
        if (!isMounted) return
        const msg = err instanceof Error ? err.message : 'Failed to load config.'
        setError(msg)
        appendLog(msg, 'error')
      } finally {
        if (isMounted) setLoadingConfig(false)
      }
    }
    void loadConfig()
    return () => {
      isMounted = false
      stopRecording(true)
      if (recordingUrlRef.current) URL.revokeObjectURL(recordingUrlRef.current)
      if (genTimerRef.current !== null) window.clearInterval(genTimerRef.current)
      // streamAbortRef.current?.abort()
      // if (audioCtxRef.current) audioCtxRef.current.close().catch(() => {})
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => { autoSizeTextarea(scriptInputRef.current, 140) }, [scriptText])
  useEffect(() => { autoSizeTextarea(synthInputRef.current, 100) }, [synthText])

  // ── Enumerate audio input devices (without requesting mic permission on load) ──
  async function refreshAudioDevices(): Promise<void> {
    try {
      const devices = await navigator.mediaDevices.enumerateDevices()
      const mics = devices.filter((d) => d.kind === 'audioinput' && d.deviceId)
      setAudioDevices(mics)
      if (mics.length > 0 && !selectedDeviceId) setSelectedDeviceId(mics[0].deviceId)
    } catch {
      // No devices — leave empty, will use default
    }
  }
  useEffect(() => {
    void refreshAudioDevices()
    navigator.mediaDevices?.addEventListener('devicechange', refreshAudioDevices)
    return () => { navigator.mediaDevices?.removeEventListener('devicechange', refreshAudioDevices) }
  }, []) // eslint-disable-line react-hooks/exhaustive-deps

  // ── Recording ──
  async function startRecording(): Promise<void> {
    setError(''); setWarning('')
    if (!navigator.mediaDevices?.getUserMedia) {
      const m = 'Browser does not support microphone capture.'
      setError(m); appendLog(m, 'error'); return
    }
    try {
      if (recordingUrl) URL.revokeObjectURL(recordingUrl)
      setRecordingUrl(''); recordingUrlRef.current = ''; setRecordingBlob(null)
      setSynthStatus('')
      const audioConstraints: MediaTrackConstraints = selectedDeviceId ? { deviceId: { exact: selectedDeviceId } } : true
      const stream = await navigator.mediaDevices.getUserMedia({ audio: audioConstraints })
      streamRef.current = stream
      const mimeCandidates = ['audio/webm;codecs=opus', 'audio/webm', 'audio/mp4']
      let options: MediaRecorderOptions | undefined
      for (const mt of mimeCandidates) { if (window.MediaRecorder?.isTypeSupported?.(mt)) { options = { mimeType: mt }; break } }
      const recorder = options ? new MediaRecorder(stream, options) : new MediaRecorder(stream)
      chunksRef.current = []
      recorder.ondataavailable = (e: BlobEvent) => { if (e.data?.size > 0) chunksRef.current.push(e.data) }
      recorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: recorder.mimeType || 'audio/webm' })
        const url = URL.createObjectURL(blob)
        setRecordingBlob(blob); setRecordingUrl(url); recordingUrlRef.current = url; chunksRef.current = []
        appendLog('Recording captured.', 'success')
      }
      recorder.start(); mediaRecorderRef.current = recorder; setRecordingSeconds(0); setIsRecording(true)
      appendLog('Recording started.')
      void refreshAudioDevices() // Now that we have permission, get proper device labels
      timerRef.current = window.setInterval(() => setRecordingSeconds((p) => p + 1), 1000)
    } catch (err) {
      const m = err instanceof Error ? err.message : 'Unable to start recording.'
      setError(m); appendLog(m, 'error'); stopRecording(true)
    }
  }

  function stopRecording(silent = false): void {
    const rec = mediaRecorderRef.current
    const wasRec = Boolean(rec && rec.state !== 'inactive')
    if (timerRef.current !== null) { window.clearInterval(timerRef.current); timerRef.current = null }
    if (wasRec && rec) rec.stop()
    mediaRecorderRef.current = null
    if (streamRef.current) { streamRef.current.getTracks().forEach((t) => t.stop()); streamRef.current = null }
    setIsRecording(false)
    if (!silent && wasRec) appendLog('Recording stopped.')
  }

  function clearRecording(): void {
    if (recordingUrl) URL.revokeObjectURL(recordingUrl)
    setRecordingUrl(''); recordingUrlRef.current = ''; setRecordingBlob(null); setRecordingSeconds(0)
    setProfileId(''); setReferenceAudioUrl(''); setGenerations([]); setTrainStatus(''); setSynthStatus('')
    appendLog('Cleared.', 'warning')
  }

  // ── Train ──
  async function handleTrain(): Promise<void> {
    setError(''); setWarning(''); setSynthStatus('')
    if (!recordingBlob) { const m = 'Record first.'; setError(m); appendLog(m, 'error'); return }
    const formData = new FormData()
    formData.append('audio', recordingBlob, `recording_${Date.now()}.webm`)
    formData.append('script_text', scriptText)
    formData.append('device', device)
    formData.append('x_vector_only', String(xVectorOnly))
    formData.append('cache_voice', String(cacheVoice))
    formData.append('model_key', modelKey)
    try {
      setTraining(true); setTrainStatus('Training in progress...')
      appendLog('Training started.')
      const response = await fetch(apiUrl('/api/train'), { method: 'POST', body: formData })
      const payload = await readResponsePayload(response)
      if (!response.ok) throw new Error(getErrorMessage(payload, 'Training failed.'))
      const tp = isRecord(payload) ? payload : {}
      const pid = getStringField(tp, 'profile_id')
      if (!pid) throw new Error('No profile id returned.')
      setProfileId(pid)
      setReferenceAudioUrl(assetUrl(getStringField(tp, 'reference_audio_url')))
      const status = getStringField(tp, 'status') || 'Voice profile trained.'
      const warn = getStringField(tp, 'warning') || ''
      setTrainStatus(status); setWarning(warn)
      appendLog(status, 'success')
      if (warn) appendLog(warn, 'warning')
      // Auto-scroll to generate section
      window.setTimeout(() => scrollTo(generateSectionRef), 500)
    } catch (err) {
      const m = err instanceof Error ? err.message : 'Training failed.'
      setError(m); setTrainStatus('Training failed.'); appendLog(m, 'error')
    } finally { setTraining(false) }
  }

  // ── Synthesize ──
  async function handleSynthesize(): Promise<void> {
    setError(''); setWarning('')
    if (!profileId) { const m = 'Train a profile first.'; setError(m); appendLog(m, 'error'); return }
    if (!synthText.trim()) { const m = 'Enter text.'; setError(m); appendLog(m, 'error'); return }

    // Start elapsed timer
    setGenElapsed(0)
    const startTime = Date.now()
    genTimerRef.current = window.setInterval(() => {
      setGenElapsed(Math.floor((Date.now() - startTime) / 1000))
    }, 1000)

    // Build request body
    const overrides: Record<string, number> = {}
    if (settings.max_new_tokens !== null) overrides.max_new_tokens = settings.max_new_tokens
    if (settings.repetition_penalty !== null) overrides.repetition_penalty = settings.repetition_penalty
    if (settings.temperature !== null) overrides.temperature = settings.temperature
    if (settings.top_p !== null) overrides.top_p = settings.top_p
    const body = JSON.stringify({ profile_id: profileId, text: synthText, language, ...overrides })

    try {
      setGenerating(true); setSynthStatus('Generating...')
      appendLog('Synthesis started.')

      const response = await fetch(apiUrl('/api/synthesize'), {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
      })

      const payload = await readResponsePayload(response)
      if (!response.ok) throw new Error(getErrorMessage(payload, 'Generation failed.'))

      const data = isRecord(payload) ? payload : {}
      const audioUrl = getStringField(data, 'output_audio_url')
      if (!audioUrl) throw new Error('No audio URL returned.')

      const elapsed = Math.floor((Date.now() - startTime) / 1000)
      const result: GenerationResult = {
        id: `${Date.now()}_${Math.random().toString(36).slice(2, 9)}`,
        url: assetUrl(audioUrl),
        text: synthText.length > 80 ? synthText.slice(0, 80) + '...' : synthText,
        elapsed,
        createdAt: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }),
      }
      setGenerations((prev) => [result, ...prev])
      setSynthStatus(`Generated in ${elapsed}s`)
      appendLog(`Generated in ${elapsed}s`, 'success')
    } catch (err) {
      const m = err instanceof Error ? err.message : 'Generation failed.'
      setError(m); setSynthStatus('Failed.'); appendLog(m, 'error')
    } finally {
      if (genTimerRef.current !== null) { window.clearInterval(genTimerRef.current); genTimerRef.current = null }
      setGenerating(false)
    }
  }

  // ── Inline Log Component ──
  function InlineLogs({ max = 4 }: { max?: number }) {
    const visible = activityLogs.slice(0, max)
    return (
      <div className="inline-logs">
        <AnimatePresence initial={false}>
          {visible.map((entry) => (
            <motion.div
              key={entry.id}
              layout
              className={`ilog ilog-${entry.tone}`}
              initial={prefersReducedMotion ? { opacity: 1 } : { opacity: 0, x: -8 }}
              animate={{ opacity: 1, x: 0 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2, ease: 'easeOut' }}
            >
              <span className="ilog-time">{entry.time}</span>
              <span className="ilog-msg">{entry.message}</span>
            </motion.div>
          ))}
        </AnimatePresence>
      </div>
    )
  }

  // ── Spinner (uses memo-stable ref to avoid re-mount flicker on timer ticks) ──
  const spinnerRotate = useMemo(() => ({ rotate: 360 }), [])
  const spinnerTransition = useMemo(() => ({ repeat: Infinity, duration: 0.85, ease: 'linear' as const }), [])

  return (
    <div className="experience-shell">
      {/* ── Nav dots ── */}
      <nav className="step-rail" aria-label="Jump to section">
        {[
          { ref: recordSectionRef, l: 'i' },
          { ref: generateSectionRef, l: 'ii' },
        ].map((s) => (
          <motion.button
            key={s.l}
            type="button"
            className="step-dot"
            onClick={() => scrollTo(s.ref)}
            whileHover={prefersReducedMotion ? undefined : { x: -3 }}
            whileTap={prefersReducedMotion ? undefined : { scale: 0.9 }}
          >
            {s.l}
          </motion.button>
        ))}
      </nav>

      <main className="experience">

        {/* ═══════ Section 1: Record + Train ═══════ */}
        <motion.section
          ref={recordSectionRef}
          className="panel panel-record"
          variants={sectionReveal}
          initial="visible"
          animate="visible"
          whileInView="visible"
          viewport={{ amount: 0.2, once: true }}
        >
          <div className="panel-inner">
            <motion.p className="step-tag" variants={itemReveal}>Record</motion.p>
            <motion.h1 variants={itemReveal}>Your voice,<br />reproduced.</motion.h1>
            <motion.p className="lead" variants={itemReveal}>
              Read the script below out loud, naturally. It's designed to capture the full
              range of your voice, covering different sounds, rhythms, and tones. Speak
              like you would in conversation.
            </motion.p>

            <motion.div className="meta-row" variants={itemReveal}>
              <span>{wordCount} words</span>
              <span className="meta-sep">/</span>
              <span>~{estimatedSeconds}s</span>
            </motion.div>

            <AnimatePresence mode="wait">
              {loadingConfig && (
                <motion.p className="status-line" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  Loading scripts...
                </motion.p>
              )}
            </AnimatePresence>

            <motion.div className="script-block" variants={itemReveal}>
              <h3>Say this</h3>
              <textarea
                ref={scriptInputRef}
                className="script-input"
                value={scriptText}
                onChange={(e) => setScriptText(e.target.value)}
                rows={5}
                aria-label="Training script"
              />
            </motion.div>

            <motion.div className="controls" variants={itemReveal}>
              {audioDevices.length > 1 && (
                <div className="mic-select-row">
                  <label>
                    mic
                    <select
                      value={selectedDeviceId}
                      onChange={(e) => setSelectedDeviceId(e.target.value)}
                      disabled={isRecording}
                    >
                      {audioDevices.map((d) => (
                        <option key={d.deviceId} value={d.deviceId}>
                          {d.label || `Microphone ${d.deviceId.slice(0, 8)}`}
                        </option>
                      ))}
                    </select>
                  </label>
                </div>
              )}
              <div className="controls-row">
                {!isRecording ? (
                  <button type="button" className="btn btn-primary" onClick={() => void startRecording()} disabled={loadingConfig}>
                    Record
                  </button>
                ) : (
                  <button type="button" className="btn btn-primary" onClick={() => stopRecording(false)}>
                    Stop
                  </button>
                )}
                <button type="button" className="btn-link" onClick={clearRecording} disabled={isRecording}>
                  Clear
                </button>

                <AnimatePresence mode="wait">
                  {isRecording ? (
                    <motion.span
                      key="live"
                      className="rec-badge live"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                      exit={{ opacity: 0 }}
                    >
                      <span className="rec-dot" />
                      {recordingSeconds}s
                    </motion.span>
                  ) : recordingBlob ? (
                    <motion.span
                      key="done"
                      className="rec-badge"
                      initial={{ opacity: 0 }}
                      animate={{ opacity: 1 }}
                    >
                      {recordingSeconds}s captured
                    </motion.span>
                  ) : null}
                </AnimatePresence>
              </div>

              {recordingUrl && <audio controls src={recordingUrl} className="audio-el" />}

              <details className="advanced-panel">
                <summary>Training settings <span className="settings-badge">{backendEngine.toUpperCase()}</span></summary>
                <div className="settings-row">
                  <label>
                    model
                    <select value={modelKey} onChange={(e) => setModelKey(e.target.value)}>
                      {Object.keys(availableModels).map((k) => (
                        <option key={k} value={k}>{k === '1.7B' ? '1.7B (Quality)' : '0.6B (Fast)'}</option>
                      ))}
                    </select>
                  </label>
                  <label>
                    device
                    <select value={device} onChange={(e) => setDevice(e.target.value)}>
                      <option value="auto">auto</option>
                      <option value="cpu">cpu</option>
                      <option value="cuda">cuda</option>
                    </select>
                  </label>
                  <label>
                    cache
                    <select value={String(cacheVoice)} onChange={(e) => setCacheVoice(e.target.value === 'true')}>
                      <option value="true">on</option>
                      <option value="false">off</option>
                    </select>
                  </label>
                </div>
                <label className="check-row">
                  <input type="checkbox" checked={xVectorOnly} onChange={(e) => setXVectorOnly(e.target.checked)} />
                  x_vector_only (faster, lower quality)
                </label>
              </details>

              <button type="button" className="btn btn-primary btn-full" disabled={!canTrain} onClick={() => void handleTrain()}>
                {training ? 'Training...' : 'Train Voice Profile'}
              </button>

              {/* Inline training progress */}
              <AnimatePresence>
                {training && (
                  <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                    <div className="inline-spinner">
                      <motion.span
                        className="spin-ring"
                        aria-hidden="true"
                        animate={!prefersReducedMotion ? spinnerRotate : {}}
                        transition={spinnerTransition}
                      />
                      <span className="spin-label">{trainStatus || 'Training in progress...'}</span>
                    </div>
                    <InlineLogs max={4} />
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Post-training status */}
              <AnimatePresence>
                {!training && profileId && (
                  <motion.div
                    initial={prefersReducedMotion ? {} : { opacity: 0, y: 12 }}
                    animate={{ opacity: 1, y: 0 }}
                    className="train-done"
                  >
                    <p className="status-line">{trainStatus}</p>
                    {referenceAudioUrl && (
                      <>
                        <p className="status-line">Reference audio:</p>
                        <audio controls src={referenceAudioUrl} className="audio-el" />
                      </>
                    )}
                  </motion.div>
                )}
              </AnimatePresence>
            </motion.div>
          </div>
        </motion.section>

        {/* ═══════ Section 2: Generate ═══════ */}
        <motion.section
          ref={generateSectionRef}
          className="panel panel-generate"
          variants={sectionReveal}
          initial="hidden"
          whileInView="visible"
          viewport={{ amount: 0.2, once: false }}
        >
          <div className="panel-inner">
            <motion.p className="step-tag" variants={itemReveal}>Generate</motion.p>
            <motion.h2 variants={itemReveal}>Say anything.</motion.h2>
            <motion.p className="lead" variants={itemReveal}>
              Write what you want to hear. Generate it in your voice.
            </motion.p>

            <motion.div className="script-block" variants={itemReveal}>
              <h3>Your text</h3>
              <textarea
                ref={synthInputRef}
                className="synth-input"
                value={synthText}
                onChange={(e) => setSynthText(e.target.value)}
                rows={4}
                aria-label="Synthesis text"
              />
            </motion.div>

            <motion.div className="controls-row" variants={itemReveal}>
              <button type="button" className="btn btn-primary" disabled={!canGenerate} onClick={() => void handleSynthesize()}>
                {generating ? 'Generating...' : 'Generate Speech'}
              </button>
            </motion.div>

            {/* Inline generation progress */}
            <AnimatePresence>
              {generating && (
                <motion.div className="inline-spinner" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}>
                  <motion.span
                    className="spin-ring"
                    aria-hidden="true"
                    animate={!prefersReducedMotion ? spinnerRotate : {}}
                    transition={spinnerTransition}
                  />
                  <span className="spin-label">{`Generating... ${genElapsed}s`}</span>
                </motion.div>
              )}
            </AnimatePresence>

            {!generating && synthStatus && <p className="status-line">{synthStatus}</p>}

            {/* Generation history */}
            {generations.length > 0 ? (
              <div className="generation-list">
                {generations.map((gen) => (
                  <motion.div
                    key={gen.id}
                    className="generation-item"
                    initial={prefersReducedMotion ? {} : { opacity: 0, y: 8 }}
                    animate={{ opacity: 1, y: 0 }}
                    transition={{ duration: 0.3 }}
                  >
                    <div className="generation-meta">
                      <span className="generation-time">{gen.createdAt}</span>
                      <span className="generation-elapsed">{gen.elapsed}s</span>
                    </div>
                    <p className="generation-text">{gen.text}</p>
                    <audio controls src={gen.url} className="audio-el" />
                  </motion.div>
                ))}
              </div>
            ) : (
              !generating && <p className="status-line">Audio will appear here after generation.</p>
            )}

            <InlineLogs max={4} />

            <details className="advanced-panel">
              <summary>Language + model settings</summary>
              <div className="settings-grid">
                <label>
                  language
                  <select value={language} onChange={(e) => setLanguage(e.target.value)}>
                    {languages.map((l) => <option key={l} value={l}>{l}</option>)}
                  </select>
                </label>
                <label>
                  max tokens
                  <input type="number" min={200} max={4096} placeholder="SDK default"
                    value={settings.max_new_tokens ?? ''}
                    onChange={(e) => setSettings((p) => ({ ...p, max_new_tokens: e.target.value ? Number(e.target.value) : null }))} />
                </label>
                <label>
                  rep. penalty
                  <input type="number" min={1} max={2} step={0.05} placeholder="SDK default"
                    value={settings.repetition_penalty ?? ''}
                    onChange={(e) => setSettings((p) => ({ ...p, repetition_penalty: e.target.value ? Number(e.target.value) : null }))} />
                </label>
                <label>
                  temperature
                  <input type="number" min={0.01} max={2} step={0.05} placeholder="SDK default"
                    value={settings.temperature ?? ''}
                    onChange={(e) => setSettings((p) => ({ ...p, temperature: e.target.value ? Number(e.target.value) : null }))} />
                </label>
                <label>
                  top-p
                  <input type="number" min={0.01} max={1} step={0.05} placeholder="SDK default"
                    value={settings.top_p ?? ''}
                    onChange={(e) => setSettings((p) => ({ ...p, top_p: e.target.value ? Number(e.target.value) : null }))} />
                </label>
              </div>
            </details>
          </div>
        </motion.section>
      </main>

      {/* ── Notices ── */}
      <div className="notice-stack" role="status" aria-live="polite">
        <AnimatePresence>
          {error && (
            <motion.p key="error" className="notice notice-error"
              initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 10 }}
              transition={{ duration: 0.2 }}>
              {error}
            </motion.p>
          )}
        </AnimatePresence>
        <AnimatePresence>
          {warning && (
            <motion.p key="warning" className="notice notice-warning"
              initial={{ opacity: 0, y: 10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: 10 }}
              transition={{ duration: 0.2 }}>
              {warning}
            </motion.p>
          )}
        </AnimatePresence>
      </div>
    </div>
  )
}
