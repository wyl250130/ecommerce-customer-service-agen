import { FormEvent, useCallback, useEffect, useRef, useState } from 'react'
import { allMessages, api, ApiError, type ChatRequest, type Message, type Session } from './api'

type PendingSend = { sessionId: string; content: string; key: string }
const ACTIVE_KEY = 'cs-v1-active-session'
const PENDING_KEY = 'cs-v1-pending-send'

function readPending(): PendingSend | null {
  try {
    const saved = JSON.parse(localStorage.getItem(PENDING_KEY) || 'null') as PendingSend | null
    return saved?.sessionId && saved.content && saved.key ? saved : null
  } catch { return null }
}

function outcomeText(result: ChatRequest): string {
  if (result.result_type === 'handoff' || result.result_type === 'transferred') return '已提交模拟人工接管'
  if (result.result_type === 'fallback') return '已提供兜底建议'
  if (result.result_type === 'needs_user') return '等待你补充信息'
  return '已完成回复'
}

function shortDate(value: string): string {
  return new Date(value).toLocaleDateString('zh-CN', { month: 'numeric', day: 'numeric' })
}

export default function App() {
  const [sessions, setSessions] = useState<Session[]>([])
  const [activeId, setActiveId] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [draft, setDraft] = useState('')
  const [pending, setPending] = useState<PendingSend | null>(readPending)
  const [working, setWorking] = useState(false)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState('')
  const [notice, setNotice] = useState('')
  const [menuOpen, setMenuOpen] = useState(false)
  const busyRef = useRef(false)
  const endRef = useRef<HTMLDivElement>(null)

  const savePending = (value: PendingSend | null) => {
    setPending(value)
    if (value) localStorage.setItem(PENDING_KEY, JSON.stringify(value))
    else localStorage.removeItem(PENDING_KEY)
  }

  const refreshMessages = useCallback(async (sessionId: string) => {
    setMessages(await allMessages(sessionId))
  }, [])

  const refreshSessions = useCallback(async () => {
    setSessions(await api.sessions())
  }, [])

  const settle = useCallback(async (result: ChatRequest, item: PendingSend): Promise<boolean> => {
    let current = result
    for (let attempt = 0; attempt < 45; attempt++) {
      if (current.execution_status === 'succeeded') {
        await refreshMessages(item.sessionId)
        await refreshSessions()
        savePending(null)
        setDraft('')
        setNotice(outcomeText(current))
        setError('')
        return true
      }
      if (current.execution_status === 'failed') {
        await refreshMessages(item.sessionId)
        savePending(null)
        setDraft(item.content)
        setError('本次执行未完成。可修改内容后发起新请求。')
        return true
      }
      await new Promise(resolve => setTimeout(resolve, 2000))
      current = await api.requestById(current.id)
    }
    setError('请求仍在处理中，请稍后点击“检查请求状态”。')
    return false
  }, [refreshMessages, refreshSessions])

  const recover = useCallback(async (item: PendingSend, allowSubmit: boolean) => {
    if (busyRef.current) return
    busyRef.current = true
    setWorking(true)
    setError('')
    try {
      let result: ChatRequest
      try {
        result = await api.requestByKey(item.sessionId, item.key)
      } catch (lookupError) {
        if (lookupError instanceof ApiError && lookupError.status === 404) {
          if (!allowSubmit) {
            setError('尚未确认消息是否送达。点击“安全重试”会沿用原幂等键。')
            return
          }
          result = await api.send(item.sessionId, item.content, item.key)
        } else throw lookupError
      }
      await settle(result, item)
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : '网络中断，尚不能确认结果。请检查请求状态。')
    } finally {
      busyRef.current = false
      setWorking(false)
    }
  }, [settle])

  useEffect(() => {
    let cancelled = false
    async function initialize() {
      try {
        const rows = await api.sessions()
        if (cancelled) return
        setSessions(rows)
        const savedId = localStorage.getItem(ACTIVE_KEY)
        const selected = rows.find(row => row.id === savedId)?.id || rows[0]?.id || null
        setActiveId(selected)
        if (selected) {
          localStorage.setItem(ACTIVE_KEY, selected)
          await refreshMessages(selected)
        }
        const lost = readPending()
        if (lost && rows.some(row => row.id === lost.sessionId)) {
          setActiveId(lost.sessionId)
          localStorage.setItem(ACTIVE_KEY, lost.sessionId)
          if (selected !== lost.sessionId) await refreshMessages(lost.sessionId)
          await recover(lost, false)
        }
      } catch (caught) {
        if (!cancelled) setError(caught instanceof ApiError ? caught.message : '无法连接客服服务，请确认后端已启动。')
      } finally { if (!cancelled) setLoading(false) }
    }
    void initialize()
    return () => { cancelled = true }
  }, [refreshMessages, recover])

  useEffect(() => { endRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages, pending])

  async function createSession() {
    if (working || pending) return
    setWorking(true)
    setError('')
    try {
      const created = await api.createSession()
      const row: Session = { ...created, created_at: new Date().toISOString() }
      setSessions(current => [row, ...current])
      setActiveId(row.id)
      setMessages([])
      setNotice('新会话已创建')
      localStorage.setItem(ACTIVE_KEY, row.id)
      setMenuOpen(false)
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : '创建会话失败，请稍后重试。')
    } finally { setWorking(false) }
  }

  async function selectSession(id: string) {
    if (working || pending || id === activeId) return
    setActiveId(id)
    setMessages([])
    setNotice('')
    setError('')
    localStorage.setItem(ACTIVE_KEY, id)
    setMenuOpen(false)
    try { await refreshMessages(id) }
    catch { setError('历史消息加载失败，请重新选择会话。') }
  }

  async function send(event: FormEvent) {
    event.preventDefault()
    const content = draft.trim()
    if (!activeId || !content || working || pending || busyRef.current) return
    const item = { sessionId: activeId, content, key: crypto.randomUUID() }
    savePending(item)
    busyRef.current = true
    setWorking(true)
    setError('')
    setNotice('')
    try {
      await settle(await api.send(item.sessionId, item.content, item.key), item)
    } catch {
      // The response may be lost after the server committed. Query by the
      // same key; never create a fresh request automatically.
      setError('网络中断，正在查询原请求状态。')
      busyRef.current = false
      setWorking(false)
      await recover(item, false)
      return
    } finally {
      busyRef.current = false
      setWorking(false)
    }
  }

  const active = sessions.find(row => row.id === activeId)
  const transferred = active?.service_status === 'transferred'
  const canSend = Boolean(activeId && draft.trim() && !working && !pending && !transferred)

  return <div className="app-shell">
    <aside className={`sidebar ${menuOpen ? 'open' : ''}`}>
      <div className="brand"><div className="brand-mark">知</div><div><strong>知心客服</strong><span>智能服务工作台</span></div></div>
      <button className="new-session" onClick={() => void createSession()} disabled={working || Boolean(pending)}><span>＋</span> 新建会话</button>
      <div className="side-label">最近会话 <span>{sessions.length}</span></div>
      <nav className="session-list" aria-label="会话列表">
        {sessions.map((session, index) => <button key={session.id} className={`session-item ${activeId === session.id ? 'selected' : ''}`}
          disabled={working || Boolean(pending)} onClick={() => void selectSession(session.id)}>
          <span className="session-symbol">✦</span><span className="session-copy"><strong>客服会话 {sessions.length - index}</strong><small>{shortDate(session.created_at)} · {session.service_status === 'transferred' ? '已转人工' : '智能服务'}</small></span>
        </button>)}
        {!loading && sessions.length === 0 && <p className="empty-side">还没有会话，点击上方开始。</p>}
      </nav>
      <div className="side-bottom"><span className="dev-dot" /> 开发模式 · 模拟身份<div>订单事实来自本地模拟数据</div></div>
    </aside>
    {menuOpen && <button className="mobile-overlay" aria-label="关闭菜单" onClick={() => setMenuOpen(false)} />}
    <main className="main-panel">
      <header className="topbar"><button className="menu-toggle" aria-label="打开会话列表" onClick={() => setMenuOpen(true)}>☰</button><div className="top-title"><strong>{active ? '智能客服' : '欢迎使用知心客服'}</strong><span>{active ? '基于可信订单事实为你解答' : '创建一个会话，开始咨询'}</span></div><div className="top-status"><span className="online-dot" /> 本地演示</div></header>
      <div className="conversation">
        {!active && error && <div className="alert error" role="alert">{error}</div>}
        {!active && !loading && <div className="welcome"><div className="welcome-icon">✦</div><span className="eyebrow">CUSTOMER SUPPORT</span><h1>你好，有什么可以帮你？</h1><p>创建会话后，可以连续咨询订单、发货、物流和商品信息。回答会基于当前授权的模拟事实。</p><button onClick={() => void createSession()} disabled={working}>开始新会话 <span>→</span></button></div>}
        {active && <div className="chat-content"><div className="chat-intro"><div className="assistant-avatar">知</div><h2>你好，我是知心客服</h2><p>请告诉我你想查询的问题。你也可以在同一会话中继续追问。</p></div>
          {messages.map(message => <div key={message.id} className={`message-row ${message.sender_type === 'user' ? 'user' : 'assistant'}`}>
            {message.sender_type !== 'user' && <div className="message-avatar">知</div>}
            <div className="message-body"><span className="message-author">{message.sender_type === 'user' ? '你' : '知心客服'}</span><div className="message-bubble">{message.content}</div></div>
          </div>)}
          {pending?.sessionId === activeId && <div className="message-row user"><div className="message-body"><span className="message-author">你 · 发送中</span><div className="message-bubble pending-bubble">{pending.content}</div></div></div>}
          {working && pending && <div className="thinking"><span className="thinking-dots"><i /><i /><i /></span> 正在查询并整理回复…</div>}
          {transferred && <div className="handoff-banner"><strong>已提交模拟人工接管</strong><span>此会话已停止自动回复。这里是演示状态，不代表已接入真人客服。</span></div>}
          <div ref={endRef} /></div>}
      </div>
      {active && <div className="composer-area">
        {error && <div className="alert error" role="alert"><span>{error}</span>{pending && <button onClick={() => void recover(pending, true)} disabled={working}>{working ? '检查中…' : '安全重试 / 检查请求状态'}</button>}</div>}
        {notice && !error && <div className="alert notice" role="status">✓ {notice}</div>}
        <form className="composer" onSubmit={event => void send(event)}><textarea aria-label="输入消息" value={draft} onChange={event => setDraft(event.target.value)}
          placeholder={transferred ? '此会话已转入模拟人工接管' : '输入你的问题，例如：这个订单什么时候送达？'} maxLength={4000}
          disabled={Boolean(transferred || pending)} onKeyDown={event => { if (event.key === 'Enter' && !event.shiftKey && !event.nativeEvent.isComposing) { event.preventDefault(); void send(event as unknown as FormEvent) } }} />
          <div className="composer-footer"><span>Enter 发送 · Shift + Enter 换行</span><button type="submit" disabled={!canSend}>{working ? '发送中…' : '发送消息'} <span>↗</span></button></div>
        </form><p className="disclaimer">开发演示 · 模拟身份与事实数据 · 重要决定请以实际业务记录为准</p>
      </div>}
    </main>
  </div>
}
