export type Session = {
  id: string
  service_status: 'active' | 'handoff_pending' | 'transferred' | 'closed'
  created_at: string
}

export type Message = {
  id: string
  request_id: string | null
  sender_type: 'user' | 'assistant' | 'human_agent' | 'system'
  content: string
  sequence_no: number
}

export type ChatRequest = {
  id: string
  session_id: string
  execution_status: 'pending' | 'running' | 'succeeded' | 'failed'
  result_type: 'completed' | 'needs_user' | 'fallback' | 'handoff' | 'transferred' | null
  response: string | null
  error_code: string | null
}

export class ApiError extends Error {
  constructor(public status: number, public code: string, message: string) {
    super(message)
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...init?.headers },
  })
  let body: unknown
  try {
    body = await response.json()
  } catch {
    throw new ApiError(response.status, 'invalid_response', '服务返回了无法读取的内容')
  }
  if (!response.ok) {
    const error = body as { error?: string; message?: string }
    throw new ApiError(response.status, error.error || 'request_failed', error.message || '请求失败')
  }
  return body as T
}

export const api = {
  sessions: () => request<Session[]>('/api/sessions'),
  createSession: () => request<Pick<Session, 'id' | 'service_status'>>('/api/sessions', { method: 'POST' }),
  messages: (sessionId: string, after = 0) =>
    request<Message[]>(`/api/sessions/${sessionId}/messages?after_sequence=${after}&limit=100`),
  send: (sessionId: string, content: string, idempotencyKey: string) =>
    request<ChatRequest>(`/api/sessions/${sessionId}/messages`, {
      method: 'POST',
      body: JSON.stringify({ content, idempotency_key: idempotencyKey }),
    }),
  requestById: (id: string) => request<ChatRequest>(`/api/chat-requests/${id}`),
  requestByKey: (sessionId: string, key: string) =>
    request<ChatRequest>(`/api/sessions/${sessionId}/chat-requests/by-key/${encodeURIComponent(key)}`),
}

export async function allMessages(sessionId: string): Promise<Message[]> {
  const rows: Message[] = []
  let cursor = 0
  while (true) {
    const page = await api.messages(sessionId, cursor)
    rows.push(...page)
    if (page.length < 100) break
    cursor = page[page.length - 1].sequence_no
  }
  return rows
}
