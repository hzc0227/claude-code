import type { ClientOptions } from '@anthropic-ai/sdk'
import { logForDebugging } from '../../utils/debug.js'

/**
 * Minimal Anthropic message create payload shape that we need in order to
 * translate this project's existing request pipeline into an OpenAI-compatible
 * `chat/completions` request.
 */
type AnthropicMessageCreateBody = {
  model: string
  max_tokens?: number
  messages: Array<{
    role: 'assistant' | 'user'
    content: string | Array<Record<string, unknown>>
  }>
  stop_sequences?: string[]
  system?: string | Array<Record<string, unknown>>
  temperature?: number
  tool_choice?:
    | {
        type: 'any' | 'auto' | 'none'
        disable_parallel_tool_use?: boolean
      }
    | {
        type: 'tool'
        name: string
        disable_parallel_tool_use?: boolean
      }
  tools?: Array<{
    name: string
    description?: string
    input_schema?: Record<string, unknown>
  }>
  top_p?: number
  stream?: boolean
}

type OpenAIUserContentPart =
  | {
      type: 'text'
      text: string
    }
  | {
      type: 'image_url'
      image_url: {
        url: string
      }
    }

type OpenAIChatMessage =
  | {
      role: 'assistant'
      content?: string | null
      tool_calls?: OpenAIToolCall[]
    }
  | {
      role: 'system' | 'user'
      content: string | OpenAIUserContentPart[]
      tool_call_id?: string
    }
  | {
      role: 'tool'
      content: string
      tool_call_id?: string
    }

type OpenAIChatRequest = {
  model: string
  messages: OpenAIChatMessage[]
  max_tokens?: number
  stop?: string[]
  stream?: boolean
  stream_options?: {
    include_usage?: boolean
  }
  temperature?: number
  tools?: Array<{
    type: 'function'
    function: {
      name: string
      description?: string
      parameters?: Record<string, unknown>
    }
  }>
  tool_choice?:
    | 'auto'
    | 'none'
    | 'required'
    | {
        type: 'function'
        function: {
          name: string
        }
      }
  top_p?: number
}

type OpenAIToolCall = {
  id: string
  type: 'function'
  function: {
    name: string
    arguments: string
  }
}

type OpenAIChatChoice = {
  index: number
  finish_reason?: string | null
  message?: {
    role?: string
    content?: string | Array<Record<string, unknown>> | null
    tool_calls?: OpenAIToolCall[]
  }
  delta?: {
    role?: string
    content?: string | null
    tool_calls?: Array<{
      index?: number
      id?: string
      type?: 'function'
      function?: {
        name?: string
        arguments?: string
      }
    }>
  }
}

type OpenAIChatResponse = {
  id?: string
  choices?: OpenAIChatChoice[]
  usage?: {
    prompt_tokens?: number
    completion_tokens?: number
  }
}

type AnthropicStreamEvent =
  | {
      type: 'message_start'
      message: Record<string, unknown>
    }
  | {
      type: 'content_block_start'
      index: number
      content_block: Record<string, unknown>
    }
  | {
      type: 'content_block_delta'
      index: number
      delta: Record<string, unknown>
    }
  | {
      type: 'content_block_stop'
      index: number
    }
  | {
      type: 'message_delta'
      delta: {
        stop_reason: string | null
        stop_sequence: string | null
      }
      usage: {
        input_tokens?: number
        output_tokens: number
      }
    }
  | {
      type: 'message_stop'
    }

type TranslatorUsage = {
  prompt_tokens?: number
  completion_tokens?: number
}

type TranslatorBlockState = {
  index: number
  kind: 'text' | 'tool_use'
  closed: boolean
}

type TranslatorToolState = {
  block: TranslatorBlockState
  id: string
  index: number
  jsonBuffer: string
  name: string
}

const DEFAULT_OPENAI_BASE_URL = 'https://api.openai.com/v1'

/**
 * Determine whether the current session should route Anthropic SDK requests
 * through the OpenAI-compatible adapter.
 *
 * We support both an explicit provider flag and a lightweight inference mode.
 * The inference path keeps local testing convenient: if a user only supplies
 * `OPENAI_BASE_URL` + `OPENAI_API_KEY`, the adapter still activates without
 * requiring one extra environment variable.
 */
export function isOpenAIProviderEnabled(): boolean {
  return (
    isTruthy(process.env.CLAUDE_CODE_USE_OPENAI) ||
    (!!process.env.OPENAI_BASE_URL && !!process.env.OPENAI_API_KEY)
  )
}

/**
 * Resolve the base URL for the OpenAI-compatible endpoint.
 *
 * We intentionally fall back to `ANTHROPIC_BASE_URL` so an existing custom
 * settings file can be reused during migration by only adding the provider
 * flag, without forcing the user to rename every key up front.
 */
export function getOpenAIBaseURL(): string {
  return (
    process.env.ANTHROPIC_BASE_URL ||
    process.env.OPENAI_BASE_URL ||
    DEFAULT_OPENAI_BASE_URL
  )
}

/**
 * Resolve the credential that should be sent as `Authorization: Bearer ...`.
 *
 * We first prefer OpenAI-specific naming, then fall back to the Anthropic-style
 * variables that the user was already experimenting with. This makes the new
 * provider usable with both a clean OpenAI config and a minimally adjusted
 * legacy config file.
 */
export function getOpenAIAPIKey(): string | undefined {
  return (
    process.env.ANTHROPIC_AUTH_TOKEN ||
    process.env.ANTHROPIC_API_KEY ||
    process.env.OPENAI_API_KEY
  )
}

/**
 * 将当前仓库内部沿用的 Anthropic Messages 请求，翻译成 OpenAI-compatible
 * `chat/completions` 请求。
 *
 * 这里刻意只覆盖这条工程链路真正会用到的能力：
 * 1. system prompt
 * 2. 用户 / 助手消息
 * 3. tools 与 tool_choice
 * 4. tool_use / tool_result
 * 5. 用户消息里的图片块
 *
 * 这样可以让上层继续维持 Anthropic 风格的数据结构，而兼容层只在边界做协议转换。
 * 对于不认识的内容块，我们依旧会退化成文本，而不是直接抛错，避免普通编码会话被少量
 * 非核心富文本内容打断。
 */
export function buildOpenAIChatRequest(
  request: AnthropicMessageCreateBody,
): OpenAIChatRequest {
  const messages: OpenAIChatMessage[] = []
  const systemText = normalizeSystemPrompt(request.system)

  if (systemText) {
    messages.push({
      role: 'system',
      content: systemText,
    })
  }

  for (const message of request.messages) {
    if (message.role === 'assistant') {
      messages.push(...mapAssistantMessageToOpenAI(message.content))
      continue
    }
    messages.push(...mapUserMessageToOpenAI(message.content))
  }

  return {
    model: request.model,
    messages,
    ...(request.max_tokens !== undefined && {
      max_tokens: request.max_tokens,
    }),
    ...(request.stop_sequences?.length ? { stop: request.stop_sequences } : {}),
    ...(request.temperature !== undefined && {
      temperature: request.temperature,
    }),
    ...(request.top_p !== undefined && { top_p: request.top_p }),
    ...(request.tools?.length && {
      tools: request.tools.map(tool => ({
        type: 'function' as const,
        function: {
          name: tool.name,
          ...(tool.description ? { description: tool.description } : {}),
          ...(tool.input_schema ? { parameters: tool.input_schema } : {}),
        },
      })),
    }),
    ...(request.tool_choice && {
      tool_choice: mapAnthropicToolChoiceToOpenAI(request.tool_choice),
    }),
    ...(request.stream
      ? {
          stream: true,
          stream_options: {
            include_usage: true,
          },
        }
      : {}),
  }
}

/**
 * Convert a non-streaming OpenAI chat completion response into the Anthropic
 * message shape that the rest of this codebase already knows how to consume.
 */
export function mapOpenAIResponseToAnthropicMessage(
  response: OpenAIChatResponse,
  requestedModel: string,
): Record<string, unknown> {
  const choice = response.choices?.[0]
  const message = choice?.message ?? {}
  const contentBlocks: Array<Record<string, unknown>> = []
  const textContent = normalizeOpenAIContentToText(message.content)

  if (textContent) {
    contentBlocks.push({
      type: 'text',
      text: textContent,
    })
  }

  for (const toolCall of message.tool_calls ?? []) {
    contentBlocks.push({
      type: 'tool_use',
      id: toolCall.id,
      name: toolCall.function.name,
      input: safeJsonParse(toolCall.function.arguments),
    })
  }

  return {
    id: response.id || `openai-${Date.now()}`,
    type: 'message',
    role: 'assistant',
    model: requestedModel,
    content: contentBlocks,
    stop_reason: mapFinishReason(choice?.finish_reason),
    stop_sequence: null,
    usage: {
      input_tokens: response.usage?.prompt_tokens ?? 0,
      output_tokens: response.usage?.completion_tokens ?? 0,
    },
  }
}

/**
 * Stateful translator that turns OpenAI streaming deltas into the SSE event
 * sequence emitted by the Anthropic Messages API.
 *
 * Anthropic's SDK builds the final assistant message by replaying a stream of
 * `message_start` / `content_block_*` / `message_delta` / `message_stop`
 * events. Instead of rewriting the whole upper stack, we synthesize that same
 * event stream here so the rest of the CLI remains unchanged.
 */
export class OpenAIStreamTranslator {
  private readonly messageId: string
  private readonly requestedModel: string
  private readonly blocks: TranslatorBlockState[] = []
  private readonly toolStates = new Map<number, TranslatorToolState>()
  private activeTextBlock: TranslatorBlockState | null = null
  private emittedMessageStart = false
  private latestUsage: TranslatorUsage = {}
  private finishReason: string | null = null

  constructor(requestedModel: string, messageId?: string) {
    this.requestedModel = requestedModel
    this.messageId = messageId || `openai-stream-${Date.now()}`
  }

  /**
   * Feed one OpenAI streaming payload into the translator and get back the
   * Anthropic-compatible events that should be emitted immediately.
   */
  pushChunk(chunk: OpenAIChatResponse): AnthropicStreamEvent[] {
    const events: AnthropicStreamEvent[] = []

    if (!this.emittedMessageStart) {
      events.push(this.createMessageStartEvent(chunk.id))
      this.emittedMessageStart = true
    }

    const choice = chunk.choices?.[0]
    const delta = choice?.delta

    if (chunk.usage) {
      this.latestUsage = chunk.usage
    }

    if (delta?.content) {
      const block = this.ensureTextBlock(events)
      events.push({
        type: 'content_block_delta',
        index: block.index,
        delta: {
          type: 'text_delta',
          text: delta.content,
        },
      })
    }

    for (const toolCall of delta?.tool_calls ?? []) {
      const toolState = this.ensureToolState(toolCall, events)
      const partialArguments = toolCall.function?.arguments ?? ''

      if (partialArguments) {
        toolState.jsonBuffer += partialArguments
        events.push({
          type: 'content_block_delta',
          index: toolState.block.index,
          delta: {
            type: 'input_json_delta',
            partial_json: partialArguments,
          },
        })
      }
    }

    if (choice?.finish_reason) {
      this.finishReason = choice.finish_reason
    }

    return events
  }

  /**
   * Finish the translation and emit the closing Anthropic events in the exact
   * order expected by the upstream SDK.
   */
  finish(usageOverride?: TranslatorUsage): AnthropicStreamEvent[] {
    const events: AnthropicStreamEvent[] = []
    const usage = usageOverride || this.latestUsage

    for (const block of this.blocks) {
      if (!block.closed) {
        events.push({
          type: 'content_block_stop',
          index: block.index,
        })
        block.closed = true
      }
    }

    events.push({
      type: 'message_delta',
      delta: {
        stop_reason: mapFinishReason(this.finishReason),
        stop_sequence: null,
      },
      usage: {
        ...(usage.prompt_tokens !== undefined && {
          input_tokens: usage.prompt_tokens,
        }),
        output_tokens: usage.completion_tokens ?? 0,
      },
    })
    events.push({
      type: 'message_stop',
    })

    return events
  }

  /**
   * Create the initial assistant message snapshot that anchors the Anthropic
   * event stream.
   */
  private createMessageStartEvent(
    messageId: string | undefined,
  ): AnthropicStreamEvent {
    return {
      type: 'message_start',
      message: {
        id: messageId || this.messageId,
        type: 'message',
        role: 'assistant',
        model: this.requestedModel,
        content: [],
        stop_reason: null,
        stop_sequence: null,
        usage: {
          input_tokens: 0,
          output_tokens: 0,
        },
      },
    }
  }

  /**
   * Open a text content block lazily, only when the first text delta arrives.
   */
  private ensureTextBlock(
    events: AnthropicStreamEvent[],
  ): TranslatorBlockState {
    if (this.activeTextBlock && !this.activeTextBlock.closed) {
      return this.activeTextBlock
    }

    const block: TranslatorBlockState = {
      index: this.blocks.length,
      kind: 'text',
      closed: false,
    }

    this.blocks.push(block)
    this.activeTextBlock = block
    events.push({
      type: 'content_block_start',
      index: block.index,
      content_block: {
        type: 'text',
        text: '',
      },
    })

    return block
  }

  /**
   * Open or update a tool-use block for a streamed OpenAI tool call delta.
   */
  private ensureToolState(
    toolCall: NonNullable<
      NonNullable<OpenAIChatChoice['delta']>['tool_calls']
    >[number],
    events: AnthropicStreamEvent[],
  ): TranslatorToolState {
    const toolIndex = toolCall.index ?? 0
    const existing = this.toolStates.get(toolIndex)

    if (existing) {
      if (toolCall.id) existing.id = toolCall.id
      if (toolCall.function?.name) existing.name = toolCall.function.name
      return existing
    }

    const block: TranslatorBlockState = {
      index: this.blocks.length,
      kind: 'tool_use',
      closed: false,
    }
    const state: TranslatorToolState = {
      block,
      id: toolCall.id || `tool-call-${toolIndex}`,
      index: toolIndex,
      jsonBuffer: '',
      name: toolCall.function?.name || 'unknown',
    }

    this.blocks.push(block)
    this.toolStates.set(toolIndex, state)
    this.activeTextBlock = null
    events.push({
      type: 'content_block_start',
      index: block.index,
      content_block: {
        type: 'tool_use',
        id: state.id,
        name: state.name,
        input: {},
      },
    })

    return state
  }
}

/**
 * Wrap the existing fetch pipeline with an Anthropic-to-OpenAI protocol
 * adapter.
 *
 * The Anthropic SDK still believes it is talking to `/v1/messages`, but the
 * adapter intercepts those requests, translates them to `chat/completions`,
 * and rewrites the response back into Anthropic's response shapes.
 */
export function createOpenAICompatibleFetch(
  fetchOverride?: ClientOptions['fetch'],
): ClientOptions['fetch'] {
  return async (input, init) => {
    const url = input instanceof Request ? input.url : String(input)
    const pathname = safeGetPathname(url)

    if (
      pathname !== '/v1/messages' &&
      pathname !== '/v1/messages/count_tokens'
    ) {
      const innerFetch = fetchOverride ?? globalThis.fetch
      return innerFetch(input, init)
    }

    const request = await parseAnthropicRequestBody(input, init)

    if (pathname === '/v1/messages/count_tokens') {
      return createJsonResponse({
        input_tokens: estimateInputTokens(request),
      })
    }

    const openAIRequest = buildOpenAIChatRequest(request)
    const response = await dispatchOpenAIRequest(
      openAIRequest,
      init,
      fetchOverride,
    )

    if (!response.ok) {
      return rewriteOpenAIErrorResponse(response)
    }

    if (openAIRequest.stream) {
      return transformOpenAIStreamResponse(response, request.model)
    }

    const payload = (await response.json()) as OpenAIChatResponse
    return createJsonResponse(
      mapOpenAIResponseToAnthropicMessage(payload, request.model),
      response,
    )
  }
}

/**
 * Send a translated request to the OpenAI-compatible endpoint.
 */
async function dispatchOpenAIRequest(
  request: OpenAIChatRequest,
  init: RequestInit | undefined,
  fetchOverride?: ClientOptions['fetch'],
): Promise<Response> {
  const fetchImpl = fetchOverride ?? globalThis.fetch
  const headers = new Headers(init?.headers)
  const apiKey = getOpenAIAPIKey()

  headers.set('Content-Type', 'application/json')
  headers.delete('anthropic-version')
  headers.delete('anthropic-beta')
  headers.delete('x-api-key')

  if (apiKey && !headers.has('Authorization')) {
    headers.set('Authorization', `Bearer ${apiKey}`)
  }

  const targetUrl = buildOpenAIChatCompletionsURL(getOpenAIBaseURL())
  logForDebugging(
    `[OpenAI compat] dispatch ${targetUrl} auth=${headers.has('Authorization')} keySource=${apiKey ? 'present' : 'missing'}`,
  )

  return fetchImpl(targetUrl, {
    ...init,
    method: 'POST',
    headers,
    body: JSON.stringify(request),
  })
}

/**
 * Rewrite an OpenAI-style error payload into Anthropic's error envelope so the
 * existing SDK error parsing remains readable.
 */
async function rewriteOpenAIErrorResponse(
  response: Response,
): Promise<Response> {
  let message = response.statusText || 'OpenAI-compatible request failed'
  let type = 'invalid_request_error'

  try {
    const payload = await response.json()
    const error = (payload as { error?: { message?: string; type?: string } })
      .error
    if (error?.message) {
      message = error.message
    }
    if (error?.type) {
      type = error.type
    }
  } catch {
    // Fall back to the HTTP status text when the upstream body is not JSON.
  }

  return new Response(
    JSON.stringify({
      type: 'error',
      error: {
        type,
        message,
      },
    }),
    {
      status: response.status,
      headers: cloneResponseHeaders(response, 'application/json'),
    },
  )
}

/**
 * Convert an OpenAI streaming response into Anthropic-style SSE events.
 */
async function transformOpenAIStreamResponse(
  response: Response,
  requestedModel: string,
): Promise<Response> {
  const translator = new OpenAIStreamTranslator(requestedModel)
  const encoder = new TextEncoder()
  const stream = new ReadableStream<Uint8Array>({
    async start(controller) {
      try {
        for await (const payload of iterateOpenAIStreamPayloads(
          response.body,
        )) {
          for (const event of translator.pushChunk(payload)) {
            controller.enqueue(encoder.encode(toAnthropicSSE(event)))
          }
        }
        for (const event of translator.finish()) {
          controller.enqueue(encoder.encode(toAnthropicSSE(event)))
        }
        controller.close()
      } catch (error) {
        controller.error(error)
      }
    },
  })

  return new Response(stream, {
    status: response.status,
    headers: cloneResponseHeaders(response, 'text/event-stream'),
  })
}

/**
 * Read an Anthropic request body out of the SDK-generated fetch invocation.
 */
async function parseAnthropicRequestBody(
  input: RequestInfo | URL,
  init: RequestInit | undefined,
): Promise<AnthropicMessageCreateBody> {
  const rawBody =
    typeof init?.body === 'string'
      ? init.body
      : input instanceof Request
        ? await input.clone().text()
        : ''

  return JSON.parse(rawBody) as AnthropicMessageCreateBody
}

/**
 * Iterate over parsed JSON payloads from an OpenAI SSE stream.
 */
async function* iterateOpenAIStreamPayloads(
  body: ReadableStream<Uint8Array> | null,
): AsyncGenerator<OpenAIChatResponse> {
  if (!body) {
    return
  }

  const reader = body.getReader()
  const decoder = new TextDecoder()
  let buffer = ''

  while (true) {
    const { done, value } = await reader.read()
    if (done) {
      break
    }

    buffer += decoder.decode(value, { stream: true })

    while (true) {
      const boundaryIndex = buffer.indexOf('\n\n')
      if (boundaryIndex === -1) {
        break
      }

      const rawEvent = buffer.slice(0, boundaryIndex)
      buffer = buffer.slice(boundaryIndex + 2)
      const payload = parseOpenAISSEEvent(rawEvent)

      if (payload === '[DONE]') {
        return
      }
      if (payload) {
        yield JSON.parse(payload) as OpenAIChatResponse
      }
    }
  }

  const trailingPayload = parseOpenAISSEEvent(buffer)
  if (trailingPayload && trailingPayload !== '[DONE]') {
    yield JSON.parse(trailingPayload) as OpenAIChatResponse
  }
}

/**
 * Extract the joined `data:` payload from one SSE event block.
 */
function parseOpenAISSEEvent(rawEvent: string): string | null {
  const dataLines = rawEvent
    .split(/\r?\n/)
    .filter(line => line.startsWith('data:'))
    .map(line => line.slice(5).trim())

  if (dataLines.length === 0) {
    return null
  }

  return dataLines.join('\n')
}

/**
 * Serialize one Anthropic stream event back into SSE wire format.
 */
function toAnthropicSSE(event: AnthropicStreamEvent): string {
  return `event: ${event.type}\ndata: ${JSON.stringify(event)}\n\n`
}

/**
 * Map Anthropic's top-level system prompt into one plain system string.
 */
function normalizeSystemPrompt(
  system: AnthropicMessageCreateBody['system'],
): string {
  if (!system) {
    return ''
  }
  if (typeof system === 'string') {
    return system
  }

  return system
    .map(block => {
      if (block.type === 'text' && typeof block.text === 'string') {
        return block.text
      }
      return ''
    })
    .filter(Boolean)
    .join('\n\n')
}

/**
 * Translate one Anthropic assistant message into one OpenAI assistant message.
 */
function mapAssistantMessageToOpenAI(
  content: AnthropicMessageCreateBody['messages'][number]['content'],
): OpenAIChatMessage[] {
  if (typeof content === 'string') {
    return [
      {
        role: 'assistant',
        content,
      },
    ]
  }

  const textParts: string[] = []
  const toolCalls: OpenAIToolCall[] = []

  for (const block of content) {
    if (block.type === 'text' && typeof block.text === 'string') {
      textParts.push(block.text)
      continue
    }
    if (block.type === 'tool_use') {
      toolCalls.push({
        id: String(block.id),
        type: 'function',
        function: {
          name: String(block.name),
          arguments: JSON.stringify(block.input ?? {}),
        },
      })
      continue
    }
  }

  return [
    {
      role: 'assistant',
      ...(textParts.length
        ? { content: textParts.join('\n') }
        : { content: '' }),
      ...(toolCalls.length ? { tool_calls: toolCalls } : {}),
    },
  ]
}

/**
 * 把 Anthropic 风格的 user message 翻译成 OpenAI 的 `user` / `tool` 消息序列。
 *
 * 这里最关键的约束是“顺序不能乱”：
 * - 普通文本和图片要保留原始出现顺序
 * - 一旦遇到 `tool_result`，前面缓冲的用户内容必须先 flush 成一个 user message
 * - `tool_result` 自己要单独变成 OpenAI 的 `tool` role
 *
 * 这样模型在 OpenAI 侧看到的上下文顺序，才能和 Anthropic 侧原始语义保持一致。
 */
function mapUserMessageToOpenAI(
  content: AnthropicMessageCreateBody['messages'][number]['content'],
): OpenAIChatMessage[] {
  if (typeof content === 'string') {
    return [
      {
        role: 'user',
        content,
      },
    ]
  }

  const output: OpenAIChatMessage[] = []
  let bufferedUserContent: OpenAIUserContentPart[] = []

  const flushUserText = () => {
    if (bufferedUserContent.length === 0) {
      return
    }
    output.push({
      role: 'user',
      content: normalizeUserContentForOpenAI(bufferedUserContent),
    })
    bufferedUserContent = []
  }

  for (const block of content) {
    if (block.type === 'tool_result') {
      flushUserText()
      output.push({
        role: 'tool',
        tool_call_id: String(block.tool_use_id),
        content: normalizeToolResultContent(block.content),
      })
      continue
    }

    bufferedUserContent.push(...normalizeBlockToOpenAIUserContent(block))
  }

  flushUserText()
  return output
}

/**
 * 将用户消息里的 Anthropic 内容块翻译成 OpenAI 可接受的多模态内容块。
 *
 * 目前重点支持两类：
 * 1. `text` -> OpenAI `text`
 * 2. `image` -> OpenAI `image_url`
 *
 * 其它未知块仍然降级成文本，目的是让会话继续跑，而不是因为某个边缘块类型整轮失败。
 */
function normalizeBlockToOpenAIUserContent(
  block: Record<string, unknown>,
): OpenAIUserContentPart[] {
  if (block.type === 'text' && typeof block.text === 'string') {
    return [
      {
        type: 'text',
        text: block.text,
      },
    ]
  }
  if (block.type === 'image') {
    const imageUrl = normalizeImageSourceToOpenAIUrl(block.source)
    if (imageUrl) {
      return [
        {
          type: 'image_url',
          image_url: {
            url: imageUrl,
          },
        },
      ]
    }
    return [
      {
        type: 'text',
        text: '[image omitted by OpenAI compatibility adapter]',
      },
    ]
  }
  return [
    {
      type: 'text',
      text: JSON.stringify(block),
    },
  ]
}

/**
 * 根据 Anthropic 的图片 source 结构，产出 OpenAI `image_url.url` 所需的字符串。
 *
 * 两种最常见来源分别处理：
 * - `base64`：拼成 data URL，避免再依赖外部文件服务
 * - `url`：直接透传远程地址
 *
 * 其它 source 类型先返回 `null`，由上层回退成占位文本，保证兼容层行为可预测。
 */
function normalizeImageSourceToOpenAIUrl(source: unknown): string | null {
  if (!source || typeof source !== 'object') {
    return null
  }

  if (
    'type' in source &&
    source.type === 'base64' &&
    'media_type' in source &&
    typeof source.media_type === 'string' &&
    'data' in source &&
    typeof source.data === 'string'
  ) {
    return `data:${source.media_type};base64,${source.data}`
  }

  if (
    'type' in source &&
    source.type === 'url' &&
    'url' in source &&
    typeof source.url === 'string'
  ) {
    return source.url
  }

  return null
}

/**
 * 把缓冲中的用户内容块压缩成 OpenAI message.content。
 *
 * 这里保留一个小优化：
 * - 如果整段都只是文本，就继续输出字符串，兼容已有文本路径
 * - 只要出现图片，就输出多模态数组，保留文本与图片的交错顺序
 */
function normalizeUserContentForOpenAI(
  content: OpenAIUserContentPart[],
): string | OpenAIUserContentPart[] {
  if (content.every(part => part.type === 'text')) {
    return content.map(part => part.text).join('\n')
  }
  return content
}

/**
 * 把 Anthropic 的 tool_choice 语义映射到 OpenAI-compatible `tool_choice`。
 *
 * 语义对应关系如下：
 * - `auto` -> `auto`
 * - `none` -> `none`
 * - `any` -> `required`
 * - `tool(name)` -> `function(name)`
 *
 * 这样上层继续传 Anthropic 风格的“工具选择策略”，兼容层负责把约束翻译成
 * OpenAI 侧能理解的形状。
 */
function mapAnthropicToolChoiceToOpenAI(
  toolChoice: NonNullable<AnthropicMessageCreateBody['tool_choice']>,
): NonNullable<OpenAIChatRequest['tool_choice']> {
  switch (toolChoice.type) {
    case 'auto':
      return 'auto'
    case 'none':
      return 'none'
    case 'any':
      return 'required'
    case 'tool':
      return {
        type: 'function',
        function: {
          name: toolChoice.name,
        },
      }
  }
}

/**
 * Turn tool result payloads into a string payload accepted by OpenAI's `tool`
 * role message.
 */
function normalizeToolResultContent(content: unknown): string {
  if (typeof content === 'string') {
    return content
  }
  if (Array.isArray(content)) {
    return content
      .map(item => {
        if (typeof item === 'string') {
          return item
        }
        if (
          typeof item === 'object' &&
          item !== null &&
          'type' in item &&
          item.type === 'text' &&
          typeof item.text === 'string'
        ) {
          return item.text
        }
        return JSON.stringify(item)
      })
      .join('\n')
  }
  return JSON.stringify(content)
}

/**
 * Flatten OpenAI's `message.content` field to plain text.
 */
function normalizeOpenAIContentToText(
  content: string | Array<Record<string, unknown>> | null | undefined,
): string {
  if (!content) {
    return ''
  }
  if (typeof content === 'string') {
    return content
  }

  return content
    .map(part => {
      if (typeof part.text === 'string') {
        return part.text
      }
      return ''
    })
    .filter(Boolean)
    .join('\n')
}

/**
 * Map OpenAI finish reasons to the Anthropic stop_reason values used by the
 * rest of the CLI.
 */
function mapFinishReason(reason: string | null | undefined): string | null {
  switch (reason) {
    case 'length':
      return 'max_tokens'
    case 'tool_calls':
      return 'tool_use'
    case 'stop':
    case 'content_filter':
      return 'end_turn'
    default:
      return 'end_turn'
  }
}

/**
 * Parse a JSON string without throwing. Tool-call argument deltas are often
 * still streaming in flight, so we prefer a stable object fallback over a
 * hard failure.
 */
function safeJsonParse(raw: string): Record<string, unknown> {
  try {
    return JSON.parse(raw) as Record<string, unknown>
  } catch {
    return raw ? { raw } : {}
  }
}

/**
 * Estimate input tokens locally for `/count_tokens`.
 */
function estimateInputTokens(request: AnthropicMessageCreateBody): number {
  return Math.max(1, Math.round(JSON.stringify(request).length / 4))
}

/**
 * Build a JSON response while preserving useful upstream headers such as
 * request IDs whenever we have an originating response.
 */
function createJsonResponse(
  payload: unknown,
  sourceResponse?: Response,
): Response {
  return new Response(JSON.stringify(payload), {
    status: sourceResponse?.status ?? 200,
    headers: cloneResponseHeaders(sourceResponse, 'application/json'),
  })
}

/**
 * Clone response headers and normalize the content type for the rewritten body.
 */
function cloneResponseHeaders(
  response: Response | undefined,
  contentType: string,
): Headers {
  const headers = new Headers(response?.headers)
  headers.set('content-type', contentType)
  return headers
}

/**
 * Join a base URL and relative path without creating duplicate `/v1/v1` or
 * malformed double-slash paths.
 */
export function buildOpenAIChatCompletionsURL(baseUrl: string): string {
  const url = new URL(baseUrl)
  let pathname = url.pathname.replace(/\/+$/, '')

  if (!pathname) {
    pathname = '/v1'
  }

  if (pathname.endsWith('/chat/completions')) {
    url.pathname = pathname
    return url.toString()
  }

  if (!pathname.endsWith('/v1')) {
    pathname = `${pathname}/v1`
  }

  url.pathname = `${pathname}/chat/completions`
  return url.toString()
}

/**
 * Safely resolve a pathname from a fetch URL string.
 */
function safeGetPathname(url: string): string {
  try {
    return new URL(url).pathname
  } catch {
    return url
  }
}

/**
 * Minimal truthy parser shared by the compatibility layer so it does not need
 * to reach back into unrelated env helper modules.
 */
function isTruthy(value: string | undefined): boolean {
  if (!value) {
    return false
  }
  return ['1', 'true', 'yes', 'on'].includes(value.toLowerCase().trim())
}
