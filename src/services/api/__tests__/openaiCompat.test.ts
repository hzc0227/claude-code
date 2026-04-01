import { afterEach, describe, expect, test } from 'bun:test'

import { getAPIProvider } from '../../../utils/model/providers'
import {
  getDefaultMainLoopModel,
  getSmallFastModel,
} from '../../../utils/model/model'
import {
  buildOpenAIChatRequest,
  buildOpenAIChatCompletionsURL,
  getOpenAIAPIKey,
  getOpenAIBaseURL,
  mapOpenAIResponseToAnthropicMessage,
  OpenAIStreamTranslator,
} from '../openaiCompat'

describe('OpenAI provider detection', () => {
  afterEach(() => {
    delete process.env.CLAUDE_CODE_USE_OPENAI
    delete process.env.OPENAI_BASE_URL
    delete process.env.OPENAI_API_KEY
    delete process.env.OPENAI_MODEL
    delete process.env.OPENAI_SMALL_FAST_MODEL
    delete process.env.ANTHROPIC_BASE_URL
    delete process.env.ANTHROPIC_AUTH_TOKEN
  })

  test('prefers the explicit OpenAI provider flag', () => {
    process.env.CLAUDE_CODE_USE_OPENAI = '1'

    expect(getAPIProvider()).toBe('openai')
  })

  test('infers the OpenAI provider when OpenAI credentials are present', () => {
    process.env.OPENAI_BASE_URL = 'https://example.com/v1'
    process.env.OPENAI_API_KEY = 'sk-openai'

    expect(getAPIProvider()).toBe('openai')
  })

  test('prefers the active settings-compatible Anthropic env vars over stale global OpenAI env vars', () => {
    process.env.OPENAI_BASE_URL = 'https://stale.example.com/v1'
    process.env.OPENAI_API_KEY = 'stale-key'
    process.env.ANTHROPIC_BASE_URL = 'https://active.example.com'
    process.env.ANTHROPIC_AUTH_TOKEN = 'active-token'

    expect(getOpenAIBaseURL()).toBe('https://active.example.com')
    expect(getOpenAIAPIKey()).toBe('active-token')
  })
})

describe('OpenAI compatibility request mapping', () => {
  test('fills in the default /v1 prefix when the configured base URL is only a site root', () => {
    expect(buildOpenAIChatCompletionsURL('https://hzchub.asia')).toBe(
      'https://hzchub.asia/v1/chat/completions',
    )
    expect(buildOpenAIChatCompletionsURL('https://hzchub.asia/v1')).toBe(
      'https://hzchub.asia/v1/chat/completions',
    )
  })

  test('maps Anthropic-style messages, system prompt and tools into an OpenAI chat request', () => {
    const request = buildOpenAIChatRequest({
      model: 'glm-4.5',
      max_tokens: 512,
      system: '你是一个编码助手',
      messages: [
        {
          role: 'assistant',
          content: [
            { type: 'text', text: '我先查一下' },
            {
              type: 'tool_use',
              id: 'toolu_1',
              name: 'Read',
              input: { file_path: 'README.md' },
            },
          ],
        },
        {
          role: 'user',
          content: [
            {
              type: 'tool_result',
              tool_use_id: 'toolu_1',
              content: 'README 内容',
            },
            { type: 'text', text: '继续' },
          ],
        },
      ],
      tools: [
        {
          name: 'Read',
          description: '读取文件',
          input_schema: {
            type: 'object',
            properties: {
              file_path: { type: 'string' },
            },
            required: ['file_path'],
          },
        },
      ],
    })

    expect(request.model).toBe('glm-4.5')
    expect(request.messages[0]).toEqual({
      role: 'system',
      content: '你是一个编码助手',
    })
    expect(request.messages[1]).toEqual({
      role: 'assistant',
      content: '我先查一下',
      tool_calls: [
        {
          id: 'toolu_1',
          type: 'function',
          function: {
            name: 'Read',
            arguments: '{"file_path":"README.md"}',
          },
        },
      ],
    })
    expect(request.messages[2]).toEqual({
      role: 'tool',
      tool_call_id: 'toolu_1',
      content: 'README 内容',
    })
    expect(request.messages[3]).toEqual({
      role: 'user',
      content: '继续',
    })
    expect(request.tools).toEqual([
      {
        type: 'function',
        function: {
          name: 'Read',
          description: '读取文件',
          parameters: {
            type: 'object',
            properties: {
              file_path: { type: 'string' },
            },
            required: ['file_path'],
          },
        },
      },
    ])
  })

  test('maps user image blocks into OpenAI image_url parts and preserves text order', () => {
    const request = buildOpenAIChatRequest({
      model: 'gpt-5.4',
      messages: [
        {
          role: 'user',
          content: [
            { type: 'text', text: '先看第一张图' },
            {
              type: 'image',
              source: {
                type: 'base64',
                media_type: 'image/png',
                data: 'YWJjZA==',
              },
            },
            { type: 'text', text: '再看第二张图' },
            {
              type: 'image',
              source: {
                type: 'url',
                url: 'https://example.com/demo.png',
              },
            },
          ],
        },
      ],
    })

    expect(request.messages).toEqual([
      {
        role: 'user',
        content: [
          { type: 'text', text: '先看第一张图' },
          {
            type: 'image_url',
            image_url: {
              url: 'data:image/png;base64,YWJjZA==',
            },
          },
          { type: 'text', text: '再看第二张图' },
          {
            type: 'image_url',
            image_url: {
              url: 'https://example.com/demo.png',
            },
          },
        ],
      },
    ])
  })

  test('maps Anthropic tool_choice variants into OpenAI-compatible values', () => {
    expect(
      buildOpenAIChatRequest({
        model: 'gpt-5.4',
        messages: [{ role: 'user', content: 'hello' }],
        tool_choice: { type: 'auto' },
      }).tool_choice,
    ).toBe('auto')

    expect(
      buildOpenAIChatRequest({
        model: 'gpt-5.4',
        messages: [{ role: 'user', content: 'hello' }],
        tool_choice: { type: 'any' },
      }).tool_choice,
    ).toBe('required')

    expect(
      buildOpenAIChatRequest({
        model: 'gpt-5.4',
        messages: [{ role: 'user', content: 'hello' }],
        tool_choice: { type: 'none' },
      }).tool_choice,
    ).toBe('none')

    expect(
      buildOpenAIChatRequest({
        model: 'gpt-5.4',
        messages: [{ role: 'user', content: 'hello' }],
        tool_choice: { type: 'tool', name: 'Read' },
      }).tool_choice,
    ).toEqual({
      type: 'function',
      function: {
        name: 'Read',
      },
    })
  })
})

describe('OpenAI compatibility response mapping', () => {
  test('maps OpenAI chat responses into Anthropic-style assistant messages', () => {
    const message = mapOpenAIResponseToAnthropicMessage(
      {
        id: 'chatcmpl-1',
        choices: [
          {
            index: 0,
            finish_reason: 'tool_calls',
            message: {
              role: 'assistant',
              content: '我先读一下文件',
              tool_calls: [
                {
                  id: 'call_1',
                  type: 'function',
                  function: {
                    name: 'Read',
                    arguments: '{"file_path":"README.md"}',
                  },
                },
              ],
            },
          },
        ],
        usage: {
          prompt_tokens: 11,
          completion_tokens: 7,
        },
      },
      'glm-4.5',
    )

    expect(message.id).toBe('chatcmpl-1')
    expect(message.model).toBe('glm-4.5')
    expect(message.stop_reason).toBe('tool_use')
    expect(message.usage).toEqual({
      input_tokens: 11,
      output_tokens: 7,
    })
    expect(message.content).toEqual([
      { type: 'text', text: '我先读一下文件' },
      {
        type: 'tool_use',
        id: 'call_1',
        name: 'Read',
        input: { file_path: 'README.md' },
      },
    ])
  })
})

describe('OpenAI stream translation', () => {
  test('turns OpenAI streaming chunks into Anthropic-compatible SSE events', () => {
    const translator = new OpenAIStreamTranslator('glm-4.5')

    const firstEvents = translator.pushChunk({
      id: 'chatcmpl-stream',
      choices: [
        {
          index: 0,
          delta: {
            role: 'assistant',
            content: '你好',
          },
        },
      ],
    })

    expect(firstEvents.map(event => event.type)).toEqual([
      'message_start',
      'content_block_start',
      'content_block_delta',
    ])

    const secondEvents = translator.pushChunk({
      id: 'chatcmpl-stream',
      choices: [
        {
          index: 0,
          delta: {
            content: '，世界',
          },
          finish_reason: 'stop',
        },
      ],
      usage: {
        prompt_tokens: 9,
        completion_tokens: 3,
      },
    })

    const finalEvents = translator.finish({
      prompt_tokens: 9,
      completion_tokens: 3,
    })

    expect(secondEvents.map(event => event.type)).toEqual([
      'content_block_delta',
    ])
    expect(finalEvents.map(event => event.type)).toEqual([
      'content_block_stop',
      'message_delta',
      'message_stop',
    ])
    expect(finalEvents[1]).toMatchObject({
      type: 'message_delta',
      delta: {
        stop_reason: 'end_turn',
      },
      usage: {
        input_tokens: 9,
        output_tokens: 3,
      },
    })
  })
})

describe('OpenAI provider model defaults', () => {
  afterEach(() => {
    delete process.env.CLAUDE_CODE_USE_OPENAI
    delete process.env.OPENAI_MODEL
    delete process.env.OPENAI_SMALL_FAST_MODEL
  })

  test('uses GPT-5.4 as the default main model and GPT-5 mini as the small fast model', () => {
    process.env.CLAUDE_CODE_USE_OPENAI = '1'

    expect(getDefaultMainLoopModel()).toBe('gpt-5.4')
    expect(getSmallFastModel()).toBe('gpt-5-mini')
  })
})
