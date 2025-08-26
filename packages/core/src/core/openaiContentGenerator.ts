/**
 * @license
 * Copyright 2025 Google LLC
 * SPDX-License-Identifier: Apache-2.0
 */

import { OpenAI } from 'openai';
import { APIError } from 'openai';
import {
  GenerateContentParameters,
  GenerateContentResponse,
  CountTokensParameters,
  CountTokensResponse,
  EmbedContentParameters,
  EmbedContentResponse,
  Content,
  Part,
  ToolListUnion,
  FinishReason,
} from '@google/genai';
import { ContentGenerator } from './contentGenerator.js';
import { toContents } from '../code_assist/converter.js';

export class OpenAIContentGenerator implements ContentGenerator {
  private openai: OpenAI | undefined; // Use any to avoid import issues during build
  private model: string;
  private userId?: string;

  constructor(apiKey: string, model: string, proxy?: string, userId?: string) {
    // Dynamic import will be handled at runtime
    this.model = model;
    this.userId = userId;
    this.initializeOpenAI(apiKey, proxy);
  }

  private async initializeOpenAI(apiKey: string, proxy?: string) {
    try {
      const openaiModule = await import('openai');
      const OpenAI =
        openaiModule.default || openaiModule.OpenAI || openaiModule;
      
      const defaultHeaders: Record<string, string> = {};
      if (this.userId) {
        defaultHeaders['X-User-Id'] = this.userId;
      }
      
      this.openai = new OpenAI({
        apiKey,
        baseURL: proxy ? `${proxy}/v1` : undefined,
        defaultHeaders,
      });
    } catch (_error) {
      throw new Error(
        'Failed to load OpenAI module. Make sure openai package is installed.',
      );
    }
  }

  async generateContent(
    request: GenerateContentParameters,
    _userPromptId: string,
  ): Promise<GenerateContentResponse> {
    if (!this.openai) {
      throw new Error('OpenAI client is not initialized.');
    }
    try {
      const contents = toContents(request.contents);
      const messages = this.convertToOpenAIMessages(contents);
      const tools = this.convertToOpenAITools(request.config?.tools);

      const completion = await this.openai.chat.completions.create({
        model: this.model,
        messages,
        tools: tools.length > 0 ? tools : undefined,
        temperature: request.config?.temperature,
        max_tokens: request.config?.maxOutputTokens,
        top_p: request.config?.topP,
        stop: request.config?.stopSequences,
      });

      return this.convertToGenerateContentResponse(completion);
    } catch (error) {
      if (error instanceof APIError) {
        throw new Error(`OpenAI API Error: ${error.message} (${error.status})`);
      }
      if (error instanceof Error) {
        throw new Error(`OpenAI Error: ${error.message}`);
      }
      throw error;
    }
  }

  async generateContentStream(
    request: GenerateContentParameters,
    _userPromptId: string,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    if (!this.openai) {
      throw new Error('OpenAI client is not initialized.');
    }
    return async function* (this: OpenAIContentGenerator) {
      try {
        const contents = toContents(request.contents);
        const messages = this.convertToOpenAIMessages(contents);
        const tools = this.convertToOpenAITools(request.config?.tools);

        const stream = await this.openai!.chat.completions.create({
          model: this.model,
          messages,
          tools: tools.length > 0 ? tools : undefined,
          temperature: request.config?.temperature,
          max_tokens: request.config?.maxOutputTokens,
          top_p: request.config?.topP,
          stop: request.config?.stopSequences,
          stream: true,
        });

        let accumulatedContent = '';
        const accumulatedToolCalls: OpenAI.Chat.Completions.ChatCompletionMessageToolCall[] =
          [];

        let previousContentLength = 0;

        for await (const chunk of stream) {
          const delta = chunk.choices[0]?.delta;

          if (delta?.content) {
            accumulatedContent += delta.content;
            // Fix for OpenAI streaming: Only yield the new content delta, not the full accumulated content
            // This prevents the cascading repetition effect in the UI
            const newContent = accumulatedContent.substring(previousContentLength);
            if (newContent) {
              yield this.createStreamingResponse(newContent, false);
              previousContentLength = accumulatedContent.length;
            }
          }

          if (delta?.tool_calls) {
            // Handle tool calls in streaming
            for (const toolCall of delta.tool_calls) {
              if (!accumulatedToolCalls[toolCall.index]) {
                accumulatedToolCalls[toolCall.index] = {
                  id: toolCall.id!,
                  type: 'function',
                  function: { name: '', arguments: '' },
                };
              }

              if (toolCall.function?.name) {
                accumulatedToolCalls[toolCall.index].function.name +=
                  toolCall.function.name;
              }
              if (toolCall.function?.arguments) {
                accumulatedToolCalls[toolCall.index].function.arguments +=
                  toolCall.function.arguments;
              }
            }
          }

          if (chunk.choices[0]?.finish_reason) {
            // Final response with tool calls if any
            // Only yield if we have tool calls, since text content was already streamed incrementally
            if (accumulatedToolCalls.length > 0) {
              yield this.createStreamingResponse(
                accumulatedContent,
                true,
                accumulatedToolCalls,
              );
            }
          }
        }
      } catch (error) {
        if (error instanceof APIError) {
          throw new Error(
            `OpenAI API Error: ${error.message} (${error.status})`,
          );
        }
        if (error instanceof Error) {
          throw new Error(`OpenAI Error: ${error.message}`);
        }
        throw error;
      }
    }.bind(this)();
  }

  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    try {
      // Simple approximation: ~4 characters per token for English text
      // This is a rough estimate until we can properly integrate tiktoken
      let totalTokens = 0;
      const contents = toContents(request.contents);

      for (const content of contents) {
        if (content.parts) {
          for (const part of content.parts) {
            if (part.text) {
              totalTokens += Math.ceil(part.text.length / 4);
            }
          }
        }
      }

      return { totalTokens };
    } catch (error) {
      throw new Error(
        `Token counting error: ${
          error instanceof Error ? error.message : 'Unknown error'
        }`,
      );
    }
  }

  async embedContent(
    request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    if (!this.openai) {
      throw new Error('OpenAI client is not initialized.');
    }
    try {
      const contents = toContents(request.contents);
      const text = this.extractTextFromContents(contents);

      const response = await this.openai.embeddings.create({
        model: request.model || 'text-embedding-ada-002',
        input: text,
      });

      return {
        embeddings: [
          {
            values: response.data[0].embedding,
          },
        ],
      };
    } catch (error) {
      if (error instanceof APIError) {
        throw new Error(
          `OpenAI Embeddings API Error: ${error.message} (${error.status})`,
        );
      }
      if (error instanceof Error) {
        throw new Error(`OpenAI Embeddings Error: ${error.message}`);
      }
      throw error;
    }
  }

  private convertToOpenAIMessages(
    contents: Content[],
  ): OpenAI.Chat.Completions.ChatCompletionMessageParam[] {
    const messages: OpenAI.Chat.Completions.ChatCompletionMessageParam[] = [];

    for (const content of contents) {
      const role =
        content.role === 'user'
          ? 'user'
          : content.role === 'model'
            ? 'assistant'
            : 'system';

      if (
        content.parts &&
        content.parts.length === 1 &&
        content.parts[0].text
      ) {
        // Simple text message
        messages.push({
          role,
          content: content.parts[0].text,
        });
      } else if (content.parts) {
        // Complex message with multiple parts
        const contentParts: (
          | OpenAI.Chat.Completions.ChatCompletionContentPartText
          | OpenAI.Chat.Completions.ChatCompletionContentPartImage
        )[] = [];

        for (const part of content.parts) {
          if (part.text) {
            contentParts.push({
              type: 'text',
              text: part.text,
            });
          } else if (part.inlineData) {
            contentParts.push({
              type: 'image_url',
              image_url: {
                url: `data:${part.inlineData.mimeType};base64,${part.inlineData.data}`,
              },
            } as OpenAI.Chat.Completions.ChatCompletionContentPartImage);
          } else if (part.functionCall) {
            // Handle function calls
            if (role === 'assistant' && part.functionCall.name) {
              messages.push({
                role: 'assistant',
                content: null,
                tool_calls: [
                  {
                    id: `call_${Date.now()}`,
                    type: 'function',
                    function: {
                      name: part.functionCall.name,
                      arguments: JSON.stringify(part.functionCall.args || {}),
                    },
                  },
                ],
              });
              continue;
            }
          } else if (part.functionResponse) {
            // Handle function responses
            messages.push({
              role: 'tool',
              content: JSON.stringify(part.functionResponse.response),
              tool_call_id: `call_${Date.now()}`,
            });
            continue;
          }
        }

        if (contentParts.length > 0) {
          if (role === 'user') {
            messages.push({
              role: 'user',
              content: contentParts,
            });
          } else if (role === 'assistant') {
            // For assistant role, only allow text content to avoid type issues
            const textContent = contentParts
              .filter((part) => part.type === 'text')
              .map((part) => (part as OpenAI.Chat.Completions.ChatCompletionContentPartText).text)
              .join(' ');
            
            if (textContent) {
              messages.push({
                role: 'assistant',
                content: textContent,
              });
            }
          } else {
            messages.push({
              role: 'system',
              content: contentParts
                .map((part) => (part.type === 'text' ? (part as OpenAI.Chat.Completions.ChatCompletionContentPartText).text : ''))
                .join(' '),
            });
          }
        }
      }
    }

    return messages;
  }

  private convertToOpenAITools(
    tools?: ToolListUnion,
  ): OpenAI.Chat.Completions.ChatCompletionTool[] {
    if (!tools || !Array.isArray(tools)) return [];

    const result: OpenAI.Chat.Completions.ChatCompletionTool[] = [];

    for (const tool of tools) {
      if ('functionDeclarations' in tool && tool.functionDeclarations) {
        for (const func of tool.functionDeclarations) {
          if (func.name) {
            result.push({
              type: 'function' as const,
              function: {
                name: func.name,
                description: func.description,
                parameters: func.parameters as Record<string, unknown>,
              },
            });
          }
        }
      }
    }

    return result;
  }

  private convertToGenerateContentResponse(
    completion: OpenAI.Chat.Completions.ChatCompletion,
  ): GenerateContentResponse {
    const choice = completion.choices[0];
    const parts: Part[] = [];

    if (choice.message.content) {
      parts.push({ text: choice.message.content });
    }

    if (choice.message.tool_calls) {
      for (const toolCall of choice.message.tool_calls) {
        parts.push({
          functionCall: {
            name: toolCall.function.name,
            args: JSON.parse(toolCall.function.arguments || '{}'),
          },
        });
      }
    }

    const response = new GenerateContentResponse();
    response.candidates = [
      {
        content: {
          parts,
          role: 'model',
        },
        finishReason: this.mapFinishReason(
          choice.finish_reason,
        ) as FinishReason,
        index: choice.index,
      },
    ];
    response.usageMetadata = {
      promptTokenCount: completion.usage?.prompt_tokens || 0,
      candidatesTokenCount: completion.usage?.completion_tokens || 0,
      totalTokenCount: completion.usage?.total_tokens || 0,
    };

    return response;
  }

  private createStreamingResponse(
    content: string,
    isComplete: boolean,
    toolCalls?: OpenAI.Chat.Completions.ChatCompletionMessageToolCall[],
  ): GenerateContentResponse {
    const parts: Part[] = [];

    if (content) {
      parts.push({ text: content });
    }

    if (toolCalls) {
      for (const toolCall of toolCalls) {
        if (toolCall.function?.name && toolCall.function?.arguments) {
          try {
            parts.push({
              functionCall: {
                name: toolCall.function.name,
                args: JSON.parse(toolCall.function.arguments),
              },
            });
          } catch (_e) {
            // If arguments aren't complete JSON yet, skip for now
          }
        }
      }
    }

    const response = new GenerateContentResponse();
    response.candidates = [
      {
        content: {
          parts,
          role: 'model',
        },
        finishReason: isComplete ? ('STOP' as FinishReason) : undefined,
        index: 0,
      },
    ];

    return response;
  }

  private mapFinishReason(reason: string | null): FinishReason | undefined {
    switch (reason) {
      case 'stop':
        return 'STOP' as FinishReason;
      case 'length':
        return 'MAX_TOKENS' as FinishReason;
      case 'tool_calls':
        return 'STOP' as FinishReason;
      case 'content_filter':
        return 'SAFETY' as FinishReason;
      default:
        return 'OTHER' as FinishReason;
    }
  }

  private extractTextFromContents(contents: Content[]): string {
    return contents
      .flatMap((content) => content.parts || [])
      .filter((part): part is Part & { text: string } => !!part.text)
      .map((part) => part.text)
      .join(' ');
  }
}
