# Technical Specification: OpenAI Provider Integration

This document outlines the architecture and integration approach for adding an OpenAI Large Language Model (LLM) provider to the `gemini-cli` project.

## 1. OpenAI Provider Class Design

A new `OpenAIContentGenerator` class will be created to handle interactions with the OpenAI API. This class will implement the existing `ContentGenerator` interface to ensure seamless integration with the existing framework.

### 1.1. Class Structure

The `OpenAIContentGenerator` class will be located at `packages/core/src/core/openaiContentGenerator.ts`.

```typescript
// packages/core/src/core/openaiContentGenerator.ts

import { OpenAI } from 'openai';
import {
  GenerateContentParameters,
  GenerateContentResponse,
  CountTokensParameters,
  CountTokensResponse,
  EmbedContentParameters,
  EmbedContentResponse,
} from './contentGenerator'; // Simplified - will need to be adapted
import { ContentGenerator } from './contentGenerator';
import { Tiktoken } from 'tiktoken/lite';
import { cl100k_base } from 'tiktoken/ranks/cl100k_base';

export class OpenAIContentGenerator implements ContentGenerator {
  private openai: OpenAI;
  private model: string;
  private tokenizer: Tiktoken;

  constructor(apiKey: string, model: string, proxy?: string) {
    this.openai = new OpenAI({
      apiKey,
      // Support for proxy if provided
      // baseURL: proxy ? `${proxy}/v1` : 'https://api.openai.com/v1',
    });
    this.model = model;
    // Use tiktoken for token counting
    const tokenizerModel = cl100k_base;
    this.tokenizer = new Tiktoken(
      tokenizerModel.bpe_ranks,
      tokenizerModel.special_tokens,
      tokenizerModel.pat_str,
    );
  }

  async generateContent(
    request: GenerateContentParameters,
    userPromptId: string, // This parameter may not be used by OpenAI but is part of the interface
  ): Promise<GenerateContentResponse> {
    // Implementation to call OpenAI's chat completions API
    // and adapt the response to the GenerateContentResponse format.
  }

  async generateContentStream(
    request: GenerateContentParameters,
    userPromptId: string,
  ): Promise<AsyncGenerator<GenerateContentResponse>> {
    // Implementation to handle streaming responses from OpenAI API.
  }

  async countTokens(
    request: CountTokensParameters,
  ): Promise<CountTokensResponse> {
    // use this.tokenizer to count tokens
  }

  async embedContent(
    request: EmbedContentParameters,
  ): Promise<EmbedContentResponse> {
    // Implementation to call OpenAI's embeddings API.
  }
}
```

### 1.2. Error Handling and Retry Logic

The implementation will include robust error handling for common OpenAI API errors (e.g., 401, 429, 500). A retry mechanism with exponential backoff will be implemented for transient errors like rate limiting (429) and server-side issues (5xx).

## 2. Configuration Architecture

### 2.1. New AuthType

A new `AuthType` enum member will be added for OpenAI.

```typescript
// packages/core/src/core/contentGenerator.ts
export enum AuthType {
  // ... existing auth types
  USE_OPENAI = 'openai-api-key',
}
```

### 2.2. Configuration Parameters

The following configuration parameters will be needed:

- **OpenAI Configuration Block:** A new section within the `gemini-cli` configuration file will be dedicated to OpenAI settings.

```json
// Example in ~/.gemini/config.json
{
  "openai": {
    "apiKey": "YOUR_OPENAI_API_KEY",
    "model": "gpt-4-turbo",
    "gatewayUrl": "https://api.openai.com/v1",
    "modelVersion": "2024-05-13"
  }
}
```

### 2.3. Configuration File Handling

The `Config` class will be updated to read the `openai` block from the configuration file. The `createContentGeneratorConfig` function will receive these values from the `Config` instance, removing the need to read environment variables for OpenAI.

## 3. Integration Points

### 3.1. `createContentGenerator()` Factory

The `createContentGenerator` factory function will be extended to handle the new `USE_OPENAI` `authType`.

```typescript
// packages/core/src/core/contentGenerator.ts in createContentGenerator function
// ...
if (config.authType === AuthType.USE_OPENAI) {
  if (!config.apiKey) {
    throw new Error('OPENAI_API_KEY is not set.');
  }
  return new LoggingContentGenerator(
    new OpenAIContentGenerator(config.apiKey, config.model, config.proxy),
    gcConfig,
  );
}
// ...
```

### 3.2. `Config` Class

The `Config` class will be updated to handle the new `authType` and related parameters. No major structural changes are expected, but it will need to correctly pass the `authType` to `createContentGeneratorConfig`.

### 3.3. CLI Argument Additions

The CLI in `packages/cli` will be updated to:

- Support the new `--auth-type openai-api-key` option.
- Provide appropriate validation and help messages for the new options.
- The `--model` argument will be adapted to accept OpenAI model names.

## 4. Model Support Strategy

### 4.1. Initially Supported Models

- `gpt-4`
- `gpt-4-turbo`
- `gpt-3.5-turbo`

### 4.2. Model-Specific Parameters

The implementation will be designed to be extensible for model-specific parameters in the future, but the initial version will focus on a common set of parameters.

### 4.3. Token Counting

The `tiktoken` library will be used for accurate and efficient client-side token counting, which is crucial for managing context windows and predicting costs.

## 5. Dependencies and Setup

### 5.1. Required npm Packages

- `openai`: The official OpenAI Node.js library.
- `tiktoken`: For client-side token counting.

### 5.2. Installation

These dependencies will be added to `packages/core/package.json`.

```json
{
  "dependencies": {
    "openai": "^4.0.0",
    "tiktoken": "^1.0.0"
  }
}
```

### 5.3. Security Considerations

OpenAI API keys will be read from the user's config file. The system will ensure that these keys are not exposed in logs.
