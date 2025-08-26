# OpenAI Integration

Gemini CLI supports OpenAI's GPT models, allowing you to use GPT-4, GPT-3.5-turbo, and other OpenAI models directly from your terminal. This guide covers everything you need to know about using OpenAI with Gemini CLI.

## Getting Started

### 1. Get Your OpenAI API Key

1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign in to your OpenAI account (or create one)
3. Click "Create new secret key"
4. Copy your API key and store it securely

### 2. Configure Your API Key

You can configure your OpenAI API key in several ways:

#### Environment Variable (Recommended)

```bash
export OPENAI_API_KEY="sk-your-api-key-here"
gemini --openai-model gpt-4
```

#### CLI Argument

```bash
gemini --openai-api-key "sk-your-api-key-here" --openai-model gpt-4
```

#### Settings File

Add to your `~/.gemini/settings.json`:

```json
{
  "openaiApiKey": "sk-your-api-key-here",
  "openaiModel": "gpt-4"
}
```

## Configuration Options

### Core Options

| Option        | CLI Argument             | Environment Variable   | Description                                  |
| ------------- | ------------------------ | ---------------------- | -------------------------------------------- |
| API Key       | `--openai-api-key`       | `OPENAI_API_KEY`       | Your OpenAI API key                          |
| Model         | `--openai-model`         | `OPENAI_MODEL`         | OpenAI model to use                          |
| Gateway URL   | `--openai-gateway-url`   | `OPENAI_GATEWAY_URL`   | Custom API endpoint (for Azure OpenAI, etc.) |
| Model Version | `--openai-model-version` | `OPENAI_MODEL_VERSION` | Specific model version                       |

### Configuration Examples

#### Basic Usage

```bash
# Use GPT-4
gemini --openai-model gpt-4

# Use GPT-3.5-turbo
gemini --openai-model gpt-3.5-turbo
```

#### Azure OpenAI

```bash
export OPENAI_API_KEY="your-azure-api-key"
export OPENAI_GATEWAY_URL="https://your-resource.openai.azure.com"
gemini --openai-model gpt-4 --openai-model-version "2024-02-15-preview"
```

#### Custom Gateway

```bash
export OPENAI_API_KEY="your-api-key"
export OPENAI_GATEWAY_URL="https://your-custom-gateway.com/v1"
gemini --openai-model gpt-4
```

## Supported Models

### GPT-4 Family

| Model                 | Description                      | Context Window | Best For                           |
| --------------------- | -------------------------------- | -------------- | ---------------------------------- |
| `gpt-4`               | Most capable GPT-4 model         | 8,192 tokens   | Complex reasoning, code generation |
| `gpt-4-turbo`         | Faster GPT-4 with larger context | 128,000 tokens | Large codebases, long documents    |
| `gpt-4-turbo-preview` | Latest GPT-4 turbo preview       | 128,000 tokens | Cutting-edge capabilities          |

### GPT-3.5 Family

| Model               | Description              | Context Window | Best For                 |
| ------------------- | ------------------------ | -------------- | ------------------------ |
| `gpt-3.5-turbo`     | Fast and efficient       | 4,096 tokens   | Quick tasks, simple code |
| `gpt-3.5-turbo-16k` | Extended context version | 16,384 tokens  | Medium-sized projects    |

### Model Selection Tips

- **GPT-4**: Best for complex reasoning, code review, architecture decisions
- **GPT-4-turbo**: Ideal for large codebases and long conversations
- **GPT-3.5-turbo**: Great for quick tasks, simple queries, and cost-effective usage

## Usage Examples

### Basic Chat

```bash
# Start a conversation with GPT-4
gemini --openai-model gpt-4
> Explain the difference between async/await and Promises in JavaScript
```

### Code Analysis

```bash
# Analyze your codebase with GPT-4
cd your-project/
gemini --openai-model gpt-4
> Review this codebase and suggest improvements
```

### Non-Interactive Mode

```bash
# Use in scripts
gemini --openai-model gpt-4 -p "Generate a README for this project"
```

## Best Practices

### Model Selection

1. **Start with GPT-3.5-turbo** for simple tasks to save costs
2. **Use GPT-4** for complex reasoning and code generation
3. **Choose GPT-4-turbo** for large codebases or long conversations

### Cost Optimization

1. **Use appropriate models**: Don't use GPT-4 for simple tasks
2. **Manage context**: Keep conversations focused to reduce token usage
3. **Monitor usage**: Check your OpenAI dashboard regularly

### Security

1. **Protect your API key**: Never commit API keys to version control
2. **Use environment variables**: Store keys in environment variables or secure vaults
3. **Rotate keys regularly**: Generate new API keys periodically

## Troubleshooting

### Common Issues

#### Authentication Errors

**Problem**: `Invalid API key` or `Unauthorized` errors

**Solutions**:

- Verify your API key is correct and active
- Check that your OpenAI account has sufficient credits
- Ensure the API key has the necessary permissions

```bash
# Test your API key
curl -H "Authorization: Bearer $OPENAI_API_KEY" \
     https://api.openai.com/v1/models
```

#### Model Not Available

**Problem**: `Model not found` or `Model not available` errors

**Solutions**:

- Check the [OpenAI Models documentation](https://platform.openai.com/docs/models) for available models
- Verify your account has access to the requested model
- Try a different model (e.g., `gpt-3.5-turbo` instead of `gpt-4`)

#### Rate Limiting

**Problem**: `Rate limit exceeded` errors

**Solutions**:

- Wait before retrying (rate limits reset over time)
- Upgrade your OpenAI plan for higher rate limits
- Use exponential backoff in automated scripts

#### Gateway Issues

**Problem**: Connection errors with custom gateways

**Solutions**:

- Verify the gateway URL is correct and accessible
- Check authentication requirements for your gateway
- Test the gateway with curl or another HTTP client

### Debug Mode

Enable debug logging to troubleshoot issues:

```bash
export DEBUG=gemini:openai
gemini --openai-model gpt-4
```

## Migration Guide

### From Gemini to OpenAI

If you're currently using Gemini and want to try OpenAI:

1. **Get an OpenAI API key** (see Getting Started section)
2. **Set your environment variable**:
   ```bash
   export OPENAI_API_KEY="your-api-key"
   ```
3. **Start using OpenAI models**:
   ```bash
   # Instead of: gemini -m gemini-2.5-pro
   gemini --openai-model gpt-4
   ```

### Configuration Migration

Update your `~/.gemini/settings.json`:

```json
{
  // Old Gemini config
  "geminiApiKey": "your-gemini-key",
  "model": "gemini-2.5-pro",

  // Add OpenAI config
  "openaiApiKey": "your-openai-key",
  "openaiModel": "gpt-4"
}
```

### Feature Comparison

| Feature         | Gemini               | OpenAI                          |
| --------------- | -------------------- | ------------------------------- |
| Context Window  | Up to 1M tokens      | Up to 128K tokens (GPT-4-turbo) |
| Code Generation | Excellent            | Excellent                       |
| Reasoning       | Excellent            | Excellent                       |
| Cost            | Free tier available  | Pay-per-use                     |
| Models          | Gemini 2.5 Pro/Flash | GPT-4, GPT-3.5-turbo            |

## Advanced Configuration

### Custom Headers

For advanced gateway configurations, you can set custom headers:

```bash
export OPENAI_CUSTOM_HEADERS='{"Custom-Header": "value"}'
gemini --openai-model gpt-4
```

### Proxy Configuration

If you're behind a corporate proxy:

```bash
export HTTPS_PROXY="http://your-proxy:8080"
gemini --openai-model gpt-4
```

### Timeout Settings

Configure request timeouts:

```bash
export OPENAI_TIMEOUT="60000"  # 60 seconds
gemini --openai-model gpt-4
```

## Support

For OpenAI-specific issues:

1. Check the [OpenAI Status Page](https://status.openai.com/)
2. Review [OpenAI Documentation](https://platform.openai.com/docs)
3. Contact [OpenAI Support](https://help.openai.com/)

For Gemini CLI integration issues:

1. Use the `/bug` command in Gemini CLI
2. Check [GitHub Issues](https://github.com/google-gemini/gemini-cli/issues)
3. Review the [Troubleshooting Guide](../troubleshooting.md)
