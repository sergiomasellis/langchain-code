<div align="center">
  <img src="../assets/logo.png" alt="LangCode Logo" width="180" />
  <h1 align="center">LangCode</h1>

  <p align="center"><i><b>The only CLI you'll ever need!</b></i></p>
</div>

# OpenRouter Integration

LangCode now supports **OpenRouter**, a unified API gateway that provides access to hundreds of language models from multiple providers. This enables you to leverage models from OpenAI, Anthropic, Google, Mistral, Meta, and more through a single unified interface.

## What is OpenRouter?

OpenRouter (https://openrouter.ai/) is an API service that:
- Provides access to 100+ language models from multiple providers
- Offers unified authentication with a single API key
- Allows easy switching between models and providers
- Handles token accounting and billing across providers
- Provides uptime monitoring and fallback routing

## Setup

### 1. Get Your OpenRouter API Key

1. Go to [https://openrouter.ai/](https://openrouter.ai/)
2. Sign up for a free account
3. Navigate to your API keys page at [https://openrouter.ai/keys](https://openrouter.ai/keys)
4. Create a new API key
5. Copy your API key

### 2. Configure Your Environment

Add your OpenRouter API key to your `.env` file (in the folder you've chosen as your Project directory):

```bash
OPENROUTER_API_KEY=your_api_key_here
# Optional overrides:
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_SITE_URL=https://your-app.example
OPENROUTER_SITE_NAME=LangCode
```

Or set them as system environment variables:

```bash
export OPENROUTER_API_KEY=your_api_key_here
export OPENROUTER_BASE_URL=https://openrouter.ai/api/v1  # optional
export OPENROUTER_SITE_URL=https://your-app.example      # optional but recommended
export OPENROUTER_SITE_NAME="LangCode"                   # optional but recommended
```

`OPENROUTER_SITE_URL` and `OPENROUTER_SITE_NAME` are forwarded as the `HTTP-Referer` and `X-Title` headers respectively—OpenRouter recommends supplying them so requests are associated with your app.

### 3. Set OpenRouter as Your Provider

You can use OpenRouter as your LLM provider in several ways:

**Via CLI flag:**
```bash
langcode --llm openrouter
```

**Via environment variable:**
```bash
export LLM_PROVIDER=openrouter
langcode
```

## Available Models

LangCode includes pre-configured OpenRouter models:

| Model | ID | Latency | Reasoning | Context | Best For |
|-------|----|---------|-----------|---------|---------  |
| **GPT-4o** | `openai/gpt-4o` | Fast | Very High | 200K | Complex tasks, reasoning |
| **Claude 3.5 Sonnet** | `anthropic/claude-3.5-sonnet` | Medium | High | 200K | Balanced quality & speed |
| **Gemini 2.0 Flash** | `google/gemini-2.0-flash-lite` | Very Fast | Medium | 1M | Cost-effective, large contexts |
| **Llama 3.3 70B** | `meta-llama/llama-3.3-70b-instruct` | Medium | Medium | 8K | Open-source alternative |
| **Mistral Large 2** | `mistralai/mistral-large-2407` | Medium | Medium | 32K | Fast & capable open model |

### Getting All Available Models

To see all models available through OpenRouter, you can query the OpenRouter API:

```bash
curl https://openrouter.ai/api/v1/models \
     -H "Authorization: Bearer <your-api-key>"
```

This will return a JSON list of all available models with their details including pricing, context window, and more.

## Usage Examples

### Basic Usage

```bash
# Use OpenRouter with default model (Gemini 2.0 Flash Lite)
export OPENROUTER_API_KEY=your_key_here
export LLM_PROVIDER=openrouter
langcode

# Or with CLI flag
langcode --llm openrouter
```

### Using a Specific Model

When you provide a query, LangCode's intelligent router will automatically select an appropriate model based on complexity. However, you can manually select a specific model by editing your `config_core.py` or using the router's priority system.

### Python Example

LangCode uses the same LangChain primitives you would use directly. The snippet below mirrors the internal configuration—feel free to adapt it for standalone scripts:

```python
from dotenv import load_dotenv
from os import getenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain

load_dotenv()

prompt = PromptTemplate(
    template="Question: {question}\nAnswer: Let's think step by step.",
    input_variables=["question"],
)
llm = ChatOpenAI(
    api_key=getenv("OPENROUTER_API_KEY"),
    base_url=getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    model="google/gemini-2.0-flash-lite",
    default_headers={
        "HTTP-Referer": getenv("OPENROUTER_SITE_URL", ""),
        "X-Title": getenv("OPENROUTER_SITE_NAME", ""),
    },
)
chain = LLMChain(prompt=prompt, llm=llm)
print(chain.run("What NFL team won the Super Bowl in the year Justin Bieber was born?"))
```

### Using Different Priorities

LangCode supports different optimization priorities:

```bash
# Speed-first (default)
langcode --priority speed --llm openrouter "quick task"

# Quality-first  
langcode --priority quality --llm openrouter "complex reasoning task"

# Cost-first
langcode --priority cost --llm openrouter "simple task"
```

## How Model Selection Works

LangCode uses an intelligent router that analyzes your query and selects the best model:

1. **Complexity Analysis**: The query is analyzed for complexity indicators (keywords, length, structure)
2. **Complexity Classification**: Query is classified as `simple`, `medium`, `complex`, or `overly_complex`
3. **Model Selection**: 
   - **Simple**: Fastest available model (Gemini 2.0 Flash Lite)
   - **Medium**: Balanced models (Claude 3.5 Sonnet)
   - **Complex**: Higher reasoning power (GPT-4o, Claude 3.5 Sonnet)
   - **Overly Complex**: Maximum reasoning capability (GPT-4o)

## Pricing Considerations

OpenRouter provides transparent pricing. Models typically cost:

- **Gemini 2.0 Flash Lite**: ~$0.10 per million input tokens, ~$0.40 per million output tokens
- **Claude 3.5 Sonnet**: ~$3.00 per million input tokens, ~$15.00 per million output tokens
- **GPT-4o**: ~$5.00 per million input tokens, ~$15.00 per million output tokens
- **Llama 3.3 70B**: ~$0.60 per million input tokens, ~$0.60 per million output tokens
- **Mistral Large 2**: ~$2.00 per million input tokens, ~$6.00 per million output tokens

Check OpenRouter's pricing page for the most up-to-date rates.

## Troubleshooting

### Missing API Key Error

If you see:
```
Missing OPENROUTER_API_KEY. Set this in your .env (same folder you chose as Project).
Get your API key at https://openrouter.ai/
```

**Solution**: Make sure to:
1. Get your API key from https://openrouter.ai/keys
2. Add it to your `.env` file: `OPENROUTER_API_KEY=your_key`
3. Restart your CLI session

### Model Not Found

If you get an error about a model not being available:

1. Query the OpenRouter API to see available models:
   ```bash
   curl https://openrouter.ai/api/v1/models \
        -H "Authorization: Bearer <your-api-key>"
   ```

2. Update your desired model in `config_core.py` by editing the `openrouter_models` list

### Rate Limiting

OpenRouter has rate limits. If you hit them:
- Wait a few moments and retry
- Consider using faster/cheaper models
- Check your OpenRouter dashboard for usage stats

## Migration from Other Providers

Switching to OpenRouter from other providers is simple:

```bash
# Before: Using Anthropic
export LLM_PROVIDER=anthropic
export ANTHROPIC_API_KEY=your_key

# After: Using OpenRouter
export LLM_PROVIDER=openrouter
export OPENROUTER_API_KEY=your_key
```

All your workflows, agents, and tools continue to work unchanged!

## Advanced Configuration

### Custom Model List

You can extend the available models by editing `src/langchain_code/config_core.py` and adding to the `self.openrouter_models` list in the `IntelligentLLMRouter.__init__` method:

```python
self.openrouter_models = [
    # ... existing models ...
    ModelConfig(
        name="Your Custom Model",
        input_cost_per_million=0.50,
        output_cost_per_million=2.00,
        capabilities="Custom model description",
        latency_tier=2,
        reasoning_strength=7,
        context_window=128_000,
        provider="openrouter",
        model_id="provider/model-name",
        langchain_model_name="provider/model-name",
    ),
]
```

## Resources

- **OpenRouter Website**: https://openrouter.ai/
- **OpenRouter Documentation**: https://openrouter.ai/docs
- **LangChain OpenRouter**: https://github.com/langchain-ai/langchain/tree/master/libs/partners/openrouter
- **Available Models API**: https://openrouter.ai/api/v1/models

## Support

For issues with:
- **OpenRouter API/Keys**: Check https://openrouter.ai/docs or contact OpenRouter support
- **LangCode Integration**: Check the main LangCode documentation or report an issue
- **Model-specific Issues**: Refer to the model provider's documentation (OpenAI, Anthropic, Google, etc.)
