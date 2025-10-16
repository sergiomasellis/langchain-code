# OpenRouter Support - Quick Setup Guide

OpenRouter support has been successfully added to LangCode! Here's what was implemented and how to use it.

## What Was Added

### 1. **Configuration Support** (`config_core.py`)
- Added `openrouter_models` list with 5 pre-configured models:
  - **GPT-4o** (openai/gpt-4o)
  - **Claude 3.5 Sonnet** (anthropic/claude-3.5-sonnet)
  - **Gemini 2.0 Flash Lite** (google/gemini-2.0-flash-lite) - Default
  - **Llama 3.3 70B** (meta-llama/llama-3.3-70b-instruct)
  - **Mistral Large 2** (mistralai/mistral-large-2407)

- Updated `resolve_provider()` to recognize "openrouter"
- Updated `select_optimal_model_for_provider()` to handle OpenRouter models
- Updated `get_model()` to support OpenRouter provider
- Updated `get_model_info()` to provide OpenRouter default model info
- Added OpenRouter initialization in `_cached_chat_model()` with automatic base URL/default header support and a fallback to `ChatOpenAI` when `langchain-openrouter` isn't installed

### 2. **Dependencies** (`pyproject.toml`)
- Added `langchain-openrouter>=0.1.0` to project dependencies

### 3. **Documentation** (`docs/`)
- **`openrouter.md`**: Comprehensive guide including:
  - Setup instructions
  - Available models reference table
  - Usage examples
  - Model selection logic explanation
  - Pricing information
  - Troubleshooting tips
  - Advanced configuration
  
- Updated **`index.md`**: Added OpenRouter documentation link

### 4. **Utility Scripts** (`scripts/`)
- **`list_openrouter_models.py`**: Python script to fetch and display all available OpenRouter models
  - Filter by name (e.g., "claude", "gpt")
  - Sort by name, cost, or context window
  - Export to JSON
  - Full error handling

## Quick Start

### 1. Install Dependencies
```bash
pip install -e .
# or specifically:
pip install langchain-openrouter
```

### 2. Get Your API Key
1. Visit https://openrouter.ai/
2. Sign up (free tier available)
3. Get your API key from https://openrouter.ai/keys

### 3. Configure Environment
Add to your `.env` file or export as environment variables:
```bash
OPENROUTER_API_KEY=your_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1      # optional
OPENROUTER_SITE_URL=https://your-app.example          # optional but recommended
OPENROUTER_SITE_NAME="LangCode"                       # optional but recommended
```

### 4. Use OpenRouter
```bash
# Via CLI flag
langcode --llm openrouter

# Or via environment variable
export LLM_PROVIDER=openrouter
langcode

# With a specific priority
langcode --llm openrouter --priority speed "implement a feature"
```

## Environment Variables

| Variable | Purpose | Example |
|----------|---------|---------|
| `OPENROUTER_API_KEY` | **Required** API key for authentication | `sk_live_xxx...` |
| `OPENROUTER_BASE_URL` | Override base URL (defaults to `https://openrouter.ai/api/v1`) | `https://openrouter.ai/api/v1` |
| `OPENROUTER_SITE_URL` | Sent as `HTTP-Referer` header (identifies your app) | `https://your-app.example` |
| `OPENROUTER_SITE_NAME` | Sent as `X-Title` header (friendly app name) | `LangCode` |
| `LLM_PROVIDER` | Set default provider | `openrouter` |

## Available Models

Run this to see all available models on OpenRouter:
```bash
python scripts/list_openrouter_models.py
```

Filter for specific models:
```bash
python scripts/list_openrouter_models.py --filter claude --sort-by cost
python scripts/list_openrouter_models.py --filter gpt --limit 10
```

## Model Selection Logic

LangCode's intelligent router automatically selects models based on query complexity:

| Complexity | Default Model | Reasoning Strength |
|-----------|---------------|------------------|
| Simple | Gemini 2.0 Flash Lite | Medium |
| Medium | Claude 3.5 Sonnet | High |
| Complex | GPT-4o or Claude | Very High |
| Overly Complex | GPT-4o | Maximum |

## Troubleshooting

**Missing API Key Error:**
```
Missing OPENROUTER_API_KEY. Set this in your .env
```
→ Solution: Add `OPENROUTER_API_KEY=your_key` to `.env`

**Model Not Found:**
→ Solution: Run `python scripts/list_openrouter_models.py` to see available models

**Rate Limiting:**
→ Solution: Wait a moment and retry. Check your OpenRouter dashboard for usage.

## Documentation

For detailed information, see:
- `docs/openrouter.md` - Full OpenRouter integration guide
- `scripts/README.md` - Utility scripts documentation

## Pricing

Models are billed per token through OpenRouter:
- **Gemini 2.0 Flash Lite**: ~$0.10-$0.40 per million tokens (cheapest)
- **Llama/Mistral**: ~$0.60-$6.00 per million tokens
- **Claude**: ~$3-$15 per million tokens
- **GPT-4o**: ~$5-$15 per million tokens

Check https://openrouter.ai/ for current pricing and promotions.

## Migration from Other Providers

Simply switch the provider:

```bash
# Before (Gemini)
export OPENROUTER_API_KEY=...
export LLM_PROVIDER=openrouter

# All your existing workflows work unchanged!
```

## Next Steps

1. ✅ Install the dependencies
2. ✅ Get an OpenRouter API key
3. ✅ Configure your `.env` file
4. ✅ Try: `langcode --llm openrouter`
5. ✅ Read `docs/openrouter.md` for detailed usage

## Files Changed

- `src/langchain_code/config_core.py` - Added OpenRouter support
- `pyproject.toml` - Added dependency
- `docs/openrouter.md` - New comprehensive guide
- `docs/index.md` - Updated with OpenRouter link
- `scripts/list_openrouter_models.py` - New utility script
- `scripts/README.md` - Updated with new script

Enjoy using 100+ models with OpenRouter! 🚀
