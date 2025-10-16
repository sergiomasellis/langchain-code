# LangCode Scripts

Utility scripts for LangCode.

## list_openrouter_models.py

Fetch and display available models from OpenRouter.

### Usage

```bash
python scripts/list_openrouter_models.py
```

### Requirements

- `OPENROUTER_API_KEY` environment variable set, or use `--api-key` flag

### Examples

**List all available models:**
```bash
python scripts/list_openrouter_models.py
```

**Filter for Claude models:**
```bash
python scripts/list_openrouter_models.py --filter claude
```

**Filter for GPT models, sorted by cost:**
```bash
python scripts/list_openrouter_models.py --filter gpt --sort-by cost
```

**Show top 20 models with largest context windows:**
```bash
python scripts/list_openrouter_models.py --sort-by context --limit 20
```

**Get raw JSON output:**
```bash
python scripts/list_openrouter_models.py --json
```

### Options

- `--api-key KEY`: OpenRouter API key (uses `OPENROUTER_API_KEY` env var if not provided)
- `--base-url URL`: Override the OpenRouter API base URL (defaults to `OPENROUTER_BASE_URL` or `https://openrouter.ai/api/v1`)
- `--site-url URL`: Send `HTTP-Referer` header (defaults to `OPENROUTER_SITE_URL` or `YOUR_SITE_URL`)
- `--site-name NAME`: Send `X-Title` header (defaults to `OPENROUTER_SITE_NAME` or `YOUR_SITE_NAME`)
- `--filter TEXT`: Filter models by name (case-insensitive substring match)
- `--sort-by {name|cost|context}`: Sort results (default: name)
- `--limit N`: Maximum number of models to display
- `--json`: Output raw JSON instead of formatted table
