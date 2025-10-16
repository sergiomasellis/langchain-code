#!/usr/bin/env python3
"""
Utility script to fetch and display available models from OpenRouter.

Usage:
    python scripts/list_openrouter_models.py
    python scripts/list_openrouter_models.py --api-key YOUR_KEY
    python scripts/list_openrouter_models.py --base-url https://openrouter.ai/api/v1
    python scripts/list_openrouter_models.py --site-url https://your-app.example --site-name "LangCode"
    python scripts/list_openrouter_models.py --filter claude
    python scripts/list_openrouter_models.py --filter claude --sort-by cost
"""

import os
import sys
import json
import argparse
from typing import Optional, List, Dict, Any
import urllib.request
import urllib.error


def fetch_openrouter_models(
    api_key: str,
    base_url: str,
    extra_headers: Optional[Dict[str, str]] = None,
) -> Optional[List[Dict[str, Any]]]:
    """
    Fetch available models from OpenRouter API.
    
    Args:
        api_key: OpenRouter API key
        base_url: Base URL for the OpenRouter API
        extra_headers: Optional headers to merge into the request
        
    Returns:
        List of model dictionaries or None if fetch fails
    """
    url = f"{base_url.rstrip('/')}/models"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "User-Agent": "LangCode-OpenRouter-Lister/1.0",
    }
    if extra_headers:
        headers.update({k: v for k, v in extra_headers.items() if v})
    
    try:
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request, timeout=10) as response:
            data = json.loads(response.read().decode('utf-8'))
            return data.get('data', [])
    except urllib.error.HTTPError as e:
        if e.code == 401:
            print("❌ Error: Invalid API key. Check your OPENROUTER_API_KEY.", file=sys.stderr)
        else:
            print(f"❌ HTTP Error {e.code}: {e.reason}", file=sys.stderr)
    except urllib.error.URLError as e:
        print(f"❌ Network Error: {e.reason}", file=sys.stderr)
    except json.JSONDecodeError:
        print("❌ Error: Failed to parse response JSON", file=sys.stderr)
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
    
    return None


def format_price(price_or_none: Optional[float]) -> str:
    """Format price for display."""
    if price_or_none is None:
        return "Free"
    if price_or_none == 0:
        return "Free"
    return f"${price_or_none:.2e}"


def display_models(models: List[Dict[str, Any]], 
                  filter_text: Optional[str] = None,
                  sort_by: str = "name",
                  limit: Optional[int] = None) -> None:
    """
    Display models in a formatted table.
    
    Args:
        models: List of model dictionaries
        filter_text: Optional filter text (case-insensitive)
        sort_by: Sort key ('name', 'cost', 'context')
        limit: Maximum number of models to display
    """
    # Filter models
    filtered = models
    if filter_text:
        filter_lower = filter_text.lower()
        filtered = [m for m in models if filter_lower in m.get('id', '').lower()]
    
    if not filtered:
        print(f"No models found matching '{filter_text}'")
        return
    
    # Sort models
    if sort_by == "cost":
        filtered = sorted(filtered, key=lambda m: m.get('pricing', {}).get('prompt', 0))
    elif sort_by == "context":
        filtered = sorted(filtered, key=lambda m: m.get('context_length', 0), reverse=True)
    else:  # name (default)
        filtered = sorted(filtered, key=lambda m: m.get('id', ''))
    
    # Limit results
    if limit:
        filtered = filtered[:limit]
    
    # Display header
    print("\n" + "=" * 120)
    print(f"{'Model ID':<45} {'Provider':<15} {'Context':<12} {'Input Price':<15} {'Output Price':<15}")
    print("=" * 120)
    
    # Display models
    for model in filtered:
        model_id = model.get('id', 'N/A')
        provider = model.get('architecture', {}).get('modality', 'N/A')
        context = model.get('context_length', 'N/A')
        
        pricing = model.get('pricing', {})
        input_price = pricing.get('prompt')
        output_price = pricing.get('completion')
        
        # Truncate model ID if too long
        if len(model_id) > 45:
            model_id = model_id[:42] + "..."
        
        context_str = f"{context:,}" if isinstance(context, int) else str(context)
        
        print(f"{model_id:<45} {provider:<15} {context_str:<12} "
              f"{format_price(input_price):<15} {format_price(output_price):<15}")
    
    print("=" * 120)
    print(f"\nTotal models displayed: {len(filtered)}")
    print(f"Total models available: {len(models)}")


def main():
    parser = argparse.ArgumentParser(
        description="Fetch and display available models from OpenRouter",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/list_openrouter_models.py
  python scripts/list_openrouter_models.py --filter claude
  python scripts/list_openrouter_models.py --filter gpt --sort-by cost
  python scripts/list_openrouter_models.py --base-url https://openrouter.ai/api/v1
  python scripts/list_openrouter_models.py --site-url https://your-app.example --site-name \"LangCode\"
  python scripts/list_openrouter_models.py --sort-by context --limit 20
        """
    )
    
    parser.add_argument(
        "--api-key",
        help="OpenRouter API key (uses OPENROUTER_API_KEY env var if not provided)"
    )
    parser.add_argument(
        "--base-url",
        help="Override the OpenRouter API base URL (default: env OPENROUTER_BASE_URL or https://openrouter.ai/api/v1)"
    )
    parser.add_argument(
        "--site-url",
        help="Send HTTP-Referer header (default: env OPENROUTER_SITE_URL or YOUR_SITE_URL)"
    )
    parser.add_argument(
        "--site-name",
        help="Send X-Title header (default: env OPENROUTER_SITE_NAME or YOUR_SITE_NAME)"
    )
    parser.add_argument(
        "--filter",
        help="Filter models by name (case-insensitive substring match)"
    )
    parser.add_argument(
        "--sort-by",
        choices=["name", "cost", "context"],
        default="name",
        help="Sort results by (default: name)"
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of models to display"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw JSON instead of formatted table"
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        print("❌ Error: No API key provided.", file=sys.stderr)
        print("Set OPENROUTER_API_KEY environment variable or use --api-key flag.", file=sys.stderr)
        sys.exit(1)
    
    base_url = args.base_url or os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
    site_url = (
        args.site_url
        or os.getenv("OPENROUTER_SITE_URL")
        or os.getenv("YOUR_SITE_URL")
    )
    site_name = (
        args.site_name
        or os.getenv("OPENROUTER_SITE_NAME")
        or os.getenv("YOUR_SITE_NAME")
    )
    extra_headers = {}
    if site_url:
        extra_headers["HTTP-Referer"] = site_url
    if site_name:
        extra_headers["X-Title"] = site_name
    
    # Fetch models
    print("⏳ Fetching models from OpenRouter...")
    models = fetch_openrouter_models(api_key, base_url, extra_headers or None)
    
    if models is None:
        sys.exit(1)
    
    print(f"✅ Successfully fetched {len(models)} models")
    
    # Display results
    if args.json:
        print(json.dumps(models, indent=2))
    else:
        display_models(models, args.filter, args.sort_by, args.limit)


if __name__ == "__main__":
    main()
