from __future__ import annotations
import os
import re
import json
import logging
import subprocess
from typing import Optional, Dict, Any, Tuple, List, Dict as _Dict, Any as _Any
from dataclasses import dataclass
from dotenv import load_dotenv
import requests

load_dotenv()
logger = logging.getLogger(__name__)
def _normalize_gemini_env() -> None:
    """
    Make Gemini work regardless of whether the user set GEMINI_API_KEY or GOOGLE_API_KEY.
    LangChain's ChatGoogleGenerativeAI reads GOOGLE_API_KEY by default, so mirror whichever
    one is present into the other if it's missing.
    """
    gemini = os.environ.get("GEMINI_API_KEY")
    google = os.environ.get("GOOGLE_API_KEY")

    if gemini and not google:
        os.environ["GOOGLE_API_KEY"] = gemini

    if google and not gemini:
        os.environ["GEMINI_API_KEY"] = google

_normalize_gemini_env()


@dataclass
class ModelConfig:
    name: str
    input_cost_per_million: float
    output_cost_per_million: float
    capabilities: str
    latency_tier: int
    reasoning_strength: int
    context_window: int
    provider: str
    model_id: str
    langchain_model_name: str
    context_threshold: Optional[int] = None

class IntelligentLLMRouter:
    """
    Speed-first router with reasoning override:
      - For simple/medium: pick the *fastest* viable model (latency, then cost, then reasoning).
      - For complex/overly_complex: *force strong reasoning* (>=8 / >=9), then pick the *fastest* among them.
    """
    def __init__(self, prefer_lightweight: bool = True):
        self.prefer_lightweight = prefer_lightweight

        # Slightly higher thresholds reduce unnecessary escalation to heavy models.
        if prefer_lightweight:
            self.simple_threshold = 18
            self.medium_threshold = 48
            self.complex_threshold = 78
        else:
            self.simple_threshold = 16
            self.medium_threshold = 42
            self.complex_threshold = 70

        self.complexity_vocabulary = {
            'high_complexity': {
                'microservices', 'architecture', 'distributed', 'blockchain',
                'orchestration', 'kubernetes', 'terraform', 'observability',
                'saga', 'cqrs', 'event-sourcing', 'consensus', 'governance',
                'infrastructure', 'comprehensive', 'enterprise-grade',
                'machine-learning', 'ai', 'neural-network', 'deep-learning'
            },
            'medium_complexity': {
                'implement', 'design', 'optimize', 'integrate', 'refactor',
                'authentication', 'authorization', 'database', 'api', 'framework',
                'pipeline', 'deployment', 'monitoring', 'logging', 'testing',
                'docker', 'container', 'websocket', 'oauth', 'jwt', 'redis',
                'react', 'nodejs', 'typescript', 'migration', 'dashboard'
            },
            'dev_actions': {
                'create', 'build', 'setup', 'configure', 'generate', 'convert',
                'add', 'fix', 'write', 'develop', 'establish'
            }
        }

        self.technical_indicators = {
            'full-stack', 'real-time', 'ci/cd', 'end-to-end', 'e2e', 'unit-test',
            'error-handling', 'load-balancer', 'service-mesh', 'auto-scaling'
        }

        self.question_words = {'how', 'why', 'what', 'when', 'where', 'which', 'who'}
        self.conditional_words = {'if', 'unless', 'provided', 'assuming', 'given', 'suppose'}

        # --- Provider model catalogs ---
        self.gemini_models = [
            ModelConfig(
                name="Gemini 2.0 Flash-Lite",
                input_cost_per_million=0.075,
                output_cost_per_million=0.30,
                capabilities="Smallest, most cost-effective for simple tasks",
                latency_tier=1,
                reasoning_strength=4,
                context_window=1_000_000,
                provider="gemini",
                model_id="gemini-2.0-flash-lite",
                langchain_model_name="gemini-2.0-flash-lite"
            ),
            ModelConfig(
                name="Gemini 2.0 Flash",
                input_cost_per_million=0.10,
                output_cost_per_million=0.40,
                capabilities="Balanced multimodal model for agents",
                latency_tier=1,
                reasoning_strength=6,
                context_window=1_000_000,
                provider="gemini",
                model_id="gemini-2.0-flash",
                langchain_model_name="gemini-2.0-flash"
            ),
            ModelConfig(
                name="Gemini 2.5 Flash-Lite",
                input_cost_per_million=0.10,
                output_cost_per_million=0.40,
                capabilities="Cost-effective with thinking budgets",
                latency_tier=1,
                reasoning_strength=5,
                context_window=1_000_000,
                provider="gemini",
                model_id="gemini-2.5-flash-lite",
                langchain_model_name="gemini-2.5-flash-lite"
            ),
            ModelConfig(
                name="Gemini 2.5 Flash",
                input_cost_per_million=0.30,
                output_cost_per_million=2.50,
                capabilities="Hybrid reasoning model with thinking budgets",
                latency_tier=2,
                reasoning_strength=7,
                context_window=1_000_000,
                provider="gemini",
                model_id="gemini-2.5-flash",
                langchain_model_name="gemini-2.5-flash"
            ),
            ModelConfig(
                name="Gemini 2.5 Pro",
                input_cost_per_million=1.25,
                output_cost_per_million=10.00,
                capabilities="State-of-the-art for coding and complex reasoning",
                latency_tier=3,
                reasoning_strength=10,
                context_window=2_000_000,
                provider="gemini",
                model_id="gemini-2.5-pro",
                langchain_model_name="gemini-2.5-pro",
                context_threshold=200_000
            )
        ]

        self.anthropic_models = [
            ModelConfig(
                name="Claude 3.5 Haiku",
                input_cost_per_million=0.80,
                output_cost_per_million=4.00,
                capabilities="Fastest, most cost-effective model",
                latency_tier=1,
                reasoning_strength=5,
                context_window=200_000,
                provider="anthropic",
                model_id="claude-3-5-haiku-20241022",
                langchain_model_name="claude-3-5-haiku-20241022"
            ),
            ModelConfig(
                name="Claude Sonnet (3.7)",
                input_cost_per_million=3.00,
                output_cost_per_million=15.00,
                capabilities="Optimal balance of intelligence, cost, and speed",
                latency_tier=2,
                reasoning_strength=8,
                context_window=200_000,
                provider="anthropic",
                model_id="claude-3-7-sonnet-20250219",
                langchain_model_name="claude-3-7-sonnet-20250219",
                context_threshold=200_000
            ),
            ModelConfig(
                name="Claude Opus 4.1",
                input_cost_per_million=15.00,
                output_cost_per_million=75.00,
                capabilities="Most intelligent model for complex tasks",
                latency_tier=4,
                reasoning_strength=10,
                context_window=200_000,
                provider="anthropic",
                model_id="claude-opus-4-1-20250805",
                langchain_model_name="claude-opus-4-1-20250805"
            )
        ]

        self.openai_models = [
            ModelConfig(
                name="GPT-4o Mini",
                input_cost_per_million=0.15,
                output_cost_per_million=0.60,
                capabilities="Fast + inexpensive general model",
                latency_tier=2,
                reasoning_strength=7,
                context_window=200_000,
                provider="openai",
                model_id="gpt-4o-mini",
                langchain_model_name="gpt-4o-mini",
            ),
            ModelConfig(
                name="GPT-4o",
                input_cost_per_million=5.00,
                output_cost_per_million=15.00,
                capabilities="Higher quality multimodal/chat",
                latency_tier=3,
                reasoning_strength=9,
                context_window=200_000,
                provider="openai",
                model_id="gpt-4o",
                langchain_model_name="gpt-4o",
            ),
        ]
        self.ollama_models = [ 
            ModelConfig( 
                name="Llama 3.1 (Ollama)", 
                input_cost_per_million=0.0, 
                output_cost_per_million=0.0, 
                capabilities="Local default via Ollama", 
                latency_tier=2, 
                reasoning_strength=7, 
                context_window=128_000, 
                provider="ollama", 
                model_id="llama3.1", 
                langchain_model_name="llama3.1", 
            ), 
        ]

        self.openrouter_models = self._load_openrouter_models() or self._default_openrouter_models()


    def _load_openrouter_models(self) -> List[ModelConfig]:
        api_key = (
            os.getenv("OPENROUTER_API_KEY")
            or os.getenv("OPENROUTE_API_KEY")
        )
        base_url = (os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/")
        headers = {
            "User-Agent": "LangCode-OpenRouter-ModelLoader/1.0",
        }
        referer = os.getenv("OPENROUTER_SITE_URL") or os.getenv("YOUR_SITE_URL")
        title = os.getenv("OPENROUTER_SITE_NAME") or os.getenv("YOUR_SITE_NAME")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title

        url = f"{base_url}/models"
        try:
            response = requests.get(url, headers=headers, timeout=10)
            response.raise_for_status()
        except Exception as exc:
            logger.debug("OpenRouter model fetch failed: %s", exc)
            return []

        try:
            payload = response.json()
        except ValueError as exc:
            logger.debug("OpenRouter model response is not JSON: %s", exc)
            return []

        items = payload.get("data")
        if not isinstance(items, list):
            return []

        models: List[ModelConfig] = []
        for item in items:
            model_id = item.get("id")
            if not isinstance(model_id, str) or not model_id.strip():
                continue
            model_id = model_id.strip()
            name = (item.get("name") or model_id).strip()

            pricing = item.get("pricing") or {}
            prompt_price = self._price_to_million(pricing.get("prompt"))
            completion_price = self._price_to_million(pricing.get("completion"))

            context = self._safe_context(
                item.get("context_length"),
                (item.get("top_provider") or {}).get("context_length"),
            )

            description = (item.get("description") or "").strip()
            if description:
                capabilities = description.splitlines()[0].strip()
            else:
                modality = (item.get("architecture") or {}).get("modality") or "text"
                capabilities = f"OpenRouter: {modality}"

            latency = self._guess_latency(prompt_price, context)
            reasoning = self._guess_reasoning(prompt_price, completion_price)

            models.append(
                ModelConfig(
                    name=name,
                    input_cost_per_million=prompt_price,
                    output_cost_per_million=completion_price,
                    capabilities=capabilities,
                    latency_tier=latency,
                    reasoning_strength=reasoning,
                    context_window=context,
                    provider="openrouter",
                    model_id=model_id,
                    langchain_model_name=model_id,
                )
            )
        return models

    @staticmethod
    def _price_to_million(value: Any) -> float:
        try:
            if value in (None, ""):
                return 0.0
            return float(value) * 1_000_000
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _safe_context(*candidates: Any) -> int:
        for candidate in candidates:
            if isinstance(candidate, (int, float)) and candidate > 0:
                return int(candidate)
            if isinstance(candidate, str):
                try:
                    val = float(candidate)
                    if val > 0:
                        return int(val)
                except ValueError:
                    continue
        return 128_000

    @staticmethod
    def _guess_latency(prompt_price: float, context: int) -> int:
        if prompt_price >= 5 or context >= 400_000:
            return 3
        if prompt_price >= 1 or context >= 160_000:
            return 2
        return 1

    @staticmethod
    def _guess_reasoning(prompt_price: float, completion_price: float) -> int:
        max_price = max(prompt_price, completion_price)
        if max_price >= 5:
            return 9
        if max_price >= 3:
            return 8
        if max_price >= 1:
            return 7
        if max_price >= 0.5:
            return 6
        return 5

    @staticmethod
    def _default_openrouter_models() -> List[ModelConfig]:
        return [
            ModelConfig(
                name="GPT-4o",
                input_cost_per_million=5.00,
                output_cost_per_million=15.00,
                capabilities="OpenRouter: High quality multimodal/chat via GPT-4o",
                latency_tier=3,
                reasoning_strength=9,
                context_window=200_000,
                provider="openrouter",
                model_id="openai/gpt-4o",
                langchain_model_name="openai/gpt-4o",
            ),
            ModelConfig(
                name="Claude 3.5 Sonnet",
                input_cost_per_million=3.00,
                output_cost_per_million=15.00,
                capabilities="OpenRouter: Optimal balance of intelligence, cost, and speed",
                latency_tier=2,
                reasoning_strength=8,
                context_window=200_000,
                provider="openrouter",
                model_id="anthropic/claude-3.5-sonnet",
                langchain_model_name="anthropic/claude-3.5-sonnet",
            ),
            ModelConfig(
                name="Gemini 2.0 Flash Lite",
                input_cost_per_million=0.10,
                output_cost_per_million=0.40,
                capabilities="OpenRouter: Fast multimodal model via Gemini 2.0",
                latency_tier=1,
                reasoning_strength=6,
                context_window=1_000_000,
                provider="openrouter",
                model_id="google/gemini-2.0-flash-lite",
                langchain_model_name="google/gemini-2.0-flash-lite",
            ),
            ModelConfig(
                name="Llama 3.3 70B",
                input_cost_per_million=0.60,
                output_cost_per_million=0.60,
                capabilities="OpenRouter: Fast open-source model via Llama 3.3",
                latency_tier=2,
                reasoning_strength=7,
                context_window=8_000,
                provider="openrouter",
                model_id="meta-llama/llama-3.3-70b-instruct",
                langchain_model_name="meta-llama/llama-3.3-70b-instruct",
            ),
            ModelConfig(
                name="Mistral Large 2",
                input_cost_per_million=2.00,
                output_cost_per_million=6.00,
                capabilities="OpenRouter: Powerful open-source model via Mistral",
                latency_tier=2,
                reasoning_strength=7,
                context_window=32_000,
                provider="openrouter",
                model_id="mistralai/mistral-large-2407",
                langchain_model_name="mistralai/mistral-large-2407",
            ),
        ]


    def extract_features(self, query: str) -> Dict[str, Any]:
        if not query:
            return {
                'word_count': 0, 'char_count': 0, 'sentence_count': 0, 'avg_word_length': 0,
                'conjunction_count': 0, 'comma_count': 0, 'nested_clauses': 0, 'question_words': 0,
                'high_complexity_words': 0, 'medium_complexity_words': 0, 'dev_action_words': 0,
                'technical_indicators': 0, 'conditional_words': 0, 'unique_word_ratio': 0,
                'multiple_requests': 0, 'technical_symbols': 0, 'number_count': 0
            }

        ql = query.lower()
        words = re.findall(r'\b\w+\b', ql)
        sentences = re.split(r'[.!?]+', query)

        high_complex_count = sum(1 for w in words if w in self.complexity_vocabulary['high_complexity'])
        medium_complex_count = sum(1 for w in words if w in self.complexity_vocabulary['medium_complexity'])
        dev_action_count = sum(1 for w in words if w in self.complexity_vocabulary['dev_actions'])
        technical_indicator_count = sum(
            1 for term in self.technical_indicators
            if term in ql or term.replace('-', ' ') in ql
        )

        return {
            'word_count': len(words),
            'char_count': len(query),
            'sentence_count': len([s for s in sentences if s.strip()]),
            'avg_word_length': sum(len(w) for w in words) / max(len(words), 1),
            'conjunction_count': len(re.findall(r'\b(and|or|but|however|therefore|moreover|with)\b', ql)),
            'comma_count': query.count(','),
            'nested_clauses': query.count('(') + query.count('['),
            'question_words': sum(1 for w in words if w in self.question_words),
            'high_complexity_words': high_complex_count,
            'medium_complexity_words': medium_complex_count,
            'dev_action_words': dev_action_count,
            'technical_indicators': technical_indicator_count,
            'conditional_words': sum(1 for w in words if w in self.conditional_words),
            'unique_word_ratio': len(set(words)) / max(len(words), 1),
            'multiple_requests': len(re.findall(r'\b(also|additionally|then|next|after|plus|including)\b', ql)),
            'technical_symbols': len(re.findall(r'[{}()[\]<>/\\]', query)),
            'number_count': len(re.findall(r'\d+', query)),
        }

    def calculate_complexity_score(self, features: Dict[str, Any]) -> int:
        score = 0
        wc = features['word_count']
        if wc <= 3:
            score += 2
        elif wc <= 8:
            score += 6
        elif wc <= 15:
            score += 15
        else:
            score += 22 + (wc - 15) * 1.2

        score += features['sentence_count'] * 2.5
        score += features['conjunction_count'] * 2.5
        score += features['comma_count'] * 0.8
        score += features['nested_clauses'] * 4

        score += features['high_complexity_words'] * 7
        score += features['medium_complexity_words'] * 3.5
        score += features['dev_action_words'] * 1.5
        score += features['technical_indicators'] * 4
        score += features['question_words'] * 1.5
        score += features['conditional_words'] * 2.5

        score += features['multiple_requests'] * 3
        score += features['technical_symbols'] * 1.5
        score += min(features['number_count'], 3) * 0.8

        if features['high_complexity_words'] >= 3:
            score += 10
        if features['medium_complexity_words'] >= 4:
            score += 6
        if wc > 20 and features['technical_indicators'] > 0:
            score += 8
        if features['avg_word_length'] > 6:
            score += 4

        return min(int(score), 120)

    def classify_complexity(self, query: str) -> str:
        if not query or not query.strip():
            return "simple"
        features = self.extract_features(query)
        score = self.calculate_complexity_score(features)
        if score <= self.simple_threshold:
            return "simple"
        elif score <= self.medium_threshold:
            return "medium"
        elif score <= self.complex_threshold:
            return "complex"
        return "overly_complex"

    def select_optimal_model_for_provider(self, query: str, provider: str, priority: str = "balanced") -> ModelConfig:
        complexity = self.classify_complexity(query)

        if provider == "gemini":
            available = self.gemini_models
        elif provider == "anthropic":
            available = self.anthropic_models
        elif provider == "openai":
            available = self.openai_models
        elif provider == "openrouter":
            available = self.openrouter_models
        elif provider == "ollama":
            available = self.ollama_models
        else:
            raise ValueError(f"Unknown provider: {provider}")

        if complexity == "simple":
            candidates = [m for m in available if m.latency_tier <= 2] or available
        elif complexity == "medium":
            candidates = [m for m in available if m.latency_tier <= 3] or available
        elif complexity in {"complex", "overly_complex"}:
            # Reasoning override: require strong reasoning models
            min_reason = 8 if complexity == "complex" else 9
            candidates = [m for m in available if m.reasoning_strength >= min_reason] or \
                         sorted(available, key=lambda m: (-m.reasoning_strength, m.latency_tier))
        else:
            candidates = available

        # 2) Primary optimization is ALWAYS latency, then cost, then reasoning.
        if priority == "quality" and complexity not in {"complex", "overly_complex"}:
            hiq = [m for m in candidates if m.reasoning_strength >= 8] or candidates
            hiq.sort(key=lambda m: (m.latency_tier, -m.reasoning_strength,
                                    m.input_cost_per_million + m.output_cost_per_million))
            return hiq[0]
        if priority == "cost" and complexity not in {"complex", "overly_complex"}:
            cheap = sorted(candidates, key=lambda m: (m.latency_tier,
                                                      m.input_cost_per_million + m.output_cost_per_million,
                                                      -m.reasoning_strength))
            return cheap[0]

        # default and --speed
        fast = sorted(candidates, key=lambda m: (m.latency_tier,
                                                 m.input_cost_per_million + m.output_cost_per_million,
                                                 -m.reasoning_strength))
        return fast[0]

_router = IntelligentLLMRouter(prefer_lightweight=True)

_MODEL_CACHE: _Dict[Tuple[str, str, float], _Any] = {}

def _detect_ollama_models() -> list[str]: 
    """ 
    Probe local Ollama for installed models (best effort, very fast). 
    Returns a list of model names (e.g., ["llama3.1", "mistral", ...]). 
    """ 
    try: 
        p = subprocess.run( 
            ["ollama", "list", "--format", "json"], 
            capture_output=True, text=True, timeout=2 
        ) 
        if p.returncode == 0 and p.stdout.strip(): 
            try: 
                data = json.loads(p.stdout) 
                if isinstance(data, list): 
                    names = [] 
                    for it in data: 
                        # Some versions use "name", some "model" 
                        n = (it.get("name") or it.get("model") or "").strip() 
                        if n: 
                            names.append(n)
                    return list(dict.fromkeys(names)) 
            except Exception: 
                pass 
        p2 = subprocess.run(["ollama", "list"], capture_output=True, text=True, timeout=2) 
        if p2.returncode == 0 and p2.stdout: 
            lines = [ln.strip() for ln in p2.stdout.splitlines() if ln.strip()] 
            out = [] 
            for ln in lines[1:]: 
                name = ln.split()[0] 
                if name: 
                    out.append(name)
            return list(dict.fromkeys(out)) 
    except Exception: 
        pass 
    return [] 
 
def _pick_default_ollama_model() -> str:
    env_choice = os.getenv("LANGCODE_OLLAMA_MODEL") or os.getenv("OLLAMA_MODEL")
    if env_choice:
        return env_choice

    names = _detect_ollama_models()
    if "llama3.1" in names:
        return "llama3.1"
    if names:
        return names[0]
    return "llama3.1"

def _chosen_ollama_model() -> str: 
    """Use user-selected model if provided, otherwise fall back to default pick.""" 
    env = os.getenv("LANGCODE_OLLAMA_MODEL") 
    if isinstance(env, str):
        return env.strip()
    return _pick_default_ollama_model()

def _cached_chat_model(provider: str, model_name: str, temperature: float = 0.2):
    key = (provider, model_name, temperature)
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        m = ChatAnthropic(model=model_name, temperature=temperature)
    elif provider == "gemini":
        gkey = os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY")
        if not gkey:
            raise RuntimeError(
                "Missing GOOGLE_API_KEY / GEMINI_API_KEY. "
                "Set one of these in your .env (same folder you chose as Project)."
            )
        from langchain_google_genai import ChatGoogleGenerativeAI
        m = ChatGoogleGenerativeAI(model=model_name, temperature=temperature, google_api_key=gkey, transport="rest")
    elif provider == "openai":
        from langchain_openai import ChatOpenAI
        m = ChatOpenAI(model=model_name, temperature=temperature)
    elif provider == "openrouter":
        api_key = os.getenv("OPENROUTER_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing OPENROUTER_API_KEY. "
                "Set this in your .env (same folder you chose as Project). "
                "Get your API key at https://openrouter.ai/"
            )

        base_url = (os.getenv("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1").rstrip("/")
        referer = (
            os.getenv("OPENROUTER_SITE_URL")
            or os.getenv("YOUR_SITE_URL")
        )
        title = (
            os.getenv("OPENROUTER_SITE_NAME")
            or os.getenv("YOUR_SITE_NAME")
        )
        headers: Dict[str, str] = {}
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title

        try:
            from langchain_openrouter import ChatOpenRouter  # type: ignore
        except Exception as exc:  # pragma: no cover - fallback when optional dep is incompatible
            logger.debug("ChatOpenRouter import failed: %s", exc)
            ChatOpenRouter = None  # type: ignore
        if ChatOpenRouter:
            extra_args: Dict[str, Any] = {}
            if base_url:
                extra_args["base_url"] = base_url
            if headers:
                extra_args["default_headers"] = headers
            try:
                m = ChatOpenRouter(
                    model=model_name,
                    temperature=temperature,
                    openrouter_api_key=api_key,
                    **extra_args,
                )
            except TypeError:
                # Older langchain-openrouter versions might not accept base_url/default_headers.
                extra_args.pop("base_url", None)
                extra_args.pop("default_headers", None)
                m = ChatOpenRouter(
                    model=model_name,
                    temperature=temperature,
                    openrouter_api_key=api_key,
                    **extra_args,
                )
        else:
            from langchain_openai import ChatOpenAI
            m = ChatOpenAI(
                model=model_name,
                temperature=temperature,
                api_key=api_key,
                base_url=base_url,
                default_headers=headers or None,
            )
    elif provider == "ollama": 
        from langchain_ollama import ChatOllama 
        base_url = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST") 
        if base_url: 
            m = ChatOllama(model=model_name, temperature=temperature, base_url=base_url, timeout=120, max_retries=2) 
        else: 
            m = ChatOllama(model=model_name, temperature=temperature)
    else:
        raise ValueError(f"Unknown provider: {provider}")
    _MODEL_CACHE[key] = m
    return m

def resolve_provider(cli_llm: str | None) -> str:
    if cli_llm:
        p = cli_llm.lower()
        if p in {"claude", "anthropic"}:
            return "anthropic"
        if p in {"gemini", "google"}:
            return "gemini"
        if p in {"openai", "gpt"}: 
            return "openai"
        if p in {"openrouter"}: 
            return "openrouter"
        if p in {"ollama"}: 
            return "ollama"
        return p

    env = os.getenv("LLM_PROVIDER", "gemini").lower()
    if env not in {"gemini", "anthropic", "openai", "openrouter", "ollama"}:
        env = "gemini"
    return env

def get_model(provider: str, query: Optional[str] = None, priority: str = "balanced"):
    """
    When no query is given (no router context), return a solid default:
      - anthropic    => 'claude-3-7-sonnet-20250219'
      - gemini       => 'gemini-2.0-flash'
      - openai       => 'gpt-4o-mini'
      - openrouter   => 'google/gemini-2.0-flash-lite'
      - ollama       => detected default (prefers llama3.1)
    With a query, use the speed-first router with reasoning override and cache the model object.
    """
    if not query:
        if provider == "anthropic":
            return _cached_chat_model("anthropic", "claude-3-7-sonnet-20250219", 0.2)
        elif provider == "gemini":
            return _cached_chat_model("gemini", "gemini-2.0-flash", 0.2)
        elif provider == "openai": 
            return _cached_chat_model("openai", "gpt-4o-mini", 0.2)
        elif provider == "openrouter": 
            return _cached_chat_model("openrouter", "google/gemini-2.0-flash-lite", 0.2)
        elif provider == "ollama": 
            return _cached_chat_model("ollama", _chosen_ollama_model(), 0.2)
        else:
            raise ValueError(f"Unknown provider: {provider}")

    optimal = _router.select_optimal_model_for_provider(query, provider, priority)
    if provider == "anthropic":
        return _cached_chat_model("anthropic", optimal.langchain_model_name, 0.2)
    elif provider == "gemini":
        return _cached_chat_model("gemini", optimal.langchain_model_name, 0.2)
    elif provider == "openai": 
        return _cached_chat_model("openai", optimal.langchain_model_name, 0.2)
    elif provider == "openrouter": 
        return _cached_chat_model("openrouter", optimal.langchain_model_name, 0.2)
    elif provider == "ollama": 
        name = os.getenv("LANGCODE_OLLAMA_MODEL") or _chosen_ollama_model()
        return _cached_chat_model("ollama", name, 0.2)
    else:
        raise ValueError(f"Unknown provider: {provider}")

def get_model_info(provider: str, query: Optional[str] = None, priority: str = "balanced") -> Dict[str, Any]:
    if not query:
        if provider == "anthropic":
            return {
                'model_name': 'Claude Sonnet (Default)',
                'langchain_model_name': 'claude-3-7-sonnet-20250219',
                'provider': provider,
                'complexity': 'default',
                'note': 'Using default model - no query provided for optimization'
            }
        elif provider == "gemini":
            return {
                'model_name': 'Gemini 2.0 Flash (Default)',
                'langchain_model_name': 'gemini-2.0-flash',
                'provider': provider,
                'complexity': 'default',
                'note': 'Using default model - no query provided for optimization'
            }
        elif provider == "openai": 
            return { 
                'model_name': 'GPT-4o Mini (Default)', 
                'langchain_model_name': 'gpt-4o-mini', 
                'provider': provider, 
                'complexity': 'default', 
                'note': 'Using default model - no query provided for optimization' 
            }
        elif provider == "openrouter": 
            return { 
                'model_name': 'Gemini 2.0 Flash Lite (Default)', 
                'langchain_model_name': 'google/gemini-2.0-flash-lite', 
                'provider': provider, 
                'complexity': 'default', 
                'note': 'Using default model - no query provided for optimization. Requires OPENROUTER_API_KEY.' 
            }
        elif provider == "ollama": 
            md = os.getenv("LANGCODE_OLLAMA_MODEL") or _pick_default_ollama_model()
            return { 
                'model_name': f'{md}',  
                'langchain_model_name': md, 
                'provider': provider, 
                'complexity': _router.classify_complexity(query),
                'priority_used': priority, 
                'note': 'Using selected/local Ollama model' 
            }
        else:
            raise ValueError(f"Unknown provider: {provider}")

    optimal = _router.select_optimal_model_for_provider(query, provider, priority)
    complexity = _router.classify_complexity(query)
    return {
        'model_name': optimal.name,
        'model_id': optimal.model_id,
        'langchain_model_name': optimal.langchain_model_name,
        'provider': provider,
        'complexity': complexity,
        'reasoning_strength': optimal.reasoning_strength,
        'latency_tier': optimal.latency_tier,
        'input_cost_per_million': optimal.input_cost_per_million,
        'output_cost_per_million': optimal.output_cost_per_million,
        'capabilities': optimal.capabilities,
        'context_window': optimal.context_window,
        'priority_used': priority
    }

def get_model_by_name(provider: str, model_name: str):
    """
    Return a cached chat model instance for an explicit model name.
    Minimal helper to support manual model selection from the launcher.
    """
    return _cached_chat_model(provider, model_name, 0.2)
