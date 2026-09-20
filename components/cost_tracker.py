"""
Universal Cost and Token Tracker
Provider-agnostic token and cost tracking for LangChain models using standardized usage metadata.
"""

from contextlib import contextmanager
import os
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks import UsageMetadataCallbackHandler, get_usage_metadata_callback

load_dotenv()

# Pricing in USD per 1,000,000 tokens (input/prompt, output/completion)
# Source: Official provider documentation
MODEL_PRICING_PER_1M: dict[str, dict[str, float]] = {
    # OpenAI
    "gpt-4o-mini": {"prompt": 0.15, "completion": 0.60},
    "gpt-4o": {"prompt": 2.50, "completion": 10.00},
    "gpt-4-turbo": {"prompt": 10.00, "completion": 30.00},
    "gpt-3.5-turbo": {"prompt": 0.50, "completion": 1.50},
    "text-embedding-3-small": {"prompt": 0.02, "completion": 0.0},
    "text-embedding-3-large": {"prompt": 0.13, "completion": 0.0},
    # DeepSeek
    "deepseek-chat": {"prompt": 0.14, "completion": 0.28},
    "deepseek-reasoner": {"prompt": 0.55, "completion": 2.19},
    # Anthropic
    "claude-3-5-sonnet": {"prompt": 3.00, "completion": 15.00},
    "claude-3-haiku": {"prompt": 0.25, "completion": 1.25},
    # Local / Ollama (Free)
    "ollama": {"prompt": 0.0, "completion": 0.0},
    "llama": {"prompt": 0.0, "completion": 0.0},
    "mistral": {"prompt": 0.0, "completion": 0.0},
    "local": {"prompt": 0.0, "completion": 0.0},
}


def get_model_pricing(model_name: str) -> tuple[float, float]:
    """
    Returns (prompt_rate_per_1m, completion_rate_per_1m) for a model name.
    Matches substring keys ordered by length descending to match specific prefixes first.
    """
    if not model_name:
        return 0.0, 0.0

    name_clean = model_name.lower().strip()
    # Sort keys by length descending (e.g. 'gpt-4o-mini' matches before 'gpt-4o')
    sorted_keys = sorted(MODEL_PRICING_PER_1M.keys(), key=len, reverse=True)
    for key in sorted_keys:
        if key in name_clean:
            pricing = MODEL_PRICING_PER_1M[key]
            return pricing["prompt"], pricing["completion"]

    return 0.0, 0.0


class NullCostTracker:
    """Dummy tracker returned when cost tracking is disabled."""

    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost: float = 0.0
    successful_requests: int = 0
    enabled: bool = False
    usage_metadata: dict[str, dict[str, Any]] = {}


class UniversalCostTracker:
    """Tracks token usage and calculates USD costs across any LLM provider."""

    def __init__(
        self,
        callback_handler: UsageMetadataCallbackHandler | None = None,
        enabled: bool = True,
    ) -> None:
        self._cb = callback_handler
        self.enabled = enabled

    @property
    def usage_metadata(self) -> dict[str, dict[str, Any]]:
        """Raw model-by-model token usage dictionary."""
        if not self._cb or not hasattr(self._cb, "usage_metadata"):
            return {}
        return self._cb.usage_metadata

    @property
    def prompt_tokens(self) -> int:
        """Total input/prompt tokens across all models invoked."""
        return sum(m.get("input_tokens", 0) for m in self.usage_metadata.values())

    @property
    def completion_tokens(self) -> int:
        """Total output/completion tokens across all models invoked."""
        return sum(m.get("output_tokens", 0) for m in self.usage_metadata.values())

    @property
    def total_tokens(self) -> int:
        """Total tokens across all models invoked."""
        return sum(m.get("total_tokens", 0) for m in self.usage_metadata.values())

    @property
    def total_cost(self) -> float:
        """Calculated total cost in USD based on model pricing table."""
        cost = 0.0
        for model_name, usage in self.usage_metadata.items():
            prompt_rate, completion_rate = get_model_pricing(model_name)
            p_tok = usage.get("input_tokens", 0)
            c_tok = usage.get("output_tokens", 0)
            cost += (p_tok * prompt_rate / 1_000_000.0) + (c_tok * completion_rate / 1_000_000.0)
        return cost

    @property
    def successful_requests(self) -> int:
        """Count of distinct models or successful request records tracked."""
        return len(self.usage_metadata)


@contextmanager
def track_cost(enabled: bool | None = None):
    """
    Decoupled context manager for tracking LLM token costs across all providers.

    Toggling:
    - Pass `enabled=True` or `enabled=False` directly, OR
    - Set `TRACK_COSTS=true` / `false` in your `.env` file (defaults to True).
    """
    if enabled is None:
        # Check .env (defaults to True if not explicitly set to 'false' or '0')
        env_val = os.getenv("TRACK_COSTS", "true").lower().strip()
        enabled = env_val not in ("false", "0", "no", "off")

    if not enabled:
        # Zero overhead: does not hook into LangChain callbacks
        yield NullCostTracker()
    else:
        with get_usage_metadata_callback() as cb:
            tracker = UniversalCostTracker(callback_handler=cb, enabled=True)
            yield tracker


def format_cost_summary(cb) -> str:
    """Helper to format cost results into a clean 1-line string."""
    if not getattr(cb, "enabled", False):
        return "[Cost Tracking: Disabled]"
    return (
        f"Tokens: {cb.total_tokens:,} "
        f"(Prompt: {cb.prompt_tokens:,}, Completion: {cb.completion_tokens:,}) | "
        f"Cost: ${cb.total_cost:.6f}"
    )


def main():
    print("Testing universal cost tracker in isolation...")

    # Test 1: Disabled mode
    with track_cost(enabled=False) as tracker:
        pass
    print("Disabled Test:", format_cost_summary(tracker))

    # Test 2: Enabled mode (with no LLM calls)
    with track_cost(enabled=True) as tracker:
        pass
    print("Enabled Test (0 calls):", format_cost_summary(tracker))


if __name__ == "__main__":
    main()
