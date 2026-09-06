import os
from contextlib import contextmanager
from dotenv import load_dotenv
from langchain_community.callbacks import get_openai_callback

load_dotenv()


class NullCostTracker:
    """Dummy tracker returned when cost tracking is disabled."""
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_cost: float = 0.0
    successful_requests: int = 0
    enabled: bool = False

    def summary(self) -> str:
        return "Cost tracking is disabled."


@contextmanager
def track_cost(enabled: bool | None = None):
    """
    Decoupled context manager for tracking LLM token costs.

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
        with get_openai_callback() as cb:
            # Attach enabled flag for clean reporting
            setattr(cb, "enabled", True)
            yield cb


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
    print("Testing cost tracker in isolation...")

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

