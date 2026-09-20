"""
Centralized Metric Registry for RAG Evaluation.
Provides a scalable catalog of retriever, generator, operational, and custom metrics.
Allows dynamic discovery by the API and Web UI.
"""

from typing import Any
def _make_deepeval_metric(cls_name: str, **kwargs):
    import deepeval.metrics as dm
    cls = getattr(dm, cls_name)
    return cls(**kwargs)


# Metric Definitions with metadata and lazy factory instantiation
METRIC_REGISTRY: dict[str, dict[str, Any]] = {
    # --- Retriever Metrics ---
    "contextual_recall": {
        "id": "contextual_recall",
        "name": "Contextual Recall",
        "category": "retriever",
        "category_label": "Retriever Metrics",
        "description": "Measures whether retrieved context contains all facts needed to answer the query compared to ground truth.",
        "default_checked": True,
        "factory": lambda model: _make_deepeval_metric("ContextualRecallMetric", threshold=0.7, model=model, include_reason=False),
    },
    "contextual_precision": {
        "id": "contextual_precision",
        "name": "Contextual Precision",
        "category": "retriever",
        "category_label": "Retriever Metrics",
        "description": "Measures whether the most relevant chunks are ranked at the top of the context window.",
        "default_checked": True,
        "factory": lambda model: _make_deepeval_metric("ContextualPrecisionMetric", threshold=0.7, model=model, include_reason=False),
    },

    # --- Generator Metrics ---
    "faithfulness": {
        "id": "faithfulness",
        "name": "Faithfulness",
        "category": "generator",
        "category_label": "Generator Metrics",
        "description": "Verifies that the generated answer is strictly grounded in the retrieved chunks without unsupported claims.",
        "default_checked": True,
        "factory": lambda model: _make_deepeval_metric("FaithfulnessMetric", threshold=0.7, model=model, include_reason=False),
    },
    "answer_relevancy": {
        "id": "answer_relevancy",
        "name": "Answer Relevancy",
        "category": "generator",
        "category_label": "Generator Metrics",
        "description": "Evaluates how directly and concisely the answer addresses the user's query.",
        "default_checked": True,
        "factory": lambda model: _make_deepeval_metric("AnswerRelevancyMetric", threshold=0.7, model=model, include_reason=False),
    },
    "hallucination": {
        "id": "hallucination",
        "name": "Hallucination",
        "category": "generator",
        "category_label": "Generator Metrics",
        "description": "Detects if the generator introduces external or contradictory facts beyond the ground truth context.",
        "default_checked": True,
        "factory": lambda model: _make_deepeval_metric("HallucinationMetric", threshold=0.7, model=model, include_reason=False),
    },

    # --- Safety & Robustness Metrics (Opt-in) ---
    "toxicity": {
        "id": "toxicity",
        "name": "Toxicity",
        "category": "safety",
        "category_label": "Safety & Robustness",
        "description": "Measures whether the generated output contains toxic, hateful, or harmful statements.",
        "default_checked": False,
        "factory": lambda model: _make_deepeval_metric("ToxicityMetric", threshold=0.7, model=model, include_reason=False),
    },
    "bias": {
        "id": "bias",
        "name": "Bias",
        "category": "safety",
        "category_label": "Safety & Robustness",
        "description": "Measures whether the generated output contains unfair stereotyping, bias, or prejudice.",
        "default_checked": False,
        "factory": lambda model: _make_deepeval_metric("BiasMetric", threshold=0.7, model=model, include_reason=False),
    },

    # --- Operational & Cost Metrics (Always available across all scopes) ---
    "ops_latency": {
        "id": "ops_latency",
        "name": "Latency Breakdown (P50/P95)",
        "category": "ops",
        "category_label": "Operations & Cost",
        "description": "Tracks retrieval latency, generation latency, and end-to-end execution time.",
        "default_checked": True,
        "factory": None,  # Computed natively by the ops profiler
    },
    "ops_cost": {
        "id": "ops_cost",
        "name": "Token Usage & Projected Cost",
        "category": "ops",
        "category_label": "Operations & Cost",
        "description": "Tracks input/output token counts and calculates estimated USD cost per 1k queries.",
        "default_checked": True,
        "factory": None,  # Computed natively by the cost tracker
    },
}


def get_metrics_catalog() -> list[dict[str, Any]]:
    """
    Returns a lightweight catalog of all registered metrics grouped for UI and API consumption.
    Omits Python callables/factories so the result is 100% JSON serializable.
    """
    catalog = []
    for metric_id, info in METRIC_REGISTRY.items():
        catalog.append({
            "id": info["id"],
            "name": info["name"],
            "category": info["category"],
            "category_label": info["category_label"],
            "description": info["description"],
            "default_checked": info["default_checked"],
        })
    return catalog


def build_selected_metrics(
    selected_ids: list[str] | None,
    eval_model: str = "gpt-4o-mini",
) -> tuple[list[Any], set[str]]:
    """
    Builds and instantiates selected DeepEval metrics and returns active ops flags.
    Returns:
        (deepeval_metrics, active_ops_metric_ids)
    """
    if selected_ids is None:
        selected_ids = list(METRIC_REGISTRY.keys())

    deepeval_metrics = []
    active_ops = set()

    for metric_id in selected_ids:
        clean_id = metric_id.lower().strip()
        if clean_id not in METRIC_REGISTRY:
            continue

        info = METRIC_REGISTRY[clean_id]
        if info["category"] == "ops":
            active_ops.add(clean_id)
        elif info["factory"] is not None:
            deepeval_metrics.append(info["factory"](eval_model))

    return deepeval_metrics, active_ops

