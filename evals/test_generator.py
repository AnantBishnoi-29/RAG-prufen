import json
import sys
from pathlib import Path
from typing import Any

# Ensure UTF-8 output encoding for Windows consoles
if sys.platform == "win32":
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from deepeval.evaluate import evaluate
from deepeval.evaluate.configs import DisplayConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase

from components.generator import generate_answer


def run_generator_eval(
    dataset_path: str = "evals/datasets/golden_dataset.json",
    provider: str = "openai",
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    prompt_template: str = "default",
    system_prompt: str | None = None,
    max_cases: int | None = None,
    eval_model: str = "gpt-4o-mini",
) -> Any:
    """
    Evaluates generator quality (Faithfulness, Answer Relevancy, Hallucination)
    in strict isolation using ground-truth contexts from the golden dataset.
    """
    p = Path(dataset_path)
    full_dataset_path = p if p.is_absolute() else ROOT / p

    if not full_dataset_path.exists():
        raise FileNotFoundError(f"Golden dataset not found at: {full_dataset_path}")

    print("\n--- Running Generator Isolation Evaluation ---")
    print(f"Provider: {provider}")
    print(f"Model: {model_name}")
    print(f"Temperature: {temperature}")
    print(f"Prompt Template: {prompt_template}")
    print("Mode: Isolated Context (Ground Truth)\n")

    # 1. Load dataset
    with open(full_dataset_path, encoding="utf-8") as f:
        goldens = json.load(f)

    if max_cases is not None:
        goldens = goldens[:max(0, max_cases)]
        print(f"Limited evaluation to {len(goldens)} test cases.")

    if not goldens:
        print("Warning: No test cases found to evaluate.")
        return None

    # 2. Generate answers using isolated golden contexts
    test_cases = []
    print(f"Generating answers for {len(goldens)} test cases...")

    for i, item in enumerate(goldens, 1):
        query = item.get("input") or item.get("question") or item.get("query")
        if not query:
            print(f"  [{i}/{len(goldens)}] Skipping: missing input/question.")
            continue

        expected_output = item.get("expected_output")
        contexts = item.get("context") or item.get("contexts") or []
        if isinstance(contexts, str):
            contexts = [contexts]

        answer = generate_answer(
            query=query,
            contexts=contexts,
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            prompt_template=prompt_template,
            system_prompt=system_prompt,
        )

        test_cases.append(
            LLMTestCase(
                input=query,
                actual_output=answer,
                expected_output=expected_output,
                context=contexts,
                retrieval_context=contexts,
            )
        )
        print(f"  [{i}/{len(goldens)}] Answer generated.")

    if not test_cases:
        print("Warning: No valid test cases created.")
        return None

    # 3. Generator Metrics
    metrics = [
        FaithfulnessMetric(threshold=0.7, model=eval_model, include_reason=False),
        AnswerRelevancyMetric(threshold=0.7, model=eval_model, include_reason=False),
        HallucinationMetric(threshold=0.7, model=eval_model, include_reason=False),
    ]

    results_dir = ROOT / "evals" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    hyperparams = {
        "eval_type": "generator_isolation",
        "provider": provider,
        "model_name": model_name,
        "temperature": temperature,
        "prompt_template": prompt_template,
        "system_prompt": system_prompt,
        "eval_model": eval_model,
    }

    display_cfg = DisplayConfig(
        results_folder=str(results_dir),
        print_results=True,
    )

    print("\nEvaluating Faithfulness, Relevancy, and Hallucination with DeepEval...")
    return evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters=hyperparams,
        display_config=display_cfg,
    )


if __name__ == "__main__":
    run_generator_eval(max_cases=2)
