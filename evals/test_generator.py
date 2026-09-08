import argparse
import json
import sys
from pathlib import Path

# Ensure UTF-8 output encoding for Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
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
    max_cases: int | None = None,
):
    """
    Evaluates generator quality (Faithfulness, Answer Relevancy, Hallucination)
    in strict isolation using ground-truth contexts from the golden dataset.
    """
    full_dataset_path = ROOT / dataset_path if not Path(dataset_path).is_absolute() else Path(dataset_path)

    print("\n--- Running Generator Isolation Evaluation ---")
    print(f"Provider: {provider}")
    print(f"Model: {model_name}")
    print(f"Temperature: {temperature}")
    print(f"Prompt Template: {prompt_template}")
    print("Mode: Isolated Context (Ground Truth)\n")

    # 1. Load dataset
    with open(full_dataset_path, encoding="utf-8") as f:
        goldens = json.load(f)

    if max_cases:
        goldens = goldens[:max_cases]
        print(f"Limited evaluation to {max_cases} test cases.")

    # 2. Generate answers using isolated golden contexts
    test_cases = []
    print(f"Generating answers for {len(goldens)} test cases...")

    for i, item in enumerate(goldens, 1):
        query = item["input"]
        expected_output = item.get("expected_output")
        contexts = item.get("context", [])

        answer = generate_answer(
            query=query,
            contexts=contexts,
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            prompt_template=prompt_template,
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

    # 3. Generator Metrics
    metrics = [
        FaithfulnessMetric(threshold=0.7, model="gpt-4o-mini", include_reason=False),
        AnswerRelevancyMetric(threshold=0.7, model="gpt-4o-mini", include_reason=False),
        HallucinationMetric(threshold=0.7, model="gpt-4o-mini", include_reason=False),
    ]

    results_dir = ROOT / "evals" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    hyperparams = {
        "eval_type": "generator_isolation",
        "provider": provider,
        "model_name": model_name,
        "temperature": temperature,
        "prompt_template": prompt_template,
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
    parser = argparse.ArgumentParser(description="Evaluate RAG Generator performance in isolation.")
    parser.add_argument("--provider", type=str, default="openai", help="LLM provider (openai, deepseek, ollama)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="Model name")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--prompt", type=str, default="default", help="Prompt template name (default, concise, reasoning)")
    parser.add_argument("--max-cases", type=int, default=None, help="Limit number of test cases to evaluate")

    args = parser.parse_args()

    run_generator_eval(
        provider=args.provider,
        model_name=args.model,
        temperature=args.temperature,
        prompt_template=args.prompt,
        max_cases=args.max_cases,
    )

