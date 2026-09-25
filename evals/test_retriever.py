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
from deepeval.metrics import ContextualPrecisionMetric, ContextualRecallMetric
from deepeval.test_case import LLMTestCase

from components.retriever import build_retriever, retrieve_contexts


def run_retriever_eval(
    doc_path: str = "docs/Facebooks-Corporate-Human-Rights-Policy.pdf",
    dataset_path: str = "evals/datasets/golden_dataset.json",
    vector_store_type: str = "chroma",
    loader_type: int | str = "auto",
    splitter_type: str = "recursive",
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    use_reranker: bool = False,
    top_k: int = 3,
    eval_model: str = "gpt-4o-mini",
) -> Any:
    """
    Evaluates Retriever Contextual Recall and Contextual Precision
    without calling the LLM generator.
    """
    doc_p = Path(doc_path)
    full_doc_path = doc_p if doc_p.is_absolute() else ROOT / doc_p
    dataset_p = Path(dataset_path)
    full_dataset_path = dataset_p if dataset_p.is_absolute() else ROOT / dataset_p

    if not full_dataset_path.exists():
        raise FileNotFoundError(f"Golden dataset not found at: {full_dataset_path}")

    print("\n--- Running Retriever Evaluation ---")
    print(f"Document: {full_doc_path.name}")
    print(f"Loader: {loader_type} | Splitter: {splitter_type}")
    print(f"Vector Store: {vector_store_type}")
    print(f"Chunk Size: {chunk_size} (overlap: {chunk_overlap})")
    print(f"Reranker: {'Enabled (cross-encoder)' if use_reranker else 'Disabled'}")
    print(f"Top-K Chunks: {top_k}\n")

    # 1. Build retriever
    print("Building vector store and retriever...")
    retriever = build_retriever(
        doc_path=str(full_doc_path),
        vector_store_type=vector_store_type,
        loader_type=loader_type,
        splitter_type=splitter_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        k=top_k,
    )

    # 2. Load golden dataset
    with open(full_dataset_path, encoding="utf-8") as f:
        goldens = json.load(f)

    if not goldens:
        print("Warning: No test cases found to evaluate.")
        return None

    test_cases = []

    # 3. Retrieve context for each golden question
    print(f"Retrieving contexts for {len(goldens)} test cases...")
    for item in goldens:
        query = item.get("input") or item.get("question") or item.get("query")
        if not query:
            continue
        expected = item.get("expected_output") or item.get("ground_truth")

        contexts = retrieve_contexts(
            retriever=retriever,
            query=query,
            use_reranker=use_reranker,
            top_k=top_k,
        )

        test_cases.append(
            LLMTestCase(
                input=query,
                actual_output="N/A",  # Not required for retriever evaluation
                expected_output=expected,
                retrieval_context=contexts,
            )
        )

    if not test_cases:
        print("Warning: No valid test cases created.")
        return None

    # 4. Metrics: Contextual Recall & Contextual Precision
    metrics = [
        ContextualRecallMetric(threshold=0.7, model=eval_model, include_reason=False),
        ContextualPrecisionMetric(threshold=0.7, model=eval_model, include_reason=False),
    ]

    results_dir = ROOT / "evals" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    hyperparams = {
        "document": full_doc_path.name,
        "eval_model": eval_model,
        "vector_store": vector_store_type,
        "loader_type": str(loader_type),
        "splitter_type": splitter_type,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "use_reranker": use_reranker,
        "top_k": top_k,
    }

    display_cfg = DisplayConfig(
        results_folder=str(results_dir),
        print_results=True,
    )

    print("\nEvaluating Contextual Recall & Contextual Precision with DeepEval...")
    return evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters=hyperparams,
        display_config=display_cfg,
    )


if __name__ == "__main__":
    run_retriever_eval()
