"""
End-to-End Live RAG Evaluation Runner.
Evaluates the complete RAG pipeline across both:
1. Retriever Quality:
   - Contextual Recall (Does retrieved context cover the ground truth?)
   - Contextual Precision (Are relevant chunks ranked higher?)
2. Generator Quality:
   - Faithfulness (Is the answer grounded in retrieved context?)
   - Answer Relevancy (Does the answer directly answer the query?)
   - Hallucination (Does the answer hallucinate beyond ground truth context?)
"""

import argparse
import json
from pathlib import Path
import sys

# Ensure UTF-8 output encoding for Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from components.generator import generate_answer
from components.retriever import build_retriever, retrieve_contexts
from deepeval.evaluate import evaluate
from deepeval.evaluate.configs import DisplayConfig
from deepeval.metrics import (
    AnswerRelevancyMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    FaithfulnessMetric,
    HallucinationMetric,
)
from deepeval.test_case import LLMTestCase


def run_rag_eval(
    doc_path: str = "docs/Facebooks-Corporate-Human-Rights-Policy.pdf",
    dataset_path: str = "evals/datasets/golden_dataset.json",
    vector_store_type: str = "chroma",
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    use_reranker: bool = False,
    top_k: int = 3,
    provider: str = "openai",
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    prompt_template: str = "default",
    max_cases: int | None = None,
    eval_model: str = "gpt-4o-mini",
):
    """
    End-to-end RAG evaluation:
    Runs the full live pipeline (Retriever + Generator) and benchmarks
    both Retriever and Generator performance.
    """
    full_doc_path = ROOT / doc_path if not Path(doc_path).is_absolute() else Path(doc_path)
    full_dataset_path = ROOT / dataset_path if not Path(dataset_path).is_absolute() else Path(dataset_path)

    print("\n==========================================")
    print("   Running End-to-End Live RAG Benchmark  ")
    print("==========================================")
    print(f"Document:           {full_doc_path.name}")
    print(f"Dataset:            {full_dataset_path.name}")
    print(f"Vector Store:       {vector_store_type}")
    print(f"Reranker:           {'Enabled (cross-encoder)' if use_reranker else 'Disabled'}")
    print(f"Top-K Chunks:       {top_k}")
    print(f"Generator Provider: {provider}")
    print(f"Generator Model:    {model_name} (temperature: {temperature})")
    print(f"Prompt Template:    {prompt_template}")
    print(f"Judge LLM Model:    {eval_model}\n")

    # 1. Build live retriever
    print("Building vector store and retriever...")
    retriever = build_retriever(
        doc_path=str(full_doc_path),
        vector_store_type=vector_store_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        k=top_k,
    )

    # 2. Load golden dataset
    with open(full_dataset_path, encoding="utf-8") as f:
        goldens = json.load(f)

    if max_cases:
        goldens = goldens[:max_cases]
        print(f"Limited evaluation to {max_cases} test cases.")

    # 3. Execute live RAG pipeline for each query
    test_cases: list[LLMTestCase] = []
    print(f"Running live RAG for {len(goldens)} test cases...")

    for i, item in enumerate(goldens, 1):
        query = item["input"]
        expected_output = item.get("expected_output")

        # Ground truth context from dataset
        golden_context = item.get("context", [])
        if isinstance(golden_context, str):
            golden_context = [golden_context]

        # Step 3a: Live retrieval
        retrieved_chunks = retrieve_contexts(
            retriever=retriever,
            query=query,
            use_reranker=use_reranker,
            top_k=top_k,
        )

        # Step 3b: Live answer generation
        answer = generate_answer(
            query=query,
            contexts=retrieved_chunks,
            provider=provider,
            model_name=model_name,
            temperature=temperature,
            prompt_template=prompt_template,
        )

        # Build test case with both ground truth context and live retrieved context
        test_cases.append(
            LLMTestCase(
                input=query,
                actual_output=answer,
                expected_output=expected_output,
                context=golden_context,               # Ground truth context (for Hallucination & Recall)
                retrieval_context=retrieved_chunks,   # Live retrieved context (for Faithfulness, Precision & Recall)
            )
        )
        print(f"  [{i}/{len(goldens)}] Retrieved {len(retrieved_chunks)} chunks & generated answer.")

    # 4. Comprehensive End-to-End Metrics (Both Retriever & Generator)
    metrics = [
        # --- Retriever Metrics ---
        ContextualRecallMetric(threshold=0.7, model=eval_model, include_reason=False),
        ContextualPrecisionMetric(threshold=0.7, model=eval_model, include_reason=False),
        # --- Generator Metrics ---
        FaithfulnessMetric(threshold=0.7, model=eval_model, include_reason=False),
        AnswerRelevancyMetric(threshold=0.7, model=eval_model, include_reason=False),
        HallucinationMetric(threshold=0.7, model=eval_model, include_reason=False),
    ]

    results_dir = ROOT / "evals" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    hyperparams = {
        "eval_type": "end_to_end_rag",
        "vector_store": vector_store_type,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "use_reranker": use_reranker,
        "top_k": top_k,
        "provider": provider,
        "model_name": model_name,
        "temperature": temperature,
        "prompt_template": prompt_template,
        "eval_model": eval_model,
    }

    display_cfg = DisplayConfig(
        results_folder=str(results_dir),
        print_results=True,
    )

    print("\nEvaluating Full Pipeline (Retriever + Generator) with DeepEval...")
    return evaluate(
        test_cases=test_cases,
        metrics=metrics,
        hyperparameters=hyperparams,
        display_config=display_cfg,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate End-to-End RAG pipeline performance.")
    parser.add_argument("--doc", type=str, default="docs/Facebooks-Corporate-Human-Rights-Policy.pdf", help="Document path")
    parser.add_argument("--dataset", type=str, default="evals/datasets/golden_dataset.json", help="Dataset path")
    parser.add_argument("--vector-store", type=str, default="chroma", help="Vector store (chroma, faiss, qdrant)")
    parser.add_argument("--rerank", action="store_true", help="Enable cross-encoder reranking")
    parser.add_argument("--top-k", type=int, default=3, help="Number of retrieved chunks")
    parser.add_argument("--provider", type=str, default="openai", help="LLM provider (openai, deepseek, ollama)")
    parser.add_argument("--model", type=str, default="gpt-4o-mini", help="LLM generator model")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--prompt", type=str, default="default", help="Prompt template (default, concise, reasoning)")
    parser.add_argument("--eval-model", type=str, default="gpt-4o-mini", help="Judge model used for evaluation")
    parser.add_argument("--max-cases", type=int, default=None, help="Limit number of test cases to evaluate")

    args = parser.parse_args()

    run_rag_eval(
        doc_path=args.doc,
        dataset_path=args.dataset,
        vector_store_type=args.vector_store,
        use_reranker=args.rerank,
        top_k=args.top_k,
        provider=args.provider,
        model_name=args.model,
        temperature=args.temperature,
        prompt_template=args.prompt,
        max_cases=args.max_cases,
        eval_model=args.eval_model,
    )