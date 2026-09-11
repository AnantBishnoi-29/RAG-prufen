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
Supports full pipeline benchmark, retriever-only isolation, or generator-only isolation,
with granular metric selection driven by evals.registry.
"""

import json
from pathlib import Path
import statistics
import sys
import time

# Ensure UTF-8 output encoding for Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure project root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from components.cost_tracker import track_cost
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
from evals.registry import build_selected_metrics


def calculate_percentile(data: list[float], p: float) -> float:
    """Calculates the p-th percentile of a list of floats (0 to 100)."""
    if not data:
        return 0.0
    sorted_data = sorted(data)
    k = (len(sorted_data) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(sorted_data) - 1)
    d = k - f
    return round(sorted_data[f] + (sorted_data[c] - sorted_data[f]) * d, 2)


def run_rag_eval(
    doc_path: str = "docs/Facebooks-Corporate-Human-Rights-Policy.pdf",
    dataset_path: str = "evals/datasets/golden_dataset.json",
    vector_store_type: str = "chroma",
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    embedding_provider: str = "openai",
    splitter_type: str = "recursive",
    use_reranker: bool = False,
    use_hybrid: bool = False,
    top_k: int = 3,
    provider: str = "openai",
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    prompt_template: str = "default",
    max_cases: int | None = None,
    eval_model: str = "gpt-4o-mini",
    scope: str = "all",  # "all", "retriever_only", "generator_only"
    selected_metrics: list[str] | None = None,
) -> dict:
    """
    Executes selective RAG evaluation:
    - scope="all": Full end-to-end evaluation (both retrieval and answer generation).
    - scope="retriever_only": Evaluates retriever metrics (Contextual Recall/Precision) and skips generation.
    - scope="generator_only": Evaluates generator metrics (Faithfulness, Relevancy, Hallucination).
    - scope="generator_only": Evaluates generator metrics (Faithfulness, Relevancy, Hallucination, Safety).
    - selected_metrics: List of metric IDs to execute from evals.registry.
    """
    full_doc_path = ROOT / doc_path if not Path(doc_path).is_absolute() else Path(doc_path)
    full_dataset_path = ROOT / dataset_path if not Path(dataset_path).is_absolute() else Path(dataset_path)

    # 1. Build metrics from registry
    deepeval_metrics, active_ops = build_selected_metrics(
        selected_ids=selected_metrics,
        eval_model=eval_model,
    )

    # Determine if answer generation is required
    generator_metric_names = {"FaithfulnessMetric", "AnswerRelevancyMetric", "HallucinationMetric"}
    # Determine if answer generation is required (all generator and safety metrics require model output)
    generator_metric_names = {
        "FaithfulnessMetric",
        "AnswerRelevancyMetric",
        "HallucinationMetric",
        "ToxicityMetric",
        "BiasMetric",
    }
    has_generator_metrics = any(m.__class__.__name__ in generator_metric_names for m in deepeval_metrics)
    needs_generation = (scope != "retriever_only") and has_generator_metrics

    print("\n==========================================")
    print(f"   Running RAG Evaluation [{scope.upper()}]  ")
    print("==========================================")
    print(f"Document:           {full_doc_path.name}")
    print(f"Dataset:            {full_dataset_path.name}")
    print(f"Evaluation Scope:   {scope}")
    print(f"Needs Generation:   {needs_generation} ({'Calling LLM' if needs_generation else 'Skipping LLM'})")
    print(f"Active DeepEval:    {[m.__class__.__name__ for m in deepeval_metrics]}")
    print(f"Active Ops Metrics: {list(active_ops)}\n")
    print(f"\n[RAG Eval] Scope: {scope.upper()} | Doc: {full_doc_path.name} | Dataset: {full_dataset_path.name}")
    print(f"[RAG Eval] LLM Generation: {'Active' if needs_generation else 'Bypassed'} | DeepEval: {[m.__class__.__name__ for m in deepeval_metrics]}")

    # 2. Build retriever (needed for all and retriever_only)
    retriever = None
    if scope != "generator_only":
        print("Building vector store and retriever...")
        retriever = build_retriever(
            doc_path=str(full_doc_path),
            splitter_type=splitter_type,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            embedding_provider=embedding_provider,
            vector_store_type=vector_store_type,
            k=top_k,
            use_hybrid=use_hybrid,
        )

    # 3. Load golden dataset
    with open(full_dataset_path, encoding="utf-8") as f:
        goldens = json.load(f)

    if max_cases:
        goldens = goldens[:max_cases]
        print(f"Limited evaluation to {max_cases} test cases.")

    # 4. Execute pipeline per test case
    test_cases: list[LLMTestCase] = []
    print(f"Processing {len(goldens)} test cases...")
    per_query_ops = []
    retrieval_latencies = []
    generation_latencies = []
    total_latencies = []
    all_throughputs = []
    total_prompt_tokens = 0
    total_completion_tokens = 0
    grand_total_tokens = 0
    grand_total_cost = 0.0

    print(f"Processing {len(goldens)} test cases with live ops tracking...")

    for i, item in enumerate(goldens, 1):
        query = item["input"]
        expected_output = item.get("expected_output")

        # Ground truth context from dataset
        golden_context = item.get("context", [])
        if isinstance(golden_context, str):
            golden_context = [golden_context]

        # Step A: Retrieval
        t_ret_start = time.perf_counter()
        if retriever is not None:
            retrieved_chunks = retrieve_contexts(
                retriever=retriever,
                query=query,
                use_reranker=use_reranker,
                top_k=top_k,
            )
        else:
            # In generator isolation, test against golden context directly
            retrieved_chunks = golden_context
        retrieval_sec = time.perf_counter() - t_ret_start
        retrieval_ms = round(retrieval_sec * 1000, 2)
        retrieval_latencies.append(retrieval_ms)

        # Step B: Generation (conditionally skipped if retriever_only)
        if needs_generation:
            with track_cost() as cost_tracker:
                t_gen_start = time.perf_counter()
                answer = generate_answer(
                    query=query,
                    contexts=retrieved_chunks,
                    provider=provider,
                    model_name=model_name,
                    temperature=temperature,
                    prompt_template=prompt_template,
                )
                gen_sec = time.perf_counter() - t_gen_start
            gen_ms = round(gen_sec * 1000, 2)
            p_tok = getattr(cost_tracker, "prompt_tokens", 0)
            c_tok = getattr(cost_tracker, "completion_tokens", 0)
            tot_tok = getattr(cost_tracker, "total_tokens", 0)
            c_usd = getattr(cost_tracker, "total_cost", 0.0)
        else:
            answer = "N/A"
            gen_sec = 0.0
            gen_ms = 0.0
            p_tok = 0
            c_tok = 0
            tot_tok = 0
            c_usd = 0.0

        generation_latencies.append(gen_ms)
        tot_ms = round((retrieval_sec + gen_sec) * 1000, 2)
        total_latencies.append(tot_ms)

        total_prompt_tokens += p_tok
        total_completion_tokens += c_tok
        grand_total_tokens += tot_tok
        grand_total_cost += c_usd

        throughput = round(c_tok / gen_sec, 2) if gen_sec > 0 else 0.0
        if throughput > 0:
            all_throughputs.append(throughput)

        per_query_ops.append({
            "query_index": i,
            "query": query,
            "retrieval_ms": retrieval_ms,
            "generation_ms": gen_ms,
            "total_ms": tot_ms,
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "total_tokens": tot_tok,
            "throughput_tokens_per_sec": throughput,
            "cost_usd": round(c_usd, 6),
        })

        test_cases.append(
            LLMTestCase(
                input=query,
                actual_output=answer,
                expected_output=expected_output,
                context=golden_context,
                retrieval_context=retrieved_chunks,
            )
        )
        gen_status = "Generated answer" if needs_generation else "Skipped answer generation"
        print(f"  [{i}/{len(goldens)}] Retrieved {len(retrieved_chunks)} chunk(s) | {gen_status}")
        gen_status = f"Generated in {gen_ms:.0f}ms (${c_usd:.5f})" if needs_generation else "Skipped LLM"
        print(f"  [{i}/{len(goldens)}] Ret: {retrieval_ms:.0f}ms | {gen_status}")

    # 5. Execute Evaluation with DeepEval
    # 5. Compute Aggregated Ops & Latency Metrics
    n_queries = len(goldens)
    avg_cost_per_query = grand_total_cost / n_queries if n_queries else 0.0
    projected_cost_1k = avg_cost_per_query * 1000.0

    latency_metrics = {
        "retrieval_ms": {
            "mean": round(statistics.mean(retrieval_latencies), 2) if retrieval_latencies else 0.0,
            "p50": calculate_percentile(retrieval_latencies, 50),
            "p95": calculate_percentile(retrieval_latencies, 95),
            "min": min(retrieval_latencies) if retrieval_latencies else 0.0,
            "max": max(retrieval_latencies) if retrieval_latencies else 0.0,
        },
        "generation_ms": {
            "mean": round(statistics.mean(generation_latencies), 2) if generation_latencies else 0.0,
            "p50": calculate_percentile(generation_latencies, 50),
            "p95": calculate_percentile(generation_latencies, 95),
            "min": min(generation_latencies) if generation_latencies else 0.0,
            "max": max(generation_latencies) if generation_latencies else 0.0,
        },
        "total_latency_ms": {
            "mean": round(statistics.mean(total_latencies), 2) if total_latencies else 0.0,
            "p50": calculate_percentile(total_latencies, 50),
            "p95": calculate_percentile(total_latencies, 95),
            "min": min(total_latencies) if total_latencies else 0.0,
            "max": max(total_latencies) if total_latencies else 0.0,
        },
    }

    token_metrics = {
        "total_prompt_tokens": total_prompt_tokens,
        "total_completion_tokens": total_completion_tokens,
        "grand_total_tokens": grand_total_tokens,
        "avg_tokens_per_query": round(grand_total_tokens / n_queries, 1) if n_queries else 0.0,
        "avg_throughput_tokens_per_sec": round(statistics.mean(all_throughputs), 2) if all_throughputs else 0.0,
    }

    cost_metrics = {
        "total_cost_usd": round(grand_total_cost, 6),
        "avg_cost_per_query_usd": round(avg_cost_per_query, 6),
        "projected_cost_per_1k_queries_usd": round(projected_cost_1k, 4),
    }

    # 6. Execute Evaluation with DeepEval or Export
    results_dir = ROOT / "evals" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    existing_runs = set(results_dir.glob("*.json"))

    hyperparams = {
        "eval_type": f"{scope}_eval",
        "eval_type": f"rag_eval_{scope}",
        "scope": scope,
        "vector_store": vector_store_type if scope != "generator_only" else "none",
        "vector_store_type": vector_store_type,
        "embedding_provider": embedding_provider,
        "splitter_type": splitter_type,
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "use_reranker": use_reranker,
        "use_hybrid": use_hybrid,
        "top_k": top_k,
        "provider": provider if needs_generation else "none",
        "model_name": model_name if needs_generation else "none",
        "temperature": temperature,
        "prompt_template": prompt_template,
        "eval_model": eval_model,
        "metrics_count": len(deepeval_metrics),
    }

    display_cfg = DisplayConfig(
        results_folder=str(results_dir),
        print_results=True,
    )

    if deepeval_metrics:
        print("\nEvaluating with DeepEval...")
        eval_result = evaluate(
            test_cases=test_cases,
            metrics=deepeval_metrics,
            hyperparameters=hyperparams,
            display_config=display_cfg,
        )

        # Locate the newly generated DeepEval result file and enrich with Ops metrics
        new_runs = set(results_dir.glob("*.json")) - existing_runs
        target_file = next(iter(new_runs)) if new_runs else None
        if not target_file:
            all_runs = sorted(results_dir.glob("test_run_*.json"), key=lambda x: x.stat().st_mtime, reverse=True)
            target_file = all_runs[0] if all_runs else None

        if target_file and target_file.exists():
            try:
                with open(target_file, "r", encoding="utf-8") as rf:
                    run_data = json.load(rf)
                run_data["latency_metrics"] = latency_metrics
                run_data["token_metrics"] = token_metrics
                run_data["cost_metrics"] = cost_metrics
                run_data["per_query_ops"] = per_query_ops
                with open(target_file, "w", encoding="utf-8") as wf:
                    json.dump(run_data, wf, indent=4)
                print(f"[Ops Attached] Enriched {target_file.name} with latency and cost metrics.")
            except Exception as ex:
                print(f"[Warning] Could not enrich run file with ops data: {ex}")

        return eval_result
    else:
        print("\nNo DeepEval metrics selected. Saving ops & test case execution summary.")
        timestamp_slug = int(time.time())
        summary_file = results_dir / f"test_run_{timestamp_slug}.json"
        with open(summary_file, "w", encoding="utf-8") as f:
            json.dump({
                "testCases": [tc.dict() for tc in test_cases],
                "hyperparameters": hyperparams,
                "latency_metrics": latency_metrics,
                "token_metrics": token_metrics,
                "cost_metrics": cost_metrics,
                "per_query_ops": per_query_ops,
            }, f, indent=4)
        print(f"[Export Complete] Saved run to: {summary_file.name}")
        return {"file": summary_file.name}


if __name__ == "__main__":
    run_rag_eval(max_cases=2)