import json
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

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


def run_ops_eval(
    doc_path: str = "docs/Facebooks-Corporate-Human-Rights-Policy.pdf",
    dataset_path: str = "evals/datasets/golden_dataset.json",
    vector_store_type: str = "chroma",
    chunk_size: int = 500,
    chunk_overlap: int = 100,
    embedding_provider: str = "openai",
    loader_type: str = "auto",
    splitter_type: str = "recursive",
    use_reranker: bool = False,
    use_hybrid: bool = False,
    top_k: int = 3,
    provider: str = "openai",
    model_name: str = "gpt-4o-mini",
    temperature: float = 0.0,
    prompt_template: str = "default",
    max_cases: int | None = None,
) -> dict:
    """
    Benchmarks operational performance:
    - Retrieval Latency (vector store + optional reranker)
    - Generation Latency (LLM output generation)
    - End-to-End Total Latency
    - Token consumption (Prompt, Completion, Total)
    - Cost estimation (USD per query & per 1,000 queries)
    - Throughput (Tokens per second)
    """
    full_doc_path = ROOT / doc_path if not Path(doc_path).is_absolute() else Path(doc_path)
    full_dataset_path = ROOT / dataset_path if not Path(dataset_path).is_absolute() else Path(dataset_path)

    print("\n--- Running Operations & Performance Evaluation (Ops Eval) ---")
    print(f"Document: {full_doc_path.name}")
    print(f"Vector Store: {vector_store_type}")
    print(f"Hybrid Search: {'Enabled' if use_hybrid else 'Disabled'}")
    print(f"Reranker: {'Enabled (cross-encoder)' if use_reranker else 'Disabled'}")
    print(f"Top-K Chunks: {top_k}")
    print(f"Generator: {provider} / {model_name} (temperature: {temperature})")
    print(f"Prompt Template: {prompt_template}\n")
    print(f"\n[Ops Benchmark] Doc: {full_doc_path.name} | Store: {vector_store_type} | Top-K: {top_k}")

    # 1. Build live retriever
    print("Initializing vector store and retriever...")
    t_retriever_init = time.perf_counter()
    retriever = build_retriever(
        doc_path=str(full_doc_path),
        loader_type=loader_type,
        splitter_type=splitter_type,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        embedding_provider=embedding_provider,
        vector_store_type=vector_store_type,
        k=top_k,
        use_hybrid=use_hybrid,
    )
    init_retriever_ms = round((time.perf_counter() - t_retriever_init) * 1000, 2)
    print(f"Retriever initialized in {init_retriever_ms:.2f} ms.\n")

    # 2. Load dataset
    with open(full_dataset_path, encoding="utf-8") as f:
        goldens = json.load(f)

    if max_cases:
        goldens = goldens[:max_cases]
        print(f"Benchmarking with {max_cases} queries.\n")

    per_query_results = []

    print(f"{'#':<4} | {'Retrieval (ms)':<14} | {'Gen (ms)':<10} | {'Total (ms)':<10} | {'Tokens':<8} | {'Tokens/sec':<10} | {'Cost ($)':<10}")
    print("-" * 80)

    # 3. Benchmark per query
    for i, item in enumerate(goldens, 1):
        query = item["input"]

        # 3a. Benchmark Retrieval Latency
        t0 = time.perf_counter()
        contexts = retrieve_contexts(
            retriever=retriever,
            query=query,
            use_reranker=use_reranker,
            top_k=top_k,
        )
        retrieval_sec = time.perf_counter() - t0
        retrieval_ms = round(retrieval_sec * 1000, 2)

        # 3b. Benchmark Generation Latency & Tokens/Cost
        with track_cost() as cost_tracker:
            t1 = time.perf_counter()
            answer = generate_answer(
                query=query,
                contexts=contexts,
                provider=provider,
                model_name=model_name,
                temperature=temperature,
                prompt_template=prompt_template,
            )
            generation_sec = time.perf_counter() - t1

        generation_ms = round(generation_sec * 1000, 2)
        total_ms = round((retrieval_sec + generation_sec) * 1000, 2)

        p_tokens = getattr(cost_tracker, "prompt_tokens", 0)
        c_tokens = getattr(cost_tracker, "completion_tokens", 0)
        tot_tokens = getattr(cost_tracker, "total_tokens", 0)
        cost_usd = getattr(cost_tracker, "total_cost", 0.0)

        throughput = round(c_tokens / generation_sec, 2) if generation_sec > 0 else 0.0

        per_query_results.append({
            "query_index": i,
            "query": query,
            "retrieval_ms": retrieval_ms,
            "generation_ms": generation_ms,
            "total_ms": total_ms,
            "chunks_retrieved": len(contexts),
            "prompt_tokens": p_tokens,
            "completion_tokens": c_tokens,
            "total_tokens": tot_tokens,
            "tokens_per_second": throughput,
            "cost_usd": cost_usd,
        })

        print(
            f"{i:<4} | "
            f"{retrieval_ms:<14.2f} | "
            f"{generation_ms:<10.2f} | "
            f"{total_ms:<10.2f} | "
            f"{tot_tokens:<8} | "
            f"{throughput:<10.2f} | "
            f"${cost_usd:<9.6f}"
        )

    # 4. Statistical Aggregations
    retrieval_latencies = [r["retrieval_ms"] for r in per_query_results]
    generation_latencies = [r["generation_ms"] for r in per_query_results]
    total_latencies = [r["total_ms"] for r in per_query_results]
    all_throughputs = [r["tokens_per_second"] for r in per_query_results]

    total_prompt_tokens = sum(r["prompt_tokens"] for r in per_query_results)
    total_completion_tokens = sum(r["completion_tokens"] for r in per_query_results)
    grand_total_tokens = sum(r["total_tokens"] for r in per_query_results)
    grand_total_cost = sum(r["cost_usd"] for r in per_query_results)

    n_queries = len(per_query_results)
    avg_cost_per_query = (grand_total_cost / n_queries) if n_queries else 0.0
    projected_cost_1k = avg_cost_per_query * 1000

    summary = {
        "timestamp": datetime.now().isoformat(),
        "hyperparameters": {
            "eval_type": "operations_and_performance",
            "document": full_doc_path.name,
            "vector_store": vector_store_type,
            "embedding_provider": embedding_provider,
            "loader_type": loader_type,
            "splitter_type": splitter_type,
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "use_reranker": use_reranker,
            "use_hybrid": use_hybrid,
            "top_k": top_k,
            "provider": provider,
            "model_name": model_name,
            "temperature": temperature,
            "prompt_template": prompt_template,
            "total_queries_benchmarked": n_queries,
        },
        "latency_metrics": {
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
        },
        "token_metrics": {
            "total_prompt_tokens": total_prompt_tokens,
            "total_completion_tokens": total_completion_tokens,
            "grand_total_tokens": grand_total_tokens,
            "avg_tokens_per_query": round(grand_total_tokens / n_queries, 1) if n_queries else 0.0,
            "avg_throughput_tokens_per_sec": round(statistics.mean(all_throughputs), 2) if all_throughputs else 0.0,
        },
        "cost_metrics": {
            "total_cost_usd": round(grand_total_cost, 6),
            "avg_cost_per_query_usd": round(avg_cost_per_query, 6),
            "projected_cost_per_1k_queries_usd": round(projected_cost_1k, 4),
        },
        "per_query_results": per_query_results,
    }

    # 5. Print Aggregated Summary Table
    print("\n" + "=" * 80)
    print("📊 OPS & PERFORMANCE SUMMARY BENCHMARK")
    print("=" * 80)
    print(f"{'Metric':<30} | {'Mean':<12} | {'P50 (Median)':<14} | {'P95':<12}")
    print("-" * 80)
    print(
        f"{'Retrieval Latency (ms)':<30} | "
        f"{summary['latency_metrics']['retrieval_ms']['mean']:<12.2f} | "
        f"{summary['latency_metrics']['retrieval_ms']['p50']:<14.2f} | "
        f"{summary['latency_metrics']['retrieval_ms']['p95']:<12.2f}"
    )
    print(
        f"{'Generation Latency (ms)':<30} | "
        f"{summary['latency_metrics']['generation_ms']['mean']:<12.2f} | "
        f"{summary['latency_metrics']['generation_ms']['p50']:<14.2f} | "
        f"{summary['latency_metrics']['generation_ms']['p95']:<12.2f}"
    )
    print(
        f"{'Total End-to-End Latency (ms)':<30} | "
        f"{summary['latency_metrics']['total_latency_ms']['mean']:<12.2f} | "
        f"{summary['latency_metrics']['total_latency_ms']['p50']:<14.2f} | "
        f"{summary['latency_metrics']['total_latency_ms']['p95']:<12.2f}"
    )
    print("-" * 80)
    print(f"Total Tokens: {grand_total_tokens:,} (Prompt: {total_prompt_tokens:,}, Completion: {total_completion_tokens:,})")
    print(f"Average Generation Throughput: {summary['token_metrics']['avg_throughput_tokens_per_sec']} tokens/sec")
    print(f"Total Cost: ${grand_total_cost:.6f} USD (Avg: ${avg_cost_per_query:.6f} / query)")
    print(f"Projected Cost per 1,000 Queries: ${projected_cost_1k:.4f} USD")
    print("=" * 80)

    # 6. Export Benchmark JSON to evals/results/
    results_dir = ROOT / "evals" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    json_path = results_dir / f"ops_run_{timestamp}.json"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=4)

    print(f"\n[Export Complete] Benchmark saved to: {json_path.name}")
    return summary


if __name__ == "__main__":
    run_ops_eval(max_cases=2)

