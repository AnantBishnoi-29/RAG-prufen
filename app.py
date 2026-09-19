"""
FastAPI Backend Server for RAG Eval Suite Web Dashboard.
Serves static frontend (HTML, CSS, JS) and provides interactive RAG playground,
benchmark inspection, and golden dataset synthesis APIs.
"""

import json
from pathlib import Path
import sys
import time
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

# Ensure UTF-8 output encoding for Windows consoles
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

# Root directory of the repository
ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from components.generator import generate_answer
from components.retriever import build_retriever, retrieve, retrieve_contexts
from evals.scripts.generate_goldens import generate_golden_dataset
from prompts import SYNTHESIS_SYSTEM_PROMPT, get_all_qa_presets

load_dotenv()

app = FastAPI(
    title="RAG Eval Suite API",
    description="Backend API for RAG evaluation, live query testing, and benchmarking.",
    version="0.5.0",
)

# Enable CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static directory for CSS and JS
static_dir = ROOT / "web" / "static"
if not static_dir.exists():
    static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


# ---------------------------------------------------------------------------
# Frontend Root Route
# ---------------------------------------------------------------------------
@app.get("/", include_in_schema=False)
async def serve_index():
    index_file = ROOT / "web" / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="Frontend index.html not found.")
    return FileResponse(
        str(index_file),
        headers={
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        },
    )


# ---------------------------------------------------------------------------
# API: Available Documents
# ---------------------------------------------------------------------------
@app.get("/api/documents")
def list_documents():
    """Lists available source documents in the docs/ directory."""
    docs_dir = ROOT / "docs"
    if not docs_dir.exists():
        return []

    valid_exts = (".pdf", ".txt", ".md", ".json")
    documents = []
    for f in sorted(docs_dir.iterdir()):
        if f.is_file() and f.suffix.lower() in valid_exts:
            documents.append({
                "name": f.name,
                "path": str(Path("docs") / f.name),
                "size_kb": round(f.stat().st_size / 1024, 1),
            })
    return documents


# ---------------------------------------------------------------------------
# API: Benchmark Results Viewer
# ---------------------------------------------------------------------------
@app.get("/api/results")
def list_results():
    """Lists all saved evaluation runs in evals/results/ with summary stats."""
    results_dir = ROOT / "evals" / "results"
    if not results_dir.exists():
        return []

    runs = []
    for f in sorted(results_dir.glob("*.json"), key=lambda x: x.stat().st_mtime, reverse=True):
        try:
            with open(f, "r", encoding="utf-8") as rf:
                data = json.load(rf)

            run_type = "eval"
            if f.name.startswith("ops_run_"):
                run_type = "ops"
            elif "eval_type" in data.get("hyperparameters", {}):
                run_type = data["hyperparameters"]["eval_type"]

            cost_val = data.get("evaluationCost")
            if cost_val is None and "cost_metrics" in data:
                cost_val = data["cost_metrics"].get("total_cost_usd")

            duration_val = data.get("runDuration")
            if duration_val is None and "latency_metrics" in data:
                mean_ms = data["latency_metrics"].get("total_latency_ms", {}).get("mean", 0)
                cases_list = data.get("testCases") or data.get("per_query_results") or data.get("per_query_ops") or []
                duration_val = round((mean_ms * len(cases_list)) / 1000.0, 2) if cases_list else None

            runs.append({
                "filename": f.name,
                "type": run_type,
                "date": time.strftime("%Y-%m-%d %H:%M", time.localtime(f.stat().st_mtime)),
                "duration": duration_val,
                "cost": cost_val,
            })
        except Exception:
            continue

    return runs


@app.get("/api/results/{filename}")
def get_result_detail(filename: str):
    """Returns the full JSON content of a specific evaluation run."""
    safe_name = Path(filename).name
    target_file = ROOT / "evals" / "results" / safe_name

    if not target_file.exists():
        raise HTTPException(status_code=404, detail=f"Run file '{safe_name}' not found.")

    try:
        with open(target_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read result file: {str(e)}")


# ---------------------------------------------------------------------------
# API: Golden Datasets
# ---------------------------------------------------------------------------
@app.get("/api/datasets")
def list_datasets():
    """Lists available golden datasets in evals/datasets/."""
    datasets_dir = ROOT / "evals" / "datasets"
    if not datasets_dir.exists():
        return []

    datasets = []
    for f in sorted(datasets_dir.glob("*.json")):
        if f.name.endswith(".checkpoint.json"):
            continue
        try:
            with open(f, "r", encoding="utf-8") as df:
                items = json.load(df)
            datasets.append({
                "filename": f.name,
                "count": len(items) if isinstance(items, list) else 0,
            })
        except Exception:
            continue

    return datasets


@app.get("/api/datasets/{filename}")
def get_dataset_detail(filename: str):
    """Returns the QA pairs for a specific golden dataset."""
    safe_name = Path(filename).name
    target_file = ROOT / "evals" / "datasets" / safe_name

    if not target_file.exists():
        raise HTTPException(status_code=404, detail=f"Dataset '{safe_name}' not found.")

    try:
        with open(target_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to read dataset: {str(e)}")


# ---------------------------------------------------------------------------
# API: Prompt Presets & Templates
# ---------------------------------------------------------------------------
@app.get("/api/prompts/templates")
def get_prompt_templates():
    """Returns available QA prompt presets and default synthesis prompts."""
    return {
        "presets": get_all_qa_presets(),
        "synthesis_system_prompt": SYNTHESIS_SYSTEM_PROMPT,
    }


# ---------------------------------------------------------------------------
# API: Live Playground Query Execution
# ---------------------------------------------------------------------------
class QueryRequest(BaseModel):
    query: str
    doc_path: str = "docs/Facebooks-Corporate-Human-Rights-Policy.pdf"
    vector_store: str = "chroma"
    chunk_size: int = 500
    chunk_overlap: int = 100
    embedding_provider: str = "openai"
    loader_type: str = "auto"
    splitter_type: str = "recursive"
    top_k: int = 3
    provider: str = "openai"
    model_name: str = "gpt-4o-mini"
    prompt_template: str = "default"
    system_prompt: str | None = None
    use_reranker: bool = False
    use_hybrid: bool = False
    temperature: float = 0.0


@app.post("/api/playground/query")
def run_playground_query(req: QueryRequest):
    """
    Executes a live query through the decoupled RAG pipeline:
    1. Builds retriever for selected document & vector store (supports dense, BM25, and hybrid)
    2. Retrieves top-k chunks (with optional cross-encoder reranker)
    3. Generates grounded answer via LLM
    4. Measures and returns retrieval, generation, and total latency
    """
    if req.chunk_overlap >= req.chunk_size:
        raise HTTPException(
            status_code=400,
            detail=f"Chunk overlap ({req.chunk_overlap}) cannot be greater than or equal to chunk size ({req.chunk_size}).",
        )

    clean_query = req.query.strip()
    if not clean_query:
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    full_doc_path = ROOT / req.doc_path if not Path(req.doc_path).is_absolute() else Path(req.doc_path)
    if not full_doc_path.exists():
        raise HTTPException(status_code=404, detail=f"Document '{req.doc_path}' not found.")

    try:
        # Phase 1: Retrieval
        t0 = time.perf_counter()
        retriever = build_retriever(
            doc_path=str(full_doc_path),
            loader_type=req.loader_type,
            splitter_type=req.splitter_type,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
            embedding_provider=req.embedding_provider,
            vector_store_type=req.vector_store,
            k=req.top_k,
            use_hybrid=req.use_hybrid,
        )

        retrieved_docs = retrieve(
            retriever=retriever,
            query=clean_query,
            use_reranker=req.use_reranker,
            top_k=req.top_k,
        )
        retrieved_texts = [doc.page_content for doc in retrieved_docs]
        t1 = time.perf_counter()

        # Phase 2: Generation
        answer = generate_answer(
            query=clean_query,
            contexts=retrieved_texts,
            provider=req.provider,
            model_name=req.model_name,
            temperature=req.temperature,
            prompt_template=req.prompt_template,
            system_prompt=req.system_prompt,
        )
        t2 = time.perf_counter()

        retrieval_ms = (t1 - t0) * 1000.0
        generation_ms = (t2 - t1) * 1000.0
        total_ms = (t2 - t0) * 1000.0

        def _extract_page(meta: dict | None) -> int | None:
            if not meta:
                return None
            val = meta.get("page") if "page" in meta else meta.get("page_number")
            if val is not None:
                try:
                    return int(val)
                except (ValueError, TypeError):
                    return None
            return None

        # Structure chunks for UI display with preserved metadata and page numbers
        formatted_chunks = [
            {
                "index": i + 1,
                "content": doc.page_content,
                "chars": len(doc.page_content),
                "page": _extract_page(doc.metadata if hasattr(doc, "metadata") else None),
                "metadata": doc.metadata if hasattr(doc, "metadata") and doc.metadata else {},
            }
            for i, doc in enumerate(retrieved_docs)
        ]

        return {
            "query": clean_query,
            "answer": answer,
            "chunks": formatted_chunks,
            "retrieval_latency_ms": retrieval_ms,
            "generation_latency_ms": generation_ms,
            "total_latency_ms": total_ms,
            "metadata": {
                "vector_store": req.vector_store,
                "embedding_provider": req.embedding_provider,
                "loader_type": req.loader_type,
                "splitter_type": req.splitter_type,
                "chunk_size": req.chunk_size,
                "chunk_overlap": req.chunk_overlap,
                "use_reranker": req.use_reranker,
                "use_hybrid": req.use_hybrid,
                "top_k": req.top_k,
                "model": req.model_name,
                "prompt_template": req.prompt_template,
            },
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


# ---------------------------------------------------------------------------
# API: Synthesize Golden Dataset
# ---------------------------------------------------------------------------
class GoldenGenRequest(BaseModel):
    doc_path: str = "docs/Facebooks-Corporate-Human-Rights-Policy.pdf"
    system_prompt: str | None = None
    instruction: str = "Generate clear, representative questions and answers."
    max_goldens: int = 5
    sample_strategy: str = "stride"
    loader_type: str = "auto"
    splitter_type: str = "recursive"
    output_file: str = "custom_goldens.json"


@app.post("/api/goldens/generate")
def trigger_golden_generation(req: GoldenGenRequest):
    """Triggers prompt-styled golden dataset synthesis."""
    full_doc_path = ROOT / req.doc_path if not Path(req.doc_path).is_absolute() else Path(req.doc_path)
    if not full_doc_path.exists():
        raise HTTPException(status_code=404, detail=f"Document '{req.doc_path}' not found.")

    out_name = req.output_file if req.output_file.endswith(".json") else f"{req.output_file}.json"
    out_path = ROOT / "evals" / "datasets" / out_name

    try:
        goldens = generate_golden_dataset(
            doc_path=str(full_doc_path),
            system_prompt=req.system_prompt,
            instructions=[req.instruction],
            output_path=str(out_path),
            max_goldens=req.max_goldens,
            sample_strategy=req.sample_strategy,
            loader_type=req.loader_type,
            splitter_type=req.splitter_type,
        )
        return {
            "status": "success",
            "count": len(goldens),
            "file": out_name,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {str(e)}")


# ---------------------------------------------------------------------------
# API: Metric Registry & Selective Evaluation
# ---------------------------------------------------------------------------
from evals.registry import get_metrics_catalog
from evals.test_rag import run_rag_eval


@app.get("/api/evaluate/metrics")
def list_metrics():
    """Returns the full catalog of available evaluation metrics grouped by category."""
    return get_metrics_catalog()


class EvalRunRequest(BaseModel):
    doc_path: str = "docs/Facebooks-Corporate-Human-Rights-Policy.pdf"
    dataset_path: str = "evals/datasets/golden_dataset.json"
    vector_store: str = "chroma"
    chunk_size: int = 500
    chunk_overlap: int = 100
    embedding_provider: str = "openai"
    loader_type: str = "auto"
    splitter_type: str = "recursive"
    use_reranker: bool = False
    use_hybrid: bool = False
    top_k: int = 3
    provider: str = "openai"
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0
    prompt_template: str = "default"
    system_prompt: str | None = None
    eval_model: str = "gpt-4o-mini"
    max_cases: int = 2
    scope: str = "all"  # "all", "retriever_only", "generator_only"
    selected_metrics: list[str] | None = None


@app.post("/api/evaluate/run")
def trigger_eval_run(req: EvalRunRequest):
    """Executes a selective evaluation run based on user-chosen scope and metrics."""
    if req.chunk_overlap >= req.chunk_size:
        raise HTTPException(
            status_code=400,
            detail=f"Chunk overlap ({req.chunk_overlap}) cannot be greater than or equal to chunk size ({req.chunk_size}).",
        )

    full_doc_path = ROOT / req.doc_path if not Path(req.doc_path).is_absolute() else Path(req.doc_path)
    if not full_doc_path.exists():
        raise HTTPException(status_code=404, detail=f"Document '{req.doc_path}' not found.")

    full_dataset_path = ROOT / req.dataset_path if not Path(req.dataset_path).is_absolute() else Path(req.dataset_path)
    if not full_dataset_path.exists():
        raise HTTPException(status_code=404, detail=f"Dataset '{req.dataset_path}' not found.")

    try:
        run_result = run_rag_eval(
            doc_path=str(full_doc_path),
            dataset_path=str(full_dataset_path),
            vector_store_type=req.vector_store,
            chunk_size=req.chunk_size,
            chunk_overlap=req.chunk_overlap,
            embedding_provider=req.embedding_provider,
            loader_type=req.loader_type,
            splitter_type=req.splitter_type,
            use_reranker=req.use_reranker,
            use_hybrid=req.use_hybrid,
            top_k=req.top_k,
            provider=req.provider,
            model_name=req.model_name,
            temperature=req.temperature,
            prompt_template=req.prompt_template,
            system_prompt=req.system_prompt,
            max_cases=req.max_cases,
            eval_model=req.eval_model,
            scope=req.scope,
            selected_metrics=req.selected_metrics,
        )
        filename = run_result.get("file") if isinstance(run_result, dict) else None
        return {
            "status": "success",
            "message": f"Evaluation [{req.scope}] finished successfully.",
            "scope": req.scope,
            "filename": filename,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {str(e)}")


# ---------------------------------------------------------------------------
# Server Entrypoint
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import uvicorn
    print("\n" + "=" * 55)
    print("   ⚡ Starting RAG Eval Suite Dashboard at http://127.0.0.1:8000")
    print("=" * 55 + "\n")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)

